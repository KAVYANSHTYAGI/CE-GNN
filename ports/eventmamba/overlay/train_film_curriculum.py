"""train_film_curriculum.py -- FiLM-Rate curriculum on EventMamba (DVS128 Gesture).

Reproduces the same protocol as the PointNet++ paper, on a real SOTA
point-based backbone:
  Stage 1: train EventMambaFiLM on natural-rate windows only (0.5s) -> theta*_nat
  Stage 2: continue from theta*_nat, train on the pooled multi-window (0.25/0.5/1.0s)
           synthetic-rate pool -> theta*_curr
  Baseline: unconditioned EventMamba (v2), trained on the same natural-rate data,
           for a same-backbone/same-data/same-schedule comparison.

Evaluation:
  natural-rate: accuracy on test_natural.h5 (0.5s windows, matches paper's protocol)
  synthetic-rate: accuracy on test_w{0.25,0.5,1.0}.h5, mean and worst-case across windows
"""
import os
import sys
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'models'))
import provider_data
from eventmamba_v2 import EventMamba
from eventmamba_film import EventMambaFiLM


class H5RateDataset(Dataset):
    def __init__(self, h5_path, max_samples=None):
        # BUGFIX: window=1.0 produced 5.2M samples vs natural's 93.8K (~55x
        # imbalance) since window count is governed by step_size, not window
        # length, but longer windows survive the n_raw>100 threshold far more
        # often. Loading the whole thing (~64GB) into RAM and training on it
        # unbalanced would both blow memory and make stage-2 massively
        # skewed toward the w=1.0 condition. When max_samples is set, take a
        # random (but on-disk-sorted, for reasonably efficient h5py fancy
        # indexing) subsample at load time instead of the full split.
        import h5py
        with h5py.File(h5_path, 'r') as f:
            n = f['label'].shape[0]
            if max_samples is not None and n > max_samples:
                idx = np.sort(np.random.choice(n, size=max_samples, replace=False))
                self.data = f['data'][idx].astype(np.float32)
                self.label = f['label'][idx].astype(np.int64)
                self.lograte = f['lograte'][idx].astype(np.float32)
            else:
                self.data = f['data'][:].astype(np.float32)
                self.label = f['label'][:].astype(np.int64)
                self.lograte = f['lograte'][:].astype(np.float32)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx], self.lograte[idx]


def make_loader(paths, batch_size, shuffle, max_samples_per_split=None):
    if isinstance(paths, str):
        paths = [paths]
    ds = torch.utils.data.ConcatDataset([H5RateDataset(p, max_samples=max_samples_per_split) for p in paths])
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle, num_workers=4)


def run_epoch(model, loader, device, mu_lambda, optimizer=None, use_film=True):
    train = optimizer is not None
    model.train() if train else model.eval()
    tot, correct, tot_loss = 0, 0, 0.0
    with torch.set_grad_enabled(train):
        for data, label, lograte in loader:
            data = data.to(device).permute(0, 2, 1)
            label = label.to(device)
            c = (lograte.to(device) - mu_lambda)
            if train:
                optimizer.zero_grad()
            logits = model(data, c) if use_film else model(data)
            loss = F.cross_entropy(logits, label)
            if train:
                loss.backward()
                optimizer.step()
            tot_loss += loss.item() * label.size(0)
            correct += (logits.argmax(-1) == label).sum().item()
            tot += label.size(0)
    return tot_loss / tot, correct / tot


def compute_mu_lambda(train_natural_path):
    import h5py
    with h5py.File(train_natural_path, 'r') as f:
        return float(np.mean(f['lograte'][:]))


def eval_natural_synthetic(model, data_dir, device, mu_lambda, use_film, batch_size):
    natural_loader = make_loader(os.path.join(data_dir, 'test_natural.h5'), batch_size, shuffle=False)
    _, nat_acc = run_epoch(model, natural_loader, device, mu_lambda, optimizer=None, use_film=use_film)

    # BUGFIX: test_w0.25.h5 (0.25s window at 0.25s non-overlapping step) only
    # had 2 samples total (only 2/10 classes represented) -- our DVS128
    # source is too sparse for that short a non-overlapping window to
    # survive the n_raw>100 threshold often. A 2-sample eval point is pure
    # noise (one earlier run showed a "0.00% worst-case" from curriculum
    # missing both), not a real robustness signal, so skip any split with
    # too few samples rather than let it silently dominate synth_worst.
    MIN_EVAL_SAMPLES = 200
    import h5py
    synth_accs = {}
    for wpath in sorted(glob.glob(os.path.join(data_dir, 'test_w*.h5'))):
        w = os.path.basename(wpath).replace('test_w', '').replace('.h5', '')
        with h5py.File(wpath, 'r') as f:
            n = f['label'].shape[0]
        if n < MIN_EVAL_SAMPLES:
            print(f"skipping synth-eval split test_w{w}.h5 ({n} samples < {MIN_EVAL_SAMPLES}, too noisy)")
            continue
        loader = make_loader(wpath, batch_size, shuffle=False)
        _, acc = run_epoch(model, loader, device, mu_lambda, optimizer=None, use_film=use_film)
        synth_accs[w] = acc
    synth_mean = float(np.mean(list(synth_accs.values()))) if synth_accs else float('nan')
    synth_worst = float(np.min(list(synth_accs.values()))) if synth_accs else float('nan')
    return nat_acc, synth_mean, synth_worst, synth_accs


def eval_natural_buckets(model, data_dir, device, mu_lambda, use_film, batch_size, n_buckets=7):
    """Per-sample natural-rate lograte + correctness, split into n_buckets
    equi-quantile buckets (b0=lowest rate...b6=highest), matching tab:bucket
    in the paper exactly, not just a single mean/worst number."""
    loader = make_loader(os.path.join(data_dir, 'test_natural.h5'), batch_size, shuffle=False)
    model.eval()
    all_lograte, all_correct = [], []
    with torch.no_grad():
        for data, label, lograte in loader:
            data = data.to(device).permute(0, 2, 1)
            label = label.to(device)
            c = lograte.to(device) - mu_lambda
            logits = model(data, c) if use_film else model(data)
            correct = (logits.argmax(-1) == label).float().cpu().numpy()
            all_lograte.append(lograte.numpy())
            all_correct.append(correct)
    lograte = np.concatenate(all_lograte)
    correct = np.concatenate(all_correct)
    order = np.argsort(lograte)
    buckets = np.array_split(order, n_buckets)
    bucket_accs = [float(correct[b].mean()) for b in buckets]
    return dict(bucket_acc=bucket_accs, bucket_mean=float(np.mean(bucket_accs)),
                bucket_worst=float(np.min(bucket_accs)), n_per_bucket=[len(b) for b in buckets])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--num_classes', type=int, default=11)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--epochs_stage1', type=int, default=100)
    ap.add_argument('--epochs_stage2', type=int, default=40)
    ap.add_argument('--epochs_baseline', type=int, default=100)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--out_dir', default='./runs_film_curriculum')
    ap.add_argument('--run', choices=['baseline', 'stage1', 'stage2', 'eval_all'], required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mu_lambda = compute_mu_lambda(os.path.join(args.data_dir, 'train_natural.h5'))
    print(f"mu_lambda (natural train) = {mu_lambda:.4f}")

    def new_baseline():
        return EventMamba(num_classes=args.num_classes).to(device)

    def new_film():
        return EventMambaFiLM(num_classes=args.num_classes).to(device)

    if args.run == 'baseline':
        model = new_baseline()
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        loader = make_loader(os.path.join(args.data_dir, 'train_natural.h5'), args.batch_size, shuffle=True)
        for ep in range(args.epochs_baseline):
            loss, acc = run_epoch(model, loader, device, mu_lambda, optimizer=opt, use_film=False)
            print(f"[baseline] epoch {ep} loss={loss:.4f} acc={acc:.4f}")
        torch.save(model.state_dict(), os.path.join(args.out_dir, 'baseline.pt'))

    elif args.run == 'stage1':
        model = new_film()
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        loader = make_loader(os.path.join(args.data_dir, 'train_natural.h5'), args.batch_size, shuffle=True)
        for ep in range(args.epochs_stage1):
            loss, acc = run_epoch(model, loader, device, mu_lambda, optimizer=opt, use_film=True)
            print(f"[stage1] epoch {ep} loss={loss:.4f} acc={acc:.4f}")
        torch.save(model.state_dict(), os.path.join(args.out_dir, 'stage1_theta_nat.pt'))

    elif args.run == 'stage2':
        model = new_film()
        model.load_state_dict(torch.load(os.path.join(args.out_dir, 'stage1_theta_nat.pt'), map_location=device))
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.3, weight_decay=1e-4)
        pool_paths = sorted(glob.glob(os.path.join(args.data_dir, 'train_w*.h5')))
        # cap each window-size split to ~natural-train scale so the pool is
        # balanced across rate conditions instead of dominated by whichever
        # window size happened to survive the n_raw>100 threshold the most
        import h5py
        with h5py.File(os.path.join(args.data_dir, 'train_natural.h5'), 'r') as f:
            natural_n = f['label'].shape[0]
        loader = make_loader(pool_paths, args.batch_size, shuffle=True, max_samples_per_split=natural_n)
        for ep in range(args.epochs_stage2):
            loss, acc = run_epoch(model, loader, device, mu_lambda, optimizer=opt, use_film=True)
            print(f"[stage2] epoch {ep} loss={loss:.4f} acc={acc:.4f}")
        torch.save(model.state_dict(), os.path.join(args.out_dir, 'stage2_theta_curr.pt'))

    elif args.run == 'eval_all':
        results = {}
        for name, ctor, path, use_film in [
            ('baseline', new_baseline, os.path.join(args.out_dir, 'baseline.pt'), False),
            ('film_no_aug', new_film, os.path.join(args.out_dir, 'stage1_theta_nat.pt'), True),
            ('film_curriculum', new_film, os.path.join(args.out_dir, 'stage2_theta_curr.pt'), True),
        ]:
            if not os.path.exists(path):
                continue
            model = ctor()
            model.load_state_dict(torch.load(path, map_location=device))
            nat, smean, sworst, sdet = eval_natural_synthetic(model, args.data_dir, device, mu_lambda, use_film, args.batch_size)
            buckets = eval_natural_buckets(model, args.data_dir, device, mu_lambda, use_film, args.batch_size)
            results[name] = dict(natural=nat, synth_mean=smean, synth_worst=sworst, detail=sdet,
                                  bucket_acc=buckets['bucket_acc'], bucket_mean=buckets['bucket_mean'],
                                  bucket_worst=buckets['bucket_worst'], n_per_bucket=buckets['n_per_bucket'])

        print("\n=== RESULTS ===")
        print(f"{'method':20s} {'natural':>8s} {'synth_mean':>11s} {'synth_worst':>12s} {'bucket_worst':>13s}")
        for name, r in results.items():
            print(f"{name:20s} {r['natural']*100:8.2f} {r['synth_mean']*100:11.2f} {r['synth_worst']*100:12.2f} {r['bucket_worst']*100:13.2f}")

        print("\n=== PER-BUCKET NATURAL-RATE ACCURACY (b0=lowest rate ... b6=highest) ===")
        n_buckets = len(next(iter(results.values()))['bucket_acc']) if results else 7
        header = "method".ljust(20) + "".join(f"b{i}".rjust(8) for i in range(n_buckets))
        print(header)
        for name, r in results.items():
            row = name.ljust(20) + "".join(f"{a*100:8.2f}" for a in r['bucket_acc'])
            print(row)

        import json
        with open(os.path.join(args.out_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
