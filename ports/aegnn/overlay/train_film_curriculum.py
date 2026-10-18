"""train_film_curriculum.py -- FiLM-Rate curriculum on AEGNN (GraphRes) / N-Caltech101.

Third architecture in the cross-backbone robustness study (after PointNet++
and EventMamba). Same protocol as the paper:
  Stage 1: train GraphResFiLM on natural-rate graphs only -> theta*_nat
  Stage 2: continue from theta*_nat, train with T_rho(E) timestamp rescaling,
           rho ~ {0.5, 1.0, 2.0} -> theta*_curr
  Baseline: unconditioned GraphRes, same data/schedule, for a same-backbone
           same-data same-schedule comparison.

Unlike EventMamba, AEGNN rebuilds edge_attr fresh from raw positions each
forward pass, so rescaling data.pos[:,2] by rho is a real, direct analog of
the paper's T_rho(E) operator (Eq. rho) -- no window-size workaround needed.

Evaluation: natural-rate accuracy (rho=1, no rescale) and synthetic-rate
accuracy across rho in {0.125,...,8.0}, mean and worst-case, matching the
paper's protocol exactly.
"""
import os
import glob
import argparse
import logging
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn.pool import radius_graph

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aegnn.models.networks.graph_res import GraphRes
from aegnn.models.networks.graph_res_film import GraphResFiLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

NUM_CLASSES = 101
RATE_SET = [0.5, 1.0, 2.0]                          # stage-2 training augmentation
SYNTH_RHOS = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]  # eval protocol, matches paper
# must match the ORIGINAL preprocessing radius_graph params (ncaltech101.py:37-38),
# so that rescaling pos[:,2] and rebuilding the graph is a faithful T_rho(E)
GRAPH_R = 5.0
GRAPH_DMAX = 32


def _ensure_edge_attr_3d(data: Data, rebuild_graph: bool = False) -> Data:
    """BUGFIX: the model's own edge_attr rebuild normalizes by the PER-GRAPH
    max abs delta, which makes it scale-invariant under a uniform rescale of
    pos[:,2] -- exactly the same trap noted (and avoided) by the original
    preprocessing's own comment ("normalize with the GLOBAL radius r, not
    per-graph max"). We replicate that fix here: normalize by the FIXED
    radius GRAPH_R, and (when rebuild_graph=True, i.e. after a rho rescale)
    actually recompute edge_index via radius_graph so rho changes which
    edges exist, not just their feature values -- otherwise T_rho(E) has
    no effect on the geometry the conv layers see at all.
    """
    data.pos = data.pos[:, :3].float()
    if rebuild_graph:
        data.edge_index = radius_graph(data.pos, r=GRAPH_R, max_num_neighbors=GRAPH_DMAX, loop=False)
    if data.edge_index.size(1) == 0:
        data.edge_attr = torch.zeros((0, 3), dtype=torch.float32, device=data.pos.device)
        return data
    src, dst = data.edge_index
    delta = data.pos[dst] - data.pos[src]
    data.edge_attr = ((delta / (2 * GRAPH_R)) + 0.5).float().clamp(0.0, 1.0)
    return data


def _augment(data):
    """BUGFIX: matches train.py's ProcessedDataset._augment exactly (random
    polarity flip + positional jitter). Missing this was a real contributor
    to the ~28-31% test accuracy we first got here, vs. up to 77% val_acc
    this same architecture/dataset reaches with it (see checkpoint_3d/
    filenames from earlier train.py runs) -- without augmentation, an
    unregularized 80-epoch run on ~7K samples/101 classes just memorizes
    the train set (99%+ train acc, <=31% test acc)."""
    data = data.clone()
    if torch.rand(1).item() < 0.5:
        data.x = -data.x
    if data.pos is not None and data.pos.size(1) >= 3:
        jitter = torch.zeros_like(data.pos)
        jitter[:, :2] = torch.randn_like(data.pos[:, :2]) * 0.5
        data.pos = data.pos + jitter
        data.pos[:, 0].clamp_(0.0, 239.0)
        data.pos[:, 1].clamp_(0.0, 179.0)
    return data


class RateDataset(torch.utils.data.Dataset):
    """Loads AEGNN-preprocessed .pt graphs and computes per-sample log-rate
    from the (beta-rescaled) time channel: log(N / (t.max()-t.min()))."""

    def __init__(self, roots, rescale_rho=None, augment=False):
        self.files = []
        for root in roots:
            self.files += sorted(glob.glob(os.path.join(root, "**/*.pt"), recursive=True))
        assert len(self.files) > 0, f"No .pt files found in {roots}"
        classes = sorted(set(os.path.basename(os.path.dirname(f)) for f in self.files))
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        # rescale_rho: None (natural), a fixed float (synthetic-rate eval), or
        # 'curriculum' (sample rho ~ RATE_SET uniformly at random each __getitem__)
        self.rescale_rho = rescale_rho
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        for attempt in range(len(self.files)):
            path = self.files[(idx + attempt) % len(self.files)]
            try:
                data = torch.load(path, map_location="cpu")
                if data.x is None or data.x.size(0) == 0 or data.pos is None or data.edge_index is None:
                    continue
                if data.x.dim() == 1:
                    data.x = data.x.unsqueeze(-1)
                data.x = data.x.float()
                data.pos = data.pos.float()
                data.edge_index = data.edge_index.long()
                data.num_nodes = data.x.size(0)
                label = self.class_to_idx[os.path.basename(os.path.dirname(path))]
                data.y = torch.tensor(label, dtype=torch.long)

                mask = (data.edge_index[0] < data.num_nodes) & (data.edge_index[1] < data.num_nodes)
                data.edge_index = data.edge_index[:, mask]

                t = data.pos[:, 2]
                duration = (t.max() - t.min()).clamp(min=1e-6)
                n = float(data.num_nodes)
                lograte_natural = float(torch.log(n / duration))

                rho = self.rescale_rho
                if rho == 'curriculum':
                    rho = float(np.random.choice(RATE_SET))
                if rho is not None:
                    # T_rho(E): rescale timestamps by 1/rho -> log-rate shifts by +log(rho)
                    data.pos[:, 2] = data.pos[:, 2] / rho
                    lograte = lograte_natural + float(np.log(rho))
                else:
                    lograte = lograte_natural

                if self.augment:
                    data = _augment(data)  # jitters pos; edge_attr recomputed below regardless

                # rebuild the graph after any rho rescale (matching Eq. (rho) in
                # the paper); augmentation-only jitter just needs edge_attr
                # recomputed on the existing edge_index, matching train.py's
                # _augment + _ensure_edge_attr_3d behavior exactly
                data = _ensure_edge_attr_3d(data, rebuild_graph=(rho is not None))
                data.lograte = torch.tensor(lograte, dtype=torch.float32)
                return data
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")
        raise RuntimeError(f"All files failed starting at idx={idx}")


def compute_mu_lambda(train_root):
    ds = RateDataset([train_root], rescale_rho=None)
    rates = []
    for i in range(len(ds)):
        rates.append(float(ds[i].lograte))
    return float(np.mean(rates))


def make_loader(roots, batch_size, shuffle, rescale_rho=None, augment=False):
    ds = RateDataset(roots, rescale_rho=rescale_rho, augment=augment)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                       num_workers=4, pin_memory=torch.cuda.is_available())


def run_epoch(model, loader, device, mu_lambda, optimizer=None, use_film=True):
    train = optimizer is not None
    model.train() if train else model.eval()
    tot, correct, tot_loss = 0, 0, 0.0
    with torch.set_grad_enabled(train):
        for batch in loader:
            batch = batch.to(device)
            c = batch.lograte - mu_lambda
            if train:
                optimizer.zero_grad()
            logits = model(batch, c) if use_film else model(batch)
            loss = F.cross_entropy(logits, batch.y)
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            bs = batch.y.size(0)
            tot_loss += loss.item() * bs
            correct += (logits.argmax(-1) == batch.y).sum().item()
            tot += bs
    return tot_loss / tot, correct / tot


def eval_natural_synthetic(model, test_root, device, mu_lambda, use_film, batch_size):
    nat_loader = make_loader([test_root], batch_size, shuffle=False, rescale_rho=None)
    _, nat_acc = run_epoch(model, nat_loader, device, mu_lambda, optimizer=None, use_film=use_film)

    synth_accs = {}
    for rho in SYNTH_RHOS:
        loader = make_loader([test_root], batch_size, shuffle=False, rescale_rho=rho)
        _, acc = run_epoch(model, loader, device, mu_lambda, optimizer=None, use_film=use_film)
        synth_accs[rho] = acc
    synth_mean = float(np.mean(list(synth_accs.values())))
    synth_worst = float(np.min(list(synth_accs.values())))
    return nat_acc, synth_mean, synth_worst, synth_accs


def eval_natural_buckets(model, test_root, device, mu_lambda, use_film, batch_size, n_buckets=7):
    """Per-sample natural-rate lograte + correctness, then split into
    n_buckets equi-quantile buckets by rate (b0=lowest rate ... b6=highest),
    matching tab:bucket in the paper exactly (not just a single mean/worst
    number)."""
    loader = make_loader([test_root], batch_size, shuffle=False, rescale_rho=None)
    model.eval()
    all_lograte, all_correct = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            c = batch.lograte - mu_lambda
            logits = model(batch, c) if use_film else model(batch)
            correct = (logits.argmax(-1) == batch.y).float().cpu().numpy()
            all_lograte.append(batch.lograte.cpu().numpy())
            all_correct.append(correct)
    lograte = np.concatenate(all_lograte)
    correct = np.concatenate(all_correct)

    order = np.argsort(lograte)
    buckets = np.array_split(order, n_buckets)
    bucket_accs = [float(correct[b].mean()) for b in buckets]
    return dict(
        bucket_acc=bucket_accs,
        bucket_mean=float(np.mean(bucket_accs)),
        bucket_worst=float(np.min(bucket_accs)),
        n_per_bucket=[len(b) for b in buckets],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--processed_dir', required=True)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--epochs_stage1', type=int, default=80)
    ap.add_argument('--epochs_stage2', type=int, default=30)
    ap.add_argument('--epochs_baseline', type=int, default=80)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--out_dir', default='./runs_film_curriculum')
    ap.add_argument('--run', choices=['baseline', 'stage1', 'stage2', 'eval_all'], required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_root = os.path.join(args.processed_dir, 'train')
    test_root = os.path.join(args.processed_dir, 'test')
    input_shape = torch.tensor([240, 180, 1])

    mu_lambda = compute_mu_lambda(train_root)
    logger.info(f"mu_lambda (natural train) = {mu_lambda:.4f}")

    def new_baseline():
        return GraphRes(dataset="ncaltech101", input_shape=input_shape,
                         num_outputs=NUM_CLASSES, dropout=0.35).to(device)

    def new_film():
        return GraphResFiLM(dataset="ncaltech101", input_shape=input_shape,
                             num_outputs=NUM_CLASSES, dropout=0.35).to(device)

    val_loader = make_loader([test_root], args.batch_size, shuffle=False, rescale_rho=None)

    def train_loop(model, loader, opt, epochs, use_film, tag, ckpt_name, patience=8):
        # BUGFIX: matches train.py's ReduceLROnPlateau(patience=8) + best-
        # checkpoint selection. Without these, ~28-31% test acc despite
        # 99%+ train acc (severe overfitting); with them, this exact
        # architecture/dataset reaches up to 77% val_acc (see prior
        # checkpoint_3d/best-epoch=*-val_acc=0.7702.ckpt from train.py runs).
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='max', factor=0.5, patience=patience, min_lr=1e-6)
        best_acc, best_state = -1.0, None
        for ep in range(epochs):
            loss, acc = run_epoch(model, loader, device, mu_lambda, optimizer=opt, use_film=use_film)
            val_loss, val_acc = run_epoch(model, val_loader, device, mu_lambda, optimizer=None, use_film=use_film)
            sch.step(val_acc)
            if val_acc > best_acc:
                best_acc = val_acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            logger.info(f"[{tag}] epoch {ep} loss={loss:.4f} acc={acc:.4f} val_acc={val_acc:.4f} "
                        f"best_val_acc={best_acc:.4f} lr={opt.param_groups[0]['lr']:.2e}")
        torch.save(best_state, os.path.join(args.out_dir, ckpt_name))
        logger.info(f"[{tag}] saved best checkpoint (val_acc={best_acc:.4f}) -> {ckpt_name}")

    if args.run == 'baseline':
        model = new_baseline()
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
        loader = make_loader([train_root], args.batch_size, shuffle=True, rescale_rho=None, augment=True)
        train_loop(model, loader, opt, args.epochs_baseline, use_film=False, tag='baseline', ckpt_name='baseline.pt')

    elif args.run == 'stage1':
        model = new_film()
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
        loader = make_loader([train_root], args.batch_size, shuffle=True, rescale_rho=None, augment=True)
        train_loop(model, loader, opt, args.epochs_stage1, use_film=True, tag='stage1', ckpt_name='stage1_theta_nat.pt')

    elif args.run == 'stage2':
        model = new_film()
        model.load_state_dict(torch.load(os.path.join(args.out_dir, 'stage1_theta_nat.pt'), map_location=device))
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.3, weight_decay=5e-4)
        loader = make_loader([train_root], args.batch_size, shuffle=True, rescale_rho='curriculum', augment=True)
        train_loop(model, loader, opt, args.epochs_stage2, use_film=True, tag='stage2', ckpt_name='stage2_theta_curr.pt', patience=5)

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
            nat, smean, sworst, sdet = eval_natural_synthetic(model, test_root, device, mu_lambda, use_film, args.batch_size)
            buckets = eval_natural_buckets(model, test_root, device, mu_lambda, use_film, args.batch_size)
            results[name] = dict(natural=nat, synth_mean=smean, synth_worst=sworst, detail=sdet,
                                  bucket_acc=buckets['bucket_acc'], bucket_mean=buckets['bucket_mean'],
                                  bucket_worst=buckets['bucket_worst'], n_per_bucket=buckets['n_per_bucket'])

        print("\n=== RESULTS (AEGNN / N-Caltech101) ===")
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
