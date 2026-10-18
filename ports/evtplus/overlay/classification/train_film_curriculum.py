"""train_film_curriculum.py -- FiLM-Rate curriculum on EvT+ / DVS128 Gesture.

Fourth architecture (attention/transformer paradigm) in the cross-backbone
robustness study. Reuses EvT+'s own data pipeline pieces (Stream_CLF_Dataset,
DataAugmentation, transformations.window_partition) as plain library calls,
bypassing PyTorch Lightning's Trainer/OneCycleLR coupling for a simple,
controllable two-stage curriculum loop matching the paper's protocol:
  Stage 1: train on natural-rate (rho=1.0) data only -> theta*_nat
  Stage 2: continue from theta*_nat, train on the pooled {0.5,1.0,2.0}-rho
           data (built by dvs128_to_frames_direct.py, which rescales
           timestamps BEFORE the absolute time-chunking step -- a genuine,
           faithful T_rho(E), unlike the scale-invariance traps hit with
           AEGNN/EventMamba) -> theta*_curr
  Baseline: unconditioned EvT_CLF, same data/schedule.

Evaluation: natural-rate accuracy, synthetic-rate mean/worst across
rho in {0.125,...,8.0}, and the same 7-bucket natural-rate breakdown used
for the other two architectures.
"""
import os
import sys
import glob
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import transformations
from data_generation import Stream_CLF_Dataset
from data_augmentation import DataAugmentation
from EvT_CLF import EvT_CLF
from EvT_CLF_film import EvT_CLF_FiLM

NUM_CLASSES = 10  # matches the label set found for this raw data (see EventMamba integration)
HEIGHT = WIDTH = 128
PATCH_SIZE = 8
K = 3
EVENTS_DIM = PATCH_SIZE * PATCH_SIZE * 2 * K  # 384
CHUNK_LEN_MS = 24
MT = 256000
CLIP_LENGTH_MS = 1298
TRAIN_RHOS = [0.5, 1.0, 2.0]
EVAL_RHOS = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]

DATA_PARAMS = dict(
    dataset_name='DVS128_direct', chunk_len_ms=CHUNK_LEN_MS, k=K, mT=-1, MT=MT,
    clip_length_ms=CLIP_LENGTH_MS, h_flip=False, crop_size_perc=0.75,
    rotation_angle=0, height=HEIGHT, width=WIDTH,
    patch_size=PATCH_SIZE, min_activations_per_patch=7.5, drop_token=0.15,
    patch_by_last_k=True, shift_crop=True, shift_pixels=False,
)

MODEL_PARAMS = dict(
    embed_dim=160,
    pos_encoding={'name': 'fourier', 'params': {'shape': [HEIGHT, WIDTH], 'bands': 6}},
    pos_enc_grad=True, downsample_pos_enc=PATCH_SIZE,
    token_projection={'name': 'MLP', 'params': {'init_layers': ['ff_-1_gel'], 'embed_dim': 160}},
    preproc_tokens={'name': 'MLP', 'params': {'init_layers': ['ff_-1_gel'], 'embed_dim': 160}},
    skip_conn_backbone=True,
    init_self_attention_params={'name': 'TransformerBlock', 'num_blocks': 1,
                                 'params': {'dropout': 0.1, 'att_dropout': 0.0, 'heads': 8,
                                            'cross_heads': 8, 'add_bias_kv': False, 'opt_dim': 160, 'latent_blocks': 0}},
    latent_memory_params={'attention_params': {'heads': 8, 'dropout': 0.1, 'att_dropout': 0.0,
                                                'add_bias_kv': False, 'opt_dim': 160},
                           'num_latent_vectors': 32, 'num_self_blocks': 0},
    decoder_params={'attention_params': {'heads': 8, 'dropout': 0.1, 'att_dropout': 0.0,
                                          'add_bias_kv': False, 'opt_dim': 160},
                     'num_attention_blocks': 1, 'opt_classes': NUM_CLASSES, 'softmax': True, 'opt_dim': 160},
    events_dim=EVENTS_DIM,
)


class RateDataset(torch.utils.data.Dataset):
    """Wraps Stream_CLF_Dataset to also return the per-sample lograte stored
    by dvs128_to_frames_direct.py."""
    def __init__(self, path_dataset, validation):
        self.inner = Stream_CLF_Dataset(path_dataset=path_dataset, validation=validation, **DATA_PARAMS)

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        item = self.inner[idx]
        sample_path = self.inner.samples[idx]['path']
        with open(os.path.join(sample_path, 'lograte.txt')) as f:
            lograte = float(f.read())
        item['lograte'] = torch.tensor(lograte, dtype=torch.float32)
        return item


def collate(batch):
    labels = torch.stack([b['label'] for b in batch])
    lograte = torch.stack([b['lograte'] for b in batch])
    frames = torch.stack([b['event_frames'] for b in batch])
    return {'labels': labels, 'lograte': lograte, 'event_frames': frames}


def make_loader(paths, batch_size, shuffle, validation):
    ds = torch.utils.data.ConcatDataset([RateDataset(p, validation=validation) for p in paths])
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=4,
                       collate_fn=collate, drop_last=shuffle)


def to_tokens(batch, data_aug, validation, device):
    frames = batch['event_frames'].to(device)
    if not validation:
        frames = data_aug.random_crop(frames)
    tokens, pixels = transformations.window_partition(
        frames, PATCH_SIZE, validation=validation,
        min_activations_per_patch=DATA_PARAMS['min_activations_per_patch'],
        drop_token=DATA_PARAMS['drop_token'] if not validation else 0.0,
        chunk_len_ms=CHUNK_LEN_MS, maxTime=MT,
        patch_by_last_k=DATA_PARAMS['patch_by_last_k'], reduce_tokens=True)
    return {'event_frames': (tokens, pixels)}


def run_epoch(model, loader, data_aug, device, mu_lambda, optimizer=None, use_film=True, validation=False):
    train = optimizer is not None
    model.train() if train else model.eval()
    tot, correct, tot_loss = 0, 0, 0.0
    with torch.set_grad_enabled(train):
        for batch in loader:
            y = batch['labels'].to(device)
            c = batch['lograte'].to(device) - mu_lambda
            event_data = to_tokens(batch, data_aug, validation, device)
            if train:
                optimizer.zero_grad()
            logits = model(event_data, c) if use_film else model(event_data)
            loss = F.nll_loss(logits, y) if logits.dim() == 2 and (logits <= 0).all() else F.cross_entropy(logits, y)
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
            bs = y.size(0)
            tot_loss += loss.item() * bs
            correct += (logits.argmax(-1) == y).sum().item()
            tot += bs
    return tot_loss / tot, correct / tot


def compute_mu_lambda(natural_train_dir):
    files = glob.glob(os.path.join(natural_train_dir, '*', 'lograte.txt'))
    vals = []
    for f in files:
        with open(f) as fh:
            vals.append(float(fh.read()))
    return float(np.mean(vals))


def eval_natural_synthetic(model, data_root, device, mu_lambda, use_film, batch_size, data_aug):
    nat_loader = make_loader([os.path.join(data_root, 'natural', 'test')], batch_size, shuffle=False, validation=True)
    _, nat_acc = run_epoch(model, nat_loader, data_aug, device, mu_lambda, optimizer=None, use_film=use_film, validation=True)

    synth_accs = {}
    for rho in EVAL_RHOS:
        path = os.path.join(data_root, 'natural', 'test') if rho == 1.0 else os.path.join(data_root, f'rho{rho}', 'test')
        if not os.path.isdir(path):
            continue
        loader = make_loader([path], batch_size, shuffle=False, validation=True)
        _, acc = run_epoch(model, loader, data_aug, device, mu_lambda, optimizer=None, use_film=use_film, validation=True)
        synth_accs[rho] = acc
    synth_mean = float(np.mean(list(synth_accs.values())))
    synth_worst = float(np.min(list(synth_accs.values())))
    return nat_acc, synth_mean, synth_worst, synth_accs


def eval_natural_buckets(model, data_root, device, mu_lambda, use_film, batch_size, data_aug, n_buckets=7):
    loader = make_loader([os.path.join(data_root, 'natural', 'test')], batch_size, shuffle=False, validation=True)
    model.eval()
    all_lograte, all_correct = [], []
    with torch.no_grad():
        for batch in loader:
            y = batch['labels'].to(device)
            c = batch['lograte'].to(device) - mu_lambda
            event_data = to_tokens(batch, data_aug, True, device)
            logits = model(event_data, c) if use_film else model(event_data)
            correct = (logits.argmax(-1) == y).float().cpu().numpy()
            all_lograte.append(batch['lograte'].numpy())
            all_correct.append(correct)
    lograte = np.concatenate(all_lograte)
    correct = np.concatenate(all_correct)
    order = np.argsort(lograte)
    buckets = np.array_split(order, n_buckets)
    bucket_accs = [float(correct[b].mean()) for b in buckets]
    return dict(bucket_acc=bucket_accs, bucket_mean=float(np.mean(bucket_accs)), bucket_worst=float(np.min(bucket_accs)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--epochs_stage1', type=int, default=60)
    ap.add_argument('--epochs_stage2', type=int, default=25)
    ap.add_argument('--epochs_baseline', type=int, default=60)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--out_dir', default='./runs_film_curriculum')
    ap.add_argument('--run', choices=['baseline', 'stage1', 'stage2', 'eval_all'], required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_aug = DataAugmentation(**DATA_PARAMS)
    mu_lambda = compute_mu_lambda(os.path.join(args.data_root, 'natural', 'train'))
    print(f"mu_lambda (natural train) = {mu_lambda:.4f}", flush=True)

    def new_baseline():
        return EvT_CLF(**MODEL_PARAMS).to(device)

    def new_film():
        return EvT_CLF_FiLM(**MODEL_PARAMS).to(device)

    natural_train = os.path.join(args.data_root, 'natural', 'train')
    pool_train = [os.path.join(args.data_root, 'natural', 'train')] + \
                 [os.path.join(args.data_root, f'rho{r}', 'train') for r in TRAIN_RHOS if r != 1.0]
    pool_train = [p for p in pool_train if os.path.isdir(p)]

    if args.run == 'baseline':
        model = new_baseline()
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        loader = make_loader([natural_train], args.batch_size, shuffle=True, validation=False)
        for ep in range(args.epochs_baseline):
            loss, acc = run_epoch(model, loader, data_aug, device, mu_lambda, optimizer=opt, use_film=False)
            print(f"[baseline] epoch {ep} loss={loss:.4f} acc={acc:.4f}", flush=True)
        torch.save(model.state_dict(), os.path.join(args.out_dir, 'baseline.pt'))

    elif args.run == 'stage1':
        model = new_film()
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        loader = make_loader([natural_train], args.batch_size, shuffle=True, validation=False)
        for ep in range(args.epochs_stage1):
            loss, acc = run_epoch(model, loader, data_aug, device, mu_lambda, optimizer=opt, use_film=True)
            print(f"[stage1] epoch {ep} loss={loss:.4f} acc={acc:.4f}", flush=True)
        torch.save(model.state_dict(), os.path.join(args.out_dir, 'stage1_theta_nat.pt'))

    elif args.run == 'stage2':
        model = new_film()
        model.load_state_dict(torch.load(os.path.join(args.out_dir, 'stage1_theta_nat.pt'), map_location=device))
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.3)
        loader = make_loader(pool_train, args.batch_size, shuffle=True, validation=False)
        for ep in range(args.epochs_stage2):
            loss, acc = run_epoch(model, loader, data_aug, device, mu_lambda, optimizer=opt, use_film=True)
            print(f"[stage2] epoch {ep} loss={loss:.4f} acc={acc:.4f}", flush=True)
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
            nat, smean, sworst, sdet = eval_natural_synthetic(model, args.data_root, device, mu_lambda, use_film, args.batch_size, data_aug)
            buckets = eval_natural_buckets(model, args.data_root, device, mu_lambda, use_film, args.batch_size, data_aug)
            results[name] = dict(natural=nat, synth_mean=smean, synth_worst=sworst, detail=sdet, **buckets)

        print("\n=== RESULTS (EvT+ / DVS128 Gesture) ===")
        print(f"{'method':20s} {'natural':>8s} {'synth_mean':>11s} {'synth_worst':>12s} {'bucket_worst':>13s}")
        for name, r in results.items():
            print(f"{name:20s} {r['natural']*100:8.2f} {r['synth_mean']*100:11.2f} {r['synth_worst']*100:12.2f} {r['bucket_worst']*100:13.2f}")
        print("\n=== PER-BUCKET NATURAL-RATE ACCURACY ===")
        for name, r in results.items():
            row = name.ljust(20) + "".join(f"{a*100:8.2f}" for a in r['bucket_acc'])
            print(row)
        with open(os.path.join(args.out_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
