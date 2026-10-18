#!/usr/bin/env python3
"""train_ratemoe.py — Train RateMoE-GNN on N-Cars.

Adds learnable rate-conditioned mixture-of-experts routing on top of an
EdgeConv backbone. See cegnn/models/ratemoe.py:RateMoEGNN for the architecture.

Run:
    python scripts/negative_results/train_ratemoe.py --config configs/ncars.yaml \
        --name ratemoe --num_experts 4 --top_k 2

Ablation flags (see argparse): --no_rate_input, --random_routing,
--freeze_tau, --all_experts.
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn.pool import radius_graph

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cegnn.data.dataset import EventGraphDataset, TrainAugment
from cegnn.models import RateMoEGNN
from cegnn.utils import load_config, results_dir


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ncars.yaml")
    ap.add_argument("--name", default="ratemoe",
                    help="results subdir prefix; K and ablation flags are appended")
    ap.add_argument("--num_experts", type=int, default=4, choices=[1, 2, 4, 8])
    ap.add_argument("--top_k", type=int, default=2)
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--num_freqs", type=int, default=4)
    ap.add_argument("--alpha_lb", type=float, default=0.1)
    ap.add_argument("--beta_div", type=float, default=0.001)

    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=15,
                    help="early stop after this many epochs without val improvement")

    # Rate augmentation
    ap.add_argument("--no_rate_aug", action="store_true",
                    help="disable training-time rate augmentation")
    ap.add_argument("--rate_factors", type=float, nargs="+", default=None,
                    help="override rate factors used in augmentation; defaults to cfg")

    # Ablations
    ap.add_argument("--no_rate_input", action="store_true",
                    help="zero out lambda_local before forward (router sees no rate)")
    ap.add_argument("--random_routing", action="store_true",
                    help="freeze router with random init; no learned routing")
    ap.add_argument("--freeze_tau", action="store_true",
                    help="freeze τ_k; experts cannot adapt their temporal scale")
    ap.add_argument("--all_experts", action="store_true",
                    help="top_k := num_experts (no sparsity)")

    return ap.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# RATE AUGMENTATION (operates on a single Data object)
# ─────────────────────────────────────────────────────────────────────────────

def apply_rate_augmentation(data: Data, rate_factor: float, r: float, d_max: int):
    """In-place rate aug.

    rate_factor > 1: compress time (events come faster) — rebuild radius graph
                     with same r since time axis is now shorter.
    rate_factor < 1: subsample events (drop fraction) — keeps the spatial layout,
                     simulates slower firing.

    Updates lambda_local by adding log(rate_factor): an event in a region
    firing rate_factor× faster has λ_log → λ_log + log(rate_factor).

    rate_factor == 1: no-op.
    """
    if abs(rate_factor - 1.0) < 1e-6:
        return data

    if rate_factor < 1.0:
        # Subsample events to simulate lower rate
        n = data.x.size(0)
        keep_n = max(2, int(n * rate_factor))
        idx = torch.randperm(n)[:keep_n]
        idx, _ = idx.sort()

        data.x = data.x[idx]
        data.pos = data.pos[idx]
        if hasattr(data, "lambda_local") and data.lambda_local is not None:
            data.lambda_local = data.lambda_local[idx]

        # Rebuild edges within radius r (cheaper than remap)
        data.edge_index = radius_graph(
            data.pos, r=r, max_num_neighbors=d_max, loop=False,
        )
        if data.edge_index.numel() == 0:
            data.edge_attr = torch.zeros((0, 3), dtype=torch.float32)
        else:
            src, dst = data.edge_index
            delta = data.pos[dst] - data.pos[src]
            data.edge_attr = (delta / r).float()
    else:
        # Compress time axis to simulate higher rate; pos[:,2] /= rate_factor
        data.pos = data.pos.clone()
        data.pos[:, 2] = data.pos[:, 2] / rate_factor
        # Rebuild edges in the new t-axis
        data.edge_index = radius_graph(
            data.pos, r=r, max_num_neighbors=d_max, loop=False,
        )
        if data.edge_index.numel() == 0:
            data.edge_attr = torch.zeros((0, 3), dtype=torch.float32)
        else:
            src, dst = data.edge_index
            delta = data.pos[dst] - data.pos[src]
            data.edge_attr = (delta / r).float()

    # Update lambda_local: rate is rate_factor× higher
    if hasattr(data, "lambda_local") and data.lambda_local is not None:
        shift = float(np.log(rate_factor))
        data.lambda_local = (data.lambda_local + shift).clamp(-4.0, 4.0)

    return data


# ─────────────────────────────────────────────────────────────────────────────
# RATE-AUGMENTED + LAMBDA-AWARE DATASET WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

class _MoETrainDataset(EventGraphDataset):
    """NCars dataset that applies rate augmentation and TrainAugment per sample.

    Differs from cegnn.data.dataset.RateAugmentedDataset:
      - keeps the pre-built edge graph if rate_factor == 1 (faster path)
      - updates data.lambda_local under rate aug
    """

    def __init__(self, train_rates: List[float], r: float, d_max: int, **kwargs):
        kwargs["rebuild"] = False  # we manage transforms ourselves
        super().__init__(**kwargs)
        self.train_rates = list(train_rates)
        self.r = r
        self.d_max = d_max

    def __getitem__(self, idx: int) -> Data:
        try:
            data = torch.load(self.files[idx], weights_only=False)
        except TypeError:
            data = torch.load(self.files[idx])

        label = int("cars" in os.path.basename(self.files[idx])
                    and "background" not in os.path.basename(self.files[idx]))
        data.y = torch.tensor([label], dtype=torch.long)

        rf = random.choice(self.train_rates)
        if abs(rf - 1.0) > 1e-6:
            data = apply_rate_augmentation(data, rf, r=self.r, d_max=self.d_max)

        if self.augment_fn is not None:
            data = self.augment_fn(data)

        return data


# ─────────────────────────────────────────────────────────────────────────────
# RATE-BALANCED SAMPLER (over training clips by global rate)
# ─────────────────────────────────────────────────────────────────────────────

class RateBalancedSampler(torch.utils.data.Sampler):
    """Sample so each batch contains a mix of low- and high-rate clips.

    Computes lambda_global for every clip (single scalar per file), bins into
    quantiles, and round-robin-interleaves shuffled indices from each bin.
    """

    def __init__(self, dataset, num_bins: int = 4, cache_path: Optional[str] = None):
        self.dataset = dataset
        self.n = len(dataset)
        rates = None

        if cache_path and os.path.exists(cache_path):
            try:
                rates = np.load(cache_path)
                if len(rates) != self.n:
                    rates = None
            except Exception:
                rates = None

        if rates is None:
            print(f"[RateBalancedSampler] scanning {self.n} clips for lambda_global...")
            rates = np.zeros(self.n, dtype=np.float32)
            for i, fpath in enumerate(dataset.files):
                try:
                    try:
                        d = torch.load(fpath, weights_only=False)
                    except TypeError:
                        d = torch.load(fpath)
                    rates[i] = float(d.lambda_global.item()) \
                        if hasattr(d, "lambda_global") else 0.0
                except Exception:
                    rates[i] = 0.0
                if (i + 1) % 2000 == 0:
                    print(f"  scanned {i+1}/{self.n}")
            if cache_path:
                np.save(cache_path, rates)
                print(f"[RateBalancedSampler] cached rates → {cache_path}")

        # Quantile bin
        quantiles = np.quantile(rates, np.linspace(0, 1, num_bins + 1))
        self.bins = []
        for b in range(num_bins):
            lo = quantiles[b]
            hi = quantiles[b + 1] + (1e-9 if b == num_bins - 1 else 0.0)
            bin_idx = np.where((rates >= lo) & (rates <= hi))[0].tolist()
            if bin_idx:
                self.bins.append(bin_idx)
        print(f"[RateBalancedSampler] {len(self.bins)} bins, "
              f"sizes: {[len(b) for b in self.bins]}")

    def __iter__(self):
        per_bin = [list(np.random.permutation(b)) for b in self.bins]
        order = []
        max_len = max(len(b) for b in per_bin)
        for i in range(max_len):
            for b in per_bin:
                if i < len(b):
                    order.append(int(b[i]))
        return iter(order)

    def __len__(self):
        return self.n


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN / EVAL
# ─────────────────────────────────────────────────────────────────────────────

def _maybe_zero_rate_input(data, do_zero: bool):
    if do_zero and hasattr(data, "lambda_local"):
        data.lambda_local = torch.zeros_like(data.lambda_local)
    return data


def train_epoch(model, loader, optimizer, device, args, epoch):
    model.train()
    n_batches = 0
    sum_loss = sum_ce = sum_lb = sum_div = 0.0
    correct = total = 0

    # Anneal alpha_lb: hold for 5 epochs, then exponential decay to floor 0.01
    if epoch <= 5:
        alpha = args.alpha_lb
    else:
        alpha = max(0.01, args.alpha_lb * (0.95 ** (epoch - 5)))

    for data in loader:
        data = data.to(device)
        data = _maybe_zero_rate_input(data, args.no_rate_input)

        optimizer.zero_grad()
        logits, aux = model(data)

        y = data.y.long().view(-1)
        ce_loss = F.cross_entropy(logits, y)
        lb_loss = aux["lb_loss"]
        div_loss = aux["div_loss"]

        if args.random_routing:
            # router is frozen anyway; zero aux losses so they don't update τ etc
            lb_loss = lb_loss * 0.0
            div_loss = div_loss * 0.0

        loss = ce_loss + alpha * lb_loss + args.beta_div * div_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        n_batches += 1
        sum_loss += loss.item()
        sum_ce += ce_loss.item()
        sum_lb += float(lb_loss.item())
        sum_div += float(div_loss.item())
        pred = logits.argmax(dim=-1)
        correct += int((pred == y).sum().item())
        total += int(y.size(0))

    return {
        "loss": sum_loss / max(1, n_batches),
        "ce": sum_ce / max(1, n_batches),
        "lb": sum_lb / max(1, n_batches),
        "div": sum_div / max(1, n_batches),
        "acc": correct / max(1, total),
        "alpha_lb": alpha,
    }


@torch.no_grad()
def eval_epoch(model, loader, device, args, num_experts: int):
    model.eval()
    correct = total = 0
    gate_usage = torch.zeros(num_experts)

    for data in loader:
        data = data.to(device)
        data = _maybe_zero_rate_input(data, args.no_rate_input)
        logits, aux = model(data)
        y = data.y.long().view(-1)
        pred = logits.argmax(dim=-1)
        correct += int((pred == y).sum().item())
        total += int(y.size(0))
        # usage by # of events routed to each expert
        gate_usage += (aux["gates"] > 0).float().sum(dim=0).cpu()

    gate_pct = (gate_usage / max(1.0, gate_usage.sum().item()) * 100).tolist()
    return {"acc": correct / max(1, total), "gate_usage": gate_pct}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[RateMoE] device: {device}")

    cfg = load_config(args.config)

    # Build experiment name with K and ablations encoded
    name = f"{args.name}_K{args.num_experts}"
    if args.no_rate_input:   name += "_no_rate"
    if args.random_routing:  name += "_rand_route"
    if args.freeze_tau:      name += "_frozen_tau"
    if args.all_experts:     name += "_all_exp"
    out = results_dir(cfg, name)
    print(f"[RateMoE] outputs → {out}")

    # ── Datasets ──────────────────────────────────────────────────────────
    if args.rate_factors is not None:
        train_rates = args.rate_factors
    elif args.no_rate_aug:
        train_rates = [1.0]
    else:
        train_rates = cfg["rate_adaptive"]["train_rates"]
    print(f"[RateMoE] train rate factors: {train_rates}")

    r = cfg["graph"]["r"]
    d_max = cfg["graph"]["d_max"]

    train_ds = _MoETrainDataset(
        train_rates=train_rates, r=r, d_max=d_max,
        processed_dir=cfg["paths"]["processed"], split="train",
        cfg=cfg, augment=True,
    )
    val_ds = EventGraphDataset(
        processed_dir=cfg["paths"]["processed"], split="test",
        cfg=cfg, augment=False, rebuild=False,
    )
    print(f"[RateMoE] |train|={len(train_ds)}  |val|={len(val_ds)}")

    # Rate-balanced sampler for training (cached to results dir)
    sampler_cache = Path(cfg["paths"]["results"]) / "lambda_global_train.npy"
    sampler = RateBalancedSampler(train_ds, num_bins=4, cache_path=str(sampler_cache))

    train_loader = PyGDataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = PyGDataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    top_k = args.num_experts if args.all_experts else args.top_k
    model = RateMoEGNN(
        in_channels=2,
        hidden_dim=args.hidden_dim,
        num_classes=cfg["dataset"]["num_classes"],
        num_experts=args.num_experts,
        top_k=top_k,
        num_freqs=args.num_freqs,
    ).to(device)

    if args.freeze_tau:
        for e in model.experts:
            e.tau_raw.requires_grad_(False)
    if args.random_routing:
        for p in model.router.parameters():
            p.requires_grad_(False)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[RateMoE] params: {n_params:,}  "
          f"(top_k={top_k}/{args.num_experts}, hidden_dim={args.hidden_dim})")
    print(f"[RateMoE] initial τ: {model.get_tau_values()}")

    # ── Optimizer ─────────────────────────────────────────────────────────
    opt = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    sched = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)

    # ── Logging ───────────────────────────────────────────────────────────
    log_path = out / "train_log.csv"
    ckpt_path = out / "best.ckpt"
    last_path = out / "last.ckpt"
    header = (["epoch", "train_loss", "train_ce", "train_lb", "train_div",
               "train_acc", "val_acc", "alpha_lb", "lr"]
              + [f"tau_{k}" for k in range(args.num_experts)]
              + [f"usage_{k}" for k in range(args.num_experts)])
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(header)

    best_acc = 0.0
    epochs_since_best = 0
    print(f"\n[RateMoE] training for {args.epochs} epochs, patience={args.patience}\n")

    for epoch in range(1, args.epochs + 1):
        tm = train_epoch(model, train_loader, opt, device, args, epoch)
        vm = eval_epoch(model, val_loader, device, args, args.num_experts)
        sched.step()

        tau_vals = model.get_tau_values()
        val_acc = vm["acc"]
        cur_lr = opt.param_groups[0]["lr"]

        if val_acc > best_acc:
            best_acc = val_acc
            epochs_since_best = 0
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "val_acc": val_acc,
                "tau_values": tau_vals,
                "args": vars(args),
                "cfg": cfg,
            }, ckpt_path)
        else:
            epochs_since_best += 1

        # always save last for resumability
        torch.save({
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "val_acc": val_acc,
            "tau_values": tau_vals,
            "args": vars(args),
            "cfg": cfg,
        }, last_path)

        row = ([epoch, f"{tm['loss']:.4f}", f"{tm['ce']:.4f}",
                f"{tm['lb']:.4f}", f"{tm['div']:.4f}",
                f"{tm['acc']:.4f}", f"{val_acc:.4f}",
                f"{tm['alpha_lb']:.4f}", f"{cur_lr:.6f}"]
               + [f"{t:.3f}" for t in tau_vals]
               + [f"{u:.2f}" for u in vm["gate_usage"]])
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

        tau_str = ",".join(f"{t:.1f}" for t in tau_vals)
        usage_str = ",".join(f"{u:.0f}%" for u in vm["gate_usage"])
        print(f"ep {epoch:3d} | loss={tm['loss']:.3f} "
              f"ce={tm['ce']:.3f} lb={tm['lb']:.3f} | "
              f"train={tm['acc']:.4f} val={val_acc:.4f} "
              f"(best={best_acc:.4f}) | τ=[{tau_str}] usage=[{usage_str}]")

        if epochs_since_best >= args.patience:
            print(f"\n[RateMoE] early stop: no val improvement in "
                  f"{args.patience} epochs")
            break

    print(f"\n[RateMoE] best val_acc: {best_acc:.4f}")
    print(f"[RateMoE] checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
