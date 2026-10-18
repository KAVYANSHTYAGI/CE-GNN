"""efficiency.py

Counts floating-point operations per input event by hooking every nn.Linear
in a forward pass over the test loader (2 FLOP per multiply-accumulate). The
PointNet++ backbone is dominated by per-edge set-abstraction MLPs, so the
Linear-hook count captures essentially all of the compute. FiLM gating MLPs
and the RP-TTA prompt run once per sample on a scalar, so they contribute
almost nothing per event -- this script quantifies exactly how little.

Usage:
    python scripts/eval/efficiency.py --config configs/ncaltech101.yaml --n_batches 20
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cegnn.utils import load_config, make_loader
from cegnn.models import EventClassifier, FiLMRateClassifier


def attach_linear_flops(model):
    store = {"flops": 0}
    hooks = []

    def mk(mod):
        def hook(m, inp, out):
            x = inp[0]
            rows = int(np.prod(x.shape[:-1]))
            store["flops"] += 2 * rows * m.in_features * m.out_features
        return hook

    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            hooks.append(mod.register_forward_hook(mk(mod)))
    return store, hooks


def measure(model, loader, n_batches):
    model.eval()
    store, hooks = attach_linear_flops(model)
    tot_fl, tot_ev, nb = 0, 0, 0
    with torch.no_grad():
        for b in loader:
            store["flops"] = 0
            model(b)
            tot_fl += store["flops"]
            tot_ev += int(b.x.shape[0])
            nb += 1
            if nb >= n_batches:
                break
    for h in hooks:
        h.remove()
    return tot_fl, tot_ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n_batches", type=int, default=20)
    args = ap.parse_args()

    cfg = load_config(args.config)
    loader = make_loader(cfg, "test", augment=False)

    rows = {}
    for name, Cls in [("baseline", EventClassifier), ("film", FiLMRateClassifier)]:
        m = Cls(cfg)
        fl, ev = measure(m, loader, args.n_batches)
        params = sum(p.numel() for p in m.parameters())
        rows[name] = dict(params=params, fpe=fl / ev / 1e6, ev=ev)
        print(f"{name:10s} params={params:,}  {rows[name]['fpe']:.4f} MFLOP/ev "
              f"(over {ev} events)", flush=True)

    base = rows["baseline"]["fpe"]
    film = rows["film"]["fpe"]
    # RP-TTA prompt: one 1->512->512 MLP per sample. FLOPs/sample = 2*(1*512 + 512*512).
    # Per event = that / (events per sample). Use mean events/sample from the run.
    ev_per_sample = rows["film"]["ev"] / (args.n_batches * cfg["train"]["batch_size"])
    rptta_extra_fps = 2 * (1 * 512 + 512 * 512) / max(ev_per_sample, 1) / 1e6
    print("\n--- tab:efficiency placeholders ---")
    print(f"BASE_FPE  = {base:.3f}")
    print(f"FILM_FPE  = {film:.3f}")
    print(f"DFILM     = +{film - base:.3f}")
    print(f"PFILM     = +{(film - base) / base * 100:.1f}\\%")
    print(f"RPTTA_FPE = {film + rptta_extra_fps:.3f}")
    print(f"DRPTTA    = +{(film + rptta_extra_fps) - base:.3f}")
    print(f"PRPTTA    = +{((film + rptta_extra_fps) - base) / base * 100:.1f}\\%")


if __name__ == "__main__":
    main()
