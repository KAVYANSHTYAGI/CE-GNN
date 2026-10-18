"""eval_rptta.py

Loads a trained RPTTAClassifier (base frozen + rate-prompt trained), evaluates
at each rate factor. Optionally applies a TENT-style entropy-min step over the
prompt per batch before predicting.

Run:
    python scripts/eval/eval_rptta.py \
        --config configs/ncaltech101.yaml \
        --base results/ncaltech101/ra_aug_only/best.ckpt \
        --rptta results/ncaltech101/rptta_on_ra_aug_only/best.ckpt \
        --name rptta_eval \
        --rates 0.125 0.25 0.5 1.0 2.0 4.0 8.0 \
        --tent              # optional: per-batch entropy step
"""
import argparse
import csv
import sys
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader as PyGDataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cegnn.utils import load_config, make_loader, results_dir, get_lambda_train
from cegnn.models import EventClassifier
from cegnn.adapt.rp_tta import (
    RPTTAClassifier, tent_step, snapshot_prompt, restore_prompt,
)


def evaluate_rptta(model, loader, device, do_tent: bool, tent_lr: float):
    """Eval with optional per-batch TENT step. Prompt is reset between batches
    so each batch's TENT update is isolated (true per-batch adaptation)."""
    model.to(device)
    model.eval()

    snap = snapshot_prompt(model)
    n_correct = 0
    n_total = 0
    loss_sum = 0.0
    for batch in loader:
        batch = batch.to(device)
        if do_tent:
            tent_step(model, batch, lr=tent_lr)
        with torch.no_grad():
            logits = model(batch)
            y = batch.y.long().view(-1)
            n_correct += (logits.argmax(1) == y).sum().item()
            n_total += y.numel()
            loss_sum += torch.nn.functional.cross_entropy(
                logits, y, reduction="sum"
            ).item()
        if do_tent:
            restore_prompt(model, snap)

    return {
        "acc": n_correct / max(1, n_total),
        "loss": loss_sum / max(1, n_total),
        "n": n_total,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--rptta", required=True)
    ap.add_argument("--name", default="rptta_eval")
    ap.add_argument("--rates", type=float, nargs="+", default=None)
    ap.add_argument("--tent", action="store_true",
                    help="apply per-batch entropy step before predicting")
    ap.add_argument("--tent_lr", type=float, default=1.0e-3)
    ap.add_argument("--graph_mode", default="fixed",
                    choices=["fixed", "adapt_radius", "adapt_subsample", "adapt_both"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = results_dir(cfg, args.name)
    rates = args.rates or cfg["rate_adaptive"]["test_rates"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # build model: base weights then overlay rptta ckpt (which contains both)
    base_ckpt = torch.load(args.base, map_location="cpu")
    base_state = base_ckpt.get("state_dict", base_ckpt)
    model = RPTTAClassifier(cfg, base_state_dict=base_state)
    rptta_ckpt = torch.load(args.rptta, map_location="cpu")
    rptta_state = rptta_ckpt.get("state_dict", rptta_ckpt)
    model.load_state_dict(rptta_state, strict=True)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[rptta_eval] params trainable={n_train:,}  total={n_total:,}")
    print(f"[rptta_eval] alpha = {model.prompt.alpha.item():.4f}")
    print(f"[rptta_eval] graph_mode={args.graph_mode}  tent={args.tent}")

    lam_train = None
    if args.graph_mode in ("adapt_radius", "adapt_both"):
        lam_train = get_lambda_train(cfg)

    rows = []
    for rf in rates:
        loader = make_loader(
            cfg, "test",
            augment=False, rebuild=True,
            rate_factor=float(rf), graph_mode=args.graph_mode,
            lambda_train=lam_train, shuffle=False,
        )
        r = evaluate_rptta(model, loader, device,
                           do_tent=args.tent, tent_lr=args.tent_lr)
        rows.append({
            "model": f"rptta{'_tent' if args.tent else ''}",
            "rate_factor": float(rf),
            "test_acc": round(r["acc"], 4),
            "test_loss": round(r["loss"], 4),
            "n": r["n"],
        })
        print(f"  rate={rf:>5.3f}x  acc={r['acc']*100:5.2f}%")

    csv_path = out / "rptta_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[rptta_eval] wrote {csv_path}")

    accs = [r["test_acc"] for r in rows]
    print(f"\n[rptta_eval] summary:")
    print(f"  mean={sum(accs)/len(accs)*100:.2f}%  worst={min(accs)*100:.2f}%  "
          f"spread={(max(accs)-min(accs))*100:.2f}pp")


if __name__ == "__main__":
    main()
