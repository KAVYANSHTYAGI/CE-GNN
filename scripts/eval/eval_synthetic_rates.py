"""eval_synthetic_rates.py

Evaluates multiple trained checkpoints across all rates and produces
the main results table.

Usage:
    python scripts/eval/eval_synthetic_rates.py --config configs/ncars.yaml \
        --models \
            baseline:results/baseline/best.ckpt:fixed \
            ra_aug_only:results/ra_aug_only/best.ckpt:fixed \
            ra_full:results/ra_full/best.ckpt:adapt_both

Each entry is `name:ckpt_path:graph_mode_at_test`.
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cegnn.utils import (
    load_config, make_loader, results_dir, evaluate,
    load_checkpoint, get_lambda_train, write_csv,
)


def parse_model_spec(spec: str):
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError(f"bad model spec '{spec}', expected name:ckpt:mode")
    return {"name": parts[0], "ckpt": parts[1], "mode": parts[2]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--models", nargs="+", required=True,
                    help="entries of form name:ckpt_path:graph_mode")
    ap.add_argument("--name", default="compare_rq3")
    ap.add_argument("--rates", type=float, nargs="+", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = results_dir(cfg, args.name)
    rates = args.rates or cfg["rate_adaptive"]["test_rates"]

    lam_train = get_lambda_train(cfg)    # reference rate; only used by the adaptive graph modes
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = []
    for spec in args.models:
        m = parse_model_spec(spec)
        print(f"\n[compare] model: {m['name']}  ckpt: {m['ckpt']}  mode@test: {m['mode']}")
        model = load_checkpoint(m["ckpt"], cfg)
        for rf in rates:
            # rebuild=True: rescale timestamps by rf, then rebuild the graph at that rate
            # (log_lambda_raw is shifted by ln rf, so FiLM models see the shifted rate)
            loader = make_loader(
                cfg, "test",
                augment=False, rebuild=True,
                rate_factor=float(rf), graph_mode=m["mode"],
                lambda_train=lam_train, shuffle=False,
            )
            r = evaluate(model, loader, device=device)
            row = {
                "model": m["name"],
                "ckpt": m["ckpt"],
                "test_graph_mode": m["mode"],
                "rate_factor": float(rf),
                "test_acc": round(r["acc"], 4),
                "test_loss": round(r["loss"], 4),
                "n": r["n"],
            }
            rows.append(row)
            print(f"  rate={rf:>5.2f}x  acc={r['acc']*100:5.2f}%")

    csv_path = out / "comparison.csv"
    write_csv(csv_path, rows)
    print(f"\n[compare] wrote {csv_path}")

    # summary: mean and worst-case per model (the two aggregates reported in the paper)
    print("\n[compare] summary (mean / worst-case across all tested rates):")
    print(f"  {'model':<20}  {'mean_acc':>10}  {'worst_acc':>10}  {'spread_pp':>10}")
    seen = []
    for spec in args.models:
        name = parse_model_spec(spec)["name"]
        if name in seen:
            continue
        seen.append(name)
        accs = [r["test_acc"] for r in rows if r["model"] == name]
        if not accs:
            continue
        mean = sum(accs) / len(accs)
        worst = min(accs)
        spread = (max(accs) - min(accs)) * 100
        print(f"  {name:<20}  {mean*100:>9.2f}%  {worst*100:>9.2f}%  {spread:>9.2f}")


if __name__ == "__main__":
    main()