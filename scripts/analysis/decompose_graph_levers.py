"""decompose_graph_levers.py

Given a baseline checkpoint, evaluate it under FOUR different test-time
graph-construction modes, across all rate factors:

    fixed              - baseline (no adaptation)
    adapt_radius       - only R adapts to rate
    adapt_subsample    - only subsampling adapts (density-aware)
    adapt_both         - both adapt (full RA-GNN at test time)

This isolates each lever's contribution. If, say, adapt_radius alone closes
most of the gap, we know R is the dominant variable and can simplify the
final RA-GNN. If neither alone helps but adapt_both does, they're synergistic.

Output: a single CSV with rate x mode -> accuracy, plus a printed pivot table.

Run:
    python scripts/analysis/decompose_graph_levers.py --config configs/ncars.yaml --ckpt ...
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


GRAPH_MODES = ["fixed", "adapt_radius", "adapt_subsample", "adapt_both"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", default="decompose_rq2")
    ap.add_argument("--rates", type=float, nargs="+", default=None)
    ap.add_argument("--modes", nargs="+", default=GRAPH_MODES)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = results_dir(cfg, args.name)
    rates = args.rates or cfg["rate_adaptive"]["test_rates"]

    lam_train = get_lambda_train(cfg)
    model = load_checkpoint(args.ckpt, cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = []
    for mode in args.modes:
        print(f"\n[decompose] mode = {mode}")
        for rf in rates:
            loader = make_loader(
                cfg, "test",
                augment=False, rebuild=True,
                rate_factor=float(rf),
                graph_mode=mode,
                lambda_train=lam_train,
                shuffle=False,
            )
            m = evaluate(model, loader, device=device)
            row = {
                "graph_mode": mode,
                "rate_factor": float(rf),
                "test_acc": round(m["acc"], 4),
                "test_loss": round(m["loss"], 4),
                "n_samples": m["n"],
            }
            rows.append(row)
            print(f"  rate={rf:>5.2f}x  acc={m['acc']*100:5.2f}%")

    csv_path = out / "decomposition.csv"
    write_csv(csv_path, rows)
    print(f"\n[decompose] wrote {csv_path}")

    # print pivot
    print("\n[decompose] pivot table (rows = mode, cols = rate, cells = test_acc%):")
    header = "                  " + "  ".join(f"{r:>6.2f}x" for r in rates)
    print(header)
    for mode in args.modes:
        accs = [next(r["test_acc"] for r in rows
                     if r["graph_mode"] == mode and r["rate_factor"] == rf)
                for rf in rates]
        cells = "  ".join(f"{a*100:>6.2f}" for a in accs)
        print(f"  {mode:<16}  {cells}")


if __name__ == "__main__":
    main()