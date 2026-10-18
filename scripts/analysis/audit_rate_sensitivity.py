"""audit_rate_sensitivity.py

Loads a trained baseline checkpoint and evaluates it across a set of
test-time event-rate factors. The model itself is unchanged; only the
test-time event stream is rate-stretched (timestamps rescaled), and the
graph is rebuilt at the new rate.

This is the FIRST EXPERIMENT TO RUN after baseline training. The output
csv tells us whether the project's main hypothesis is true:
    "fixed-graph event GNNs degrade across event rates."

If they don't degrade, the project pivots. Read README.md decision points.

Run:
    python scripts/analysis/audit_rate_sensitivity.py \
        --config configs/ncars.yaml \
        --ckpt   results/baseline/best.ckpt
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cegnn.utils import (
    load_config, make_loader, results_dir, evaluate,
    load_checkpoint, write_csv,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True, help="path to trained baseline ckpt")
    ap.add_argument("--name", default="audit_rq1")
    ap.add_argument("--graph_mode", default="fixed",
                    choices=["fixed", "adapt_radius", "adapt_subsample", "adapt_both"],
                    help="graph rebuild mode at test time (default: fixed)")
    ap.add_argument("--rates", type=float, nargs="+",
                    default=None,
                    help="override config test_rates (e.g. --rates 0.5 1.0 2.0)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = results_dir(cfg, args.name)
    print(f"[audit] outputs -> {out}")

    rates = args.rates or cfg["rate_adaptive"]["test_rates"]
    print(f"[audit] testing rates: {rates}")
    print(f"[audit] graph mode at test time: {args.graph_mode}")

    # adaptive modes need lambda_train; fixed mode doesn't
    lam_train = None
    if args.graph_mode in ("adapt_radius", "adapt_both"):
        from cegnn.utils import get_lambda_train
        lam_train = get_lambda_train(cfg)

    # load model once, reuse across rates
    print(f"[audit] loading checkpoint: {args.ckpt}")
    model = load_checkpoint(args.ckpt, cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = []
    for rf in rates:
        loader = make_loader(
            cfg, "test",
            augment=False,
            rebuild=True,
            rate_factor=float(rf),
            graph_mode=args.graph_mode,
            lambda_train=lam_train,
            shuffle=False,
        )
        m = evaluate(model, loader, device=device)
        row = {
            "rate_factor": float(rf),
            "graph_mode": args.graph_mode,
            "test_acc": round(m["acc"], 4),
            "test_loss": round(m["loss"], 4),
            "n_samples": m["n"],
        }
        rows.append(row)
        print(f"  rate={rf:>5.2f}x  acc={m['acc']*100:5.2f}%  loss={m['loss']:.4f}  (n={m['n']})")

    csv_path = out / f"rate_curve_{args.graph_mode}.csv"
    write_csv(csv_path, rows)
    print(f"\n[audit] wrote {csv_path}")

    # quick summary
    accs = [r["test_acc"] for r in rows]
    if accs:
        spread = (max(accs) - min(accs)) * 100
        print(f"[audit] accuracy spread across rates: {spread:.2f} pp")
        print(f"        max: {max(accs)*100:.2f}%  min: {min(accs)*100:.2f}%")
        print(f"        decision rule (see README):")
        print(f"          > 5pp  -> continue project as planned")
        print(f"          2-5pp  -> publishable, reframe toward efficiency")
        print(f"          < 2pp  -> PIVOT: rate-robustness is a non-finding")


if __name__ == "__main__":
    main()