"""make_teaser.py

Two-panel teaser (fig:pareto in the paper), CPU-only, no model forward needed.

  Left:  histogram of per-sample log_lambda over the N-Caltech101 TEST set,
         with the 101 per-class medians overlaid as a rug. Annotates the ~26x
         natural-rate span. Data: data.log_lambda_raw + data.y in each test .pt.

  Right: scatter of synthetic-rate mean accuracy (x) vs natural-rate mean
         accuracy (y) for every method in tab:main, with the two "ours"
         frontier points (FiLM-Rate curriculum, RA-TTA on FiLM curr.)
         highlighted. Numbers are taken verbatim from tab:main so the figure
         and table cannot disagree.

Run:
    python scripts/analysis/make_teaser.py \
        --config configs/ncaltech101.yaml \
        --out figures
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# tab:main numbers: (name, nat_mean, synth_mean, family). family in
# {plain, invariance, ours, ours_frontier}. The two frontier points are the
# headline curriculum and its RA-TTA extension.
TAB_MAIN = [
    ("Baseline",                 60.9, 16.0, "plain"),
    ("Baseline widehead",        62.0, 16.0, "plain"),
    ("RateConditioned (head)",   60.0, 17.5, "plain"),
    ("FiLM-Global-Only",         60.3, 16.4, "plain"),
    ("R-Canon on baseline",      44.7, 25.2, "plain"),
    ("RA-TTA on baseline",       40.2, 30.5, "plain"),
    ("ra-aug-only",              60.3, 47.0, "invariance"),
    ("FiLM-Rate (ra-aug)",       61.7, 47.4, "invariance"),
    ("RA-TTA on ra-aug-only",    59.0, 52.4, "invariance"),
    ("RA-TTA on FiLM (ra-aug)",  60.3, 54.3, "invariance"),
    ("FiLM-Rate (no aug)",       64.4, 20.4, "ours"),
    ("FiLM-Rate (curriculum)",   67.2, 50.3, "ours_frontier"),
    ("RA-TTA on FiLM (curr.)",   64.5, 57.5, "ours_frontier"),
]

FAMILY_STYLE = {
    "plain":         dict(color="#9e9e9e", marker="o", label="normalize / plain"),
    "invariance":    dict(color="#1f77b4", marker="s", label="rate-invariance"),
    "ours":          dict(color="#d62728", marker="^", label="FiLM-Rate (ours)"),
    "ours_frontier": dict(color="#d62728", marker="*", label="ours, headline"),
}


def collect_log_lambda(processed_test_dir: str):
    files = sorted(glob.glob(os.path.join(processed_test_dir, "*.pt")))
    if not files:
        raise FileNotFoundError(f"no .pt files in {processed_test_dir}")
    ll, ys = [], []
    for f in files:
        d = torch.load(f)
        ll.append(float(d.log_lambda_raw))
        ys.append(int(d.y))
    return np.asarray(ll), np.asarray(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="(unused; kept for symmetry)")
    ap.add_argument("--processed_test",
                    default="$CEGNN_DATA_ROOT/"
                            "ncaltech101_v2_processed/test")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ll, ys = collect_log_lambda(args.processed_test)
    classes = np.unique(ys)
    class_medians = np.array([np.median(ll[ys == c]) for c in classes])
    span = float(np.exp(ll.max() - ll.min()))
    med_span = float(class_medians.max() - class_medians.min())
    print(f"[teaser] log_lambda range [{ll.min():.2f}, {ll.max():.2f}] "
          f"=> {span:.1f}x span; per-class median span {med_span:.2f} nats; "
          f"n={len(ll)} over {len(classes)} classes")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))

    # ---- Left: log-rate histogram + per-class median rug ----
    axL.hist(ll, bins=40, color="#4c72b0", alpha=0.85, edgecolor="white",
             linewidth=0.3)
    ymax = axL.get_ylim()[1]
    axL.vlines(class_medians, 0, ymax * 0.16, color="#c44e52", alpha=0.55,
               linewidth=0.8)
    axL.text(0.02, 0.95,
             f"natural span ${span:.0f}\\times$\nper-class median span "
             f"{med_span:.2f} nats",
             transform=axL.transAxes, va="top", ha="left", fontsize=10,
             bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    axL.set_xlabel(r"per-sample $\log\hat{\lambda}$ (nats)")
    axL.set_ylabel("count")
    axL.set_title("N-Caltech101 natural event rate", fontsize=11)

    # ---- Right: synth vs nat mean accuracy scatter ----
    seen = set()
    for name, nat, synth, fam in TAB_MAIN:
        st = FAMILY_STYLE[fam]
        lbl = st["label"] if st["label"] not in seen else None
        seen.add(st["label"])
        size = 230 if fam == "ours_frontier" else 70
        edge = "black" if fam == "ours_frontier" else "none"
        axR.scatter(synth, nat, s=size, c=st["color"], marker=st["marker"],
                    edgecolors=edge, linewidths=0.8, label=lbl, zorder=3,
                    alpha=0.95)
    # label the two headline points
    for name, nat, synth, fam in TAB_MAIN:
        if fam == "ours_frontier":
            axR.annotate(name.replace("RA-TTA on FiLM (curr.)", "RA-TTA+curr.")
                         .replace("FiLM-Rate (curriculum)", "curriculum"),
                         (synth, nat), textcoords="offset points",
                         xytext=(8, -4), fontsize=8.5)
    axR.set_xlabel("synthetic-rate mean accuracy (%)")
    axR.set_ylabel("natural-rate mean accuracy (%)")
    axR.set_title("Specialization vs. robustness frontier", fontsize=11)
    axR.legend(fontsize=8, loc="lower right", framealpha=0.9)
    axR.grid(True, alpha=0.25)

    fig.tight_layout()
    path = out / "fig_teaser.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[teaser] wrote {path}")


if __name__ == "__main__":
    main()
