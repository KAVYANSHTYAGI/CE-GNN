"""visualize_film.py

For a trained FiLMRateClassifier checkpoint, walk the test set and record:
  (a) per-sample bottleneck feature (the input to the head, 2*gd dim)
  (b) per-sample, per-layer learned gamma and beta vectors from each FiLM
  (c) per-sample log_lambda_raw bucket id and class label

Then produce the figures the paper needs:
  fig_gamma_by_bucket.pdf    distribution of mean(|gamma|) by FiLM layer x rate bucket
  fig_beta_by_bucket.pdf     distribution of mean(|beta|)  by FiLM layer x rate bucket
  fig_tsne_bucket.pdf        t-SNE of bottleneck features colored by rate bucket
  fig_tsne_class.pdf         t-SNE of bottleneck features colored by class (top-N classes)
  fig_separability.pdf       silhouette score (over classes) per rate bucket,
                              baseline vs curriculum

Run:
    python scripts/analysis/visualize_film.py \\
        --config configs/ncaltech101.yaml \\
        --film   results/ncaltech101/film_rate_curriculum/best.ckpt \\
        --baseline results/ncaltech101/baseline/best.ckpt \\
        --out    figures
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader as PyGDataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cegnn.utils import load_config, load_checkpoint
from cegnn.data.dataset import EventGraphDataset
from cegnn.models import EventClassifier, FiLMRateClassifier


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

@torch.no_grad()
def extract_film_signals(model: FiLMRateClassifier, loader, device: str):
    """Walk loader; for each sample collect bottleneck feature + gamma/beta per FiLM layer.

    Returns dict of np arrays:
      feat: (N, 2*gd) bottleneck features
      labels: (N,)
      log_lambda: (N,)
      gamma_k: (N, dim_k) for k in {1,2,3,g}
      beta_k:  (N, dim_k)
    """
    model = model.eval().to(device)

    feats, labels, logl = [], [], []
    gammas = {k: [] for k in (1, 2, 3, "g")}
    betas = {k: [] for k in (1, 2, 3, "g")}

    # We replicate FiLMRateClassifier.forward_to_bottleneck manually to capture
    # gamma/beta as we go.
    for batch in loader:
        batch = batch.to(device)
        rate = model._rate(batch)                      # (B, 1) centered
        x = batch.x.float(); pos = batch.pos.float(); b = batch.batch
        x, pos, b = model.sa1(x, pos, b)
        g1, β1 = _eval_film(model.film1, rate[b])      # per-NODE (we then per-graph mean)
        x = (1.0 + g1) * x + β1
        x, pos, b = model.sa2(x, pos, b)
        g2, β2 = _eval_film(model.film2, rate[b])
        x = (1.0 + g2) * x + β2
        x, pos, b = model.sa3(x, pos, b)
        g3, β3 = _eval_film(model.film3, rate[b])
        x = (1.0 + g3) * x + β3
        x = model.sa_global(x, pos, b)                 # (B, 2*gd)
        gg, βg = _eval_film(model.film_global, rate)   # per-GRAPH
        x_post = (1.0 + gg) * x + βg

        # Per-graph aggregation of per-node gamma/beta values: mean over nodes per sample
        B = int(b.max().item()) + 1 if b.numel() > 0 else 1
        for k, gpn, bpn in ((1, g1, β1), (2, g2, β2), (3, g3, β3)):
            # gpn shape (N_k, d_k), need per-sample mean over nodes
            n_per = torch.zeros(B, device=device); n_per.scatter_add_(0, b, torch.ones_like(b, dtype=torch.float))
            d = gpn.shape[1]
            g_per = torch.zeros(B, d, device=device)
            β_per = torch.zeros(B, d, device=device)
            g_per.scatter_add_(0, b.unsqueeze(-1).expand(-1, d), gpn)
            β_per.scatter_add_(0, b.unsqueeze(-1).expand(-1, d), bpn)
            n_per = n_per.clamp(min=1).unsqueeze(-1)
            gammas[k].append((g_per / n_per).cpu().numpy())
            betas[k].append((β_per / n_per).cpu().numpy())
        gammas["g"].append(gg.cpu().numpy())
        betas["g"].append(βg.cpu().numpy())

        feats.append(x_post.cpu().numpy())
        labels.append(batch.y.long().view(-1).cpu().numpy())
        logl.append(batch.log_lambda_raw.view(-1).cpu().numpy())

    return {
        "feat":   np.concatenate(feats, axis=0),
        "labels": np.concatenate(labels, axis=0),
        "log_lambda": np.concatenate(logl, axis=0),
        "gamma_1": np.concatenate(gammas[1], axis=0),
        "gamma_2": np.concatenate(gammas[2], axis=0),
        "gamma_3": np.concatenate(gammas[3], axis=0),
        "gamma_g": np.concatenate(gammas["g"], axis=0),
        "beta_1":  np.concatenate(betas[1],  axis=0),
        "beta_2":  np.concatenate(betas[2],  axis=0),
        "beta_3":  np.concatenate(betas[3],  axis=0),
        "beta_g":  np.concatenate(betas["g"],  axis=0),
    }


def _eval_film(film, cond):
    gb = film.fc2(F.gelu(film.fc1(cond)))
    return gb.chunk(2, dim=-1)


@torch.no_grad()
def extract_baseline_bottleneck(model, loader, device: str):
    """Same as above but only the bottleneck (no FiLM)."""
    model = model.eval().to(device)
    feats, labels, logl = [], [], []
    for batch in loader:
        batch = batch.to(device)
        f = model.forward_to_bottleneck(batch)         # (B, 2*gd)
        feats.append(f.cpu().numpy())
        labels.append(batch.y.long().view(-1).cpu().numpy())
        logl.append(batch.log_lambda_raw.view(-1).cpu().numpy())
    return {
        "feat":   np.concatenate(feats, axis=0),
        "labels": np.concatenate(labels, axis=0),
        "log_lambda": np.concatenate(logl, axis=0),
    }


# ============================================================================
# PLOTS
# ============================================================================

def quantile_bucket_ids(log_lambda: np.ndarray, n_buckets: int = 7) -> np.ndarray:
    order = np.argsort(log_lambda)
    bsize = math.ceil(len(log_lambda) / n_buckets)
    bid = np.empty_like(order)
    for b in range(n_buckets):
        lo, hi = b * bsize, min((b + 1) * bsize, len(log_lambda))
        if lo >= hi:
            break
        bid[order[lo:hi]] = b
    return bid


def plot_gamma_beta_by_bucket(data: dict, out_dir: Path, prefix: str = "fig"):
    """For each FiLM layer, plot the per-bucket mean of |gamma| and |beta|
    as a small-multiple bar chart."""
    import matplotlib.pyplot as plt
    bid = quantile_bucket_ids(data["log_lambda"])
    n_buckets = int(bid.max()) + 1
    layers = [("1", "Stage 1 (sa1)"), ("2", "Stage 2 (sa2)"),
              ("3", "Stage 3 (sa3)"), ("g", "Global pool")]

    for kind in ("gamma", "beta"):
        fig, axes = plt.subplots(1, 4, figsize=(14, 3), sharey=False)
        for ax, (k, lbl) in zip(axes, layers):
            vals = data[f"{kind}_{k}"]                 # (N, d_k)
            means = [np.abs(vals[bid == b]).mean() for b in range(n_buckets)]
            stds  = [np.abs(vals[bid == b]).std()  for b in range(n_buckets)]
            ax.bar(range(n_buckets), means, yerr=stds, capsize=3,
                   color="#3b7", edgecolor="black", alpha=0.85)
            ax.set_title(lbl, fontsize=11)
            ax.set_xlabel("rate bucket (low→high)")
            ax.set_xticks(range(n_buckets))
            if ax is axes[0]:
                ax.set_ylabel(f"mean | {kind} |  (per-feature)")
        fig.suptitle(f"FiLM-Curriculum: per-bucket mean |{kind}| by layer",
                     fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        path = out_dir / f"{prefix}_{kind}_by_bucket.pdf"
        fig.savefig(path, bbox_inches="tight")
        print(f"  saved {path}")
        plt.close(fig)


def plot_tsne(data: dict, out_dir: Path, prefix: str = "fig",
              top_classes: int = 10, n_per_class: int = 30):
    """t-SNE of bottleneck features colored by (a) rate bucket and (b) class."""
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    feats = data["feat"]
    labels = data["labels"]
    bid = quantile_bucket_ids(data["log_lambda"])
    print(f"  computing t-SNE on {len(feats)} samples ({feats.shape[1]} dim)...")
    proj = TSNE(n_components=2, perplexity=30, random_state=0,
                init="pca", learning_rate="auto").fit_transform(feats)

    # (a) by rate bucket
    fig, ax = plt.subplots(figsize=(6, 5))
    n_buckets = int(bid.max()) + 1
    cmap = plt.cm.viridis
    for b in range(n_buckets):
        idx = bid == b
        ax.scatter(proj[idx, 0], proj[idx, 1], s=6, alpha=0.6,
                   color=cmap(b / max(1, n_buckets - 1)),
                   label=f"b{b}")
    ax.set_title("t-SNE of curriculum bottleneck, colored by natural-rate bucket")
    ax.legend(loc="best", fontsize=8, markerscale=2)
    ax.set_xticks([]); ax.set_yticks([])
    path = out_dir / f"{prefix}_tsne_bucket.pdf"
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {path}")

    # (b) by class (top-N most populous classes)
    cls_counts = np.bincount(labels)
    top = np.argsort(-cls_counts)[:top_classes]
    keep_mask = np.isin(labels, top)
    fig, ax = plt.subplots(figsize=(6, 5))
    cmap2 = plt.cm.tab10
    for i, c in enumerate(top):
        idx = labels == c
        sel = np.where(idx)[0][:n_per_class]
        ax.scatter(proj[sel, 0], proj[sel, 1], s=8, alpha=0.8,
                   color=cmap2(i / 10), label=f"class {int(c)}")
    ax.set_title(f"t-SNE of curriculum bottleneck, top {top_classes} classes")
    ax.legend(loc="best", fontsize=7, markerscale=2)
    ax.set_xticks([]); ax.set_yticks([])
    path = out_dir / f"{prefix}_tsne_class.pdf"
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {path}")


def silhouette_per_bucket(feats: np.ndarray, labels: np.ndarray, bid: np.ndarray,
                          n_buckets: int) -> np.ndarray:
    """Class-silhouette score within each rate bucket. Returns array of shape (n_buckets,)."""
    from sklearn.metrics import silhouette_score
    out = np.zeros(n_buckets)
    for b in range(n_buckets):
        mask = bid == b
        if mask.sum() < 20:
            out[b] = np.nan; continue
        f, l = feats[mask], labels[mask]
        # silhouette requires >=2 classes with >=2 members each
        u, c = np.unique(l, return_counts=True)
        valid = u[c >= 2]
        m2 = np.isin(l, valid)
        if m2.sum() < 20 or len(valid) < 2:
            out[b] = np.nan; continue
        try:
            out[b] = silhouette_score(f[m2], l[m2], metric="cosine")
        except Exception as e:
            out[b] = np.nan
    return out


def plot_separability(curr_data: dict, base_data: dict, out_dir: Path,
                      prefix: str = "fig"):
    """Per-bucket class silhouette: baseline vs curriculum."""
    import matplotlib.pyplot as plt
    bid = quantile_bucket_ids(curr_data["log_lambda"])
    n_buckets = int(bid.max()) + 1
    # Use the same bucket assignments for both (test set is shared)
    base_bid = quantile_bucket_ids(base_data["log_lambda"])

    sil_c = silhouette_per_bucket(curr_data["feat"], curr_data["labels"],
                                  bid, n_buckets)
    sil_b = silhouette_per_bucket(base_data["feat"], base_data["labels"],
                                  base_bid, n_buckets)

    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.38
    x = np.arange(n_buckets)
    ax.bar(x - width/2, sil_b, width, color="#888", label="EventClassifier baseline")
    ax.bar(x + width/2, sil_c, width, color="#3b7", label="FiLM-Rate (curriculum)")
    ax.set_xlabel("Natural-rate bucket (low → high)")
    ax.set_ylabel("Class silhouette (cosine)")
    ax.set_title("Class separability in bottleneck features, per natural-rate bucket")
    ax.set_xticks(x); ax.set_xticklabels([f"b{i}" for i in range(n_buckets)])
    ax.legend()
    ax.axhline(0, color="black", linewidth=0.5)
    fig.tight_layout()
    path = out_dir / f"{prefix}_separability.pdf"
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {path}")
    return sil_b, sil_c


def plot_film_analysis(cdata: dict, sil_b: np.ndarray, sil_c: np.ndarray,
                       out_dir: Path, prefix: str = "fig"):
    """Combined 2-panel figure for the paper (fig:film).

    Left  : mean |gamma| (solid) and |beta| (dashed) per FiLM layer across
            natural-rate buckets.
    Right : per-bucket class silhouette, baseline vs curriculum.
    """
    import matplotlib.pyplot as plt
    bid = quantile_bucket_ids(cdata["log_lambda"])
    n_buckets = int(bid.max()) + 1
    x = np.arange(n_buckets)
    layers = [("1", "sa1"), ("2", "sa2"), ("3", "sa3"), ("g", "global")]
    colors = {"1": "#1f77b4", "2": "#2ca02c", "3": "#d62728", "g": "#9467bd"}

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))

    # --- Left: |gamma| solid, |beta| dashed, per layer across buckets ---
    for k, lbl in layers:
        g = np.array([np.abs(cdata[f"gamma_{k}"][bid == b]).mean()
                      for b in range(n_buckets)])
        be = np.array([np.abs(cdata[f"beta_{k}"][bid == b]).mean()
                       for b in range(n_buckets)])
        axL.plot(x, g, "-o", color=colors[k], ms=4, lw=1.6, label=f"{lbl} $|\\gamma|$")
        axL.plot(x, be, "--s", color=colors[k], ms=3, lw=1.2, alpha=0.8,
                 label=f"{lbl} $|\\beta|$")
    axL.set_xlabel("natural-rate bucket (low → high)")
    axL.set_ylabel("mean per-feature modulation")
    axL.set_title("FiLM modulation by layer and rate", fontsize=11)
    axL.set_xticks(x); axL.set_xticklabels([f"b{i}" for i in range(n_buckets)])
    axL.legend(fontsize=7, ncol=2, loc="upper left", framealpha=0.9)
    axL.grid(True, alpha=0.25)

    # --- Right: silhouette per bucket, baseline vs curriculum ---
    width = 0.38
    axR.bar(x - width/2, sil_b, width, color="#888",
            label="EventClassifier baseline")
    axR.bar(x + width/2, sil_c, width, color="#3b7",
            label="FiLM-Rate (curriculum)")
    axR.axhline(0, color="black", linewidth=0.5)
    axR.set_xlabel("natural-rate bucket (low → high)")
    axR.set_ylabel("class silhouette (cosine)")
    axR.set_title("Bottleneck class separability", fontsize=11)
    axR.set_xticks(x); axR.set_xticklabels([f"b{i}" for i in range(n_buckets)])
    axR.legend(fontsize=8, loc="best", framealpha=0.9)
    axR.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    path = out_dir / f"{prefix}_film_analysis.pdf"
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {path}")


# ============================================================================
# DRIVER
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--film", required=True, help="path to film_rate_curriculum/best.ckpt")
    ap.add_argument("--baseline", required=True, help="path to baseline/best.ckpt")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    ds = EventGraphDataset(processed_dir=cfg["paths"]["processed"], split="test",
                           cfg=cfg, augment=False, rebuild=False)
    loader = PyGDataLoader(ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=cfg["train"]["num_workers"])
    print(f"[viz] test set: {len(ds)} samples on {device}")

    # 1) Curriculum FiLM analysis
    print("[viz] loading curriculum checkpoint")
    curr = load_checkpoint(args.film, cfg)
    assert isinstance(curr, FiLMRateClassifier), \
        f"expected FiLMRateClassifier, got {type(curr).__name__}"
    print("[viz] extracting FiLM signals from curriculum...")
    cdata = extract_film_signals(curr, loader, device)
    print(f"[viz]   feat shape {cdata['feat'].shape}, "
          f"log_lambda range [{cdata['log_lambda'].min():.2f}, {cdata['log_lambda'].max():.2f}]")
    plot_gamma_beta_by_bucket(cdata, out_dir, prefix="fig")
    plot_tsne(cdata, out_dir, prefix="fig")

    # 2) Baseline bottleneck for comparison
    print("[viz] loading baseline checkpoint")
    base = load_checkpoint(args.baseline, cfg)
    print("[viz] extracting baseline bottleneck...")
    bdata = extract_baseline_bottleneck(base, loader, device)
    sil_b, sil_c = plot_separability(cdata, bdata, out_dir, prefix="fig")
    print(f"[viz] silhouette per-bucket baseline:   {sil_b}")
    print(f"[viz] silhouette per-bucket curriculum: {sil_c}")
    print(f"[viz] mean silhouette baseline:    {np.nanmean(sil_b):.4f}")
    print(f"[viz] mean silhouette curriculum:  {np.nanmean(sil_c):.4f}")

    # Combined 2-panel figure used in the paper (fig:film)
    plot_film_analysis(cdata, sil_b, sil_c, out_dir, prefix="fig")

    # 3) Also write a small CSV with the per-bucket gamma/beta means for paper inspection
    import csv
    csv_path = out_dir / "film_signals_by_bucket.csv"
    bid = quantile_bucket_ids(cdata["log_lambda"])
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bucket", "layer", "kind", "mean_abs_value"])
        for b in range(int(bid.max()) + 1):
            for k in (1, 2, 3, "g"):
                for kind in ("gamma", "beta"):
                    v = cdata[f"{kind}_{k}"][bid == b]
                    w.writerow([b, k, kind, float(np.abs(v).mean())])
    print(f"[viz] wrote {csv_path}")

    # Per-bucket silhouette CSV so fig_film_analysis is reproducible without a GPU
    sil_path = out_dir / "separability_by_bucket.csv"
    with open(sil_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bucket", "silhouette_baseline", "silhouette_curriculum"])
        for b in range(int(bid.max()) + 1):
            w.writerow([b, float(sil_b[b]), float(sil_c[b])])
    print(f"[viz] wrote {sil_path}")


if __name__ == "__main__":
    main()
