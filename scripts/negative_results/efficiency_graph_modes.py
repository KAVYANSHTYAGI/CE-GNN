"""efficiency_graph_modes.py

For each (graph_mode, rate_factor) condition, measures on the test set:
  - avg #nodes per sample
  - avg #edges per sample
  - avg edges/node (graph density proxy)
  - avg forward-pass GPU latency (ms)

This is what we report alongside accuracy. Density-aware subsampling
is supposed to REDUCE edge count in high-rate regimes; this script
verifies that empirically.

Run:
    python scripts/negative_results/efficiency_graph_modes.py --config configs/ncars.yaml \
        --ckpt results/baseline/best.ckpt
"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cegnn.utils import (
    load_config, make_loader, results_dir,
    load_checkpoint, get_lambda_train, write_csv,
)


GRAPH_MODES = ["fixed", "adapt_radius", "adapt_subsample", "adapt_both"]


@torch.no_grad()
def measure(model, loader, device: str, n_warmup: int = 5,
            n_timed_batches: int = 50) -> dict:
    """Returns dict with graph stats and latency stats."""
    model = model.to(device).eval()

    n_nodes_total = 0
    n_edges_total = 0
    n_samples = 0
    n_batches_seen = 0
    latencies_ms = []

    for batch in loader:
        n_nodes_total += int(batch.pos.size(0))
        n_edges_total += int(batch.edge_index.size(1))
        n_samples += int(batch.num_graphs)
        n_batches_seen += 1

        batch = batch.to(device)

        if device == "cuda" and n_batches_seen <= n_warmup:
            _ = model(batch)
            torch.cuda.synchronize()
            continue

        if n_batches_seen - n_warmup > n_timed_batches:
            continue

        if device == "cuda":
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(batch)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
        else:
            t0 = time.perf_counter()
            _ = model(batch)
            t1 = time.perf_counter()
        # per-sample latency
        latencies_ms.append((t1 - t0) * 1000.0 / max(1, batch.num_graphs))

    if not latencies_ms:
        latencies_ms = [0.0]

    avg_nodes = n_nodes_total / max(1, n_samples)
    avg_edges = n_edges_total / max(1, n_samples)
    return {
        "avg_nodes_per_sample": round(avg_nodes, 1),
        "avg_edges_per_sample": round(avg_edges, 1),
        "avg_edges_per_node": round(avg_edges / max(1, avg_nodes), 3),
        "latency_ms_mean": round(sum(latencies_ms) / len(latencies_ms), 3),
        "latency_ms_median": round(
            sorted(latencies_ms)[len(latencies_ms) // 2], 3
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", default="efficiency")
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
        print(f"\n[efficiency] mode={mode}")
        for rf in rates:
            loader = make_loader(
                cfg, "test",
                augment=False, rebuild=True,
                rate_factor=float(rf), graph_mode=mode,
                lambda_train=lam_train, shuffle=False,
            )
            stats = measure(model, loader, device=device)
            row = {"graph_mode": mode, "rate_factor": float(rf), **stats}
            rows.append(row)
            print(f"  rate={rf:>5.2f}x  "
                  f"nodes={stats['avg_nodes_per_sample']:>7.1f}  "
                  f"edges={stats['avg_edges_per_sample']:>9.1f}  "
                  f"e/n={stats['avg_edges_per_node']:>5.2f}  "
                  f"lat={stats['latency_ms_mean']:>6.2f}ms")

    csv_path = out / "efficiency.csv"
    write_csv(csv_path, rows)
    print(f"\n[efficiency] wrote {csv_path}")


if __name__ == "__main__":
    main()