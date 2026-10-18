"""temporal_reality_check.py

THE QUESTION
------------
Event-vision papers sell *temporal / asynchronous modelling* as the mechanism.
Nobody has tested that claim with the obvious control: destroy the temporal
structure and see whether accuracy moves.

Vicente-Sola et al. (Neurocomputing 2024, arXiv:2209.14915) showed DVS128
Gesture "can be solved by networks without temporal feature extraction when its
events are accumulated in frames", but that result was never propagated to the
graph / point-cloud / SSM architectures that dominate 2024-2026, and N-Caltech101's
degenerate temporality is widely *asserted* and never *measured*.

If accuracy barely drops when timestamps are destroyed, then every temporal
modelling claim evaluated on these benchmarks is unfalsifiable, and rankings on
them do not measure what the field says they measure.

THE ABLATIONS (applied to TEST data only, model weights untouched)
-----------------------------------------------------------------
  identity        : control, no change.
  time_shuffle    : randomly permute t among events. Destroys temporal ORDER,
                    preserves the (x,y) pattern and the marginal t distribution.
  time_reverse    : t -> t_max - t. Preserves order structure, reverses arrow of
                    time. A model using genuine causal dynamics should care.
  time_constant   : all t := mean(t). Destroys ALL temporal information; the
                    graph collapses to a 2-D spatial neighbourhood problem.
  polarity_shuffle: permute polarity among events. Reference control for "how
                    much does destroying a NON-temporal channel cost?" -- calibrates
                    whether the temporal drops are large or small in context.

CRITICAL: the ablation is applied BEFORE graph construction and the graph is
rebuilt, so destroying time genuinely changes what the network sees. Applying it
to a pre-built graph would leak the original temporal structure through edges.

INTERPRETATION
--------------
  drop(time_constant) ~ 0       -> benchmark cannot test temporal modelling at all
  drop(time_shuffle)  ~ 0       -> model ignores temporal ORDER (may still use extent)
  drop(time_shuffle) >> 0       -> genuine temporal dependence; benchmark is valid

Run:
    python scripts/analysis/temporal_reality_check.py \
        --config configs/ncaltech101.yaml \
        --name temporal_reality_check \
        --models baseline:results/ncaltech101/baseline/best.ckpt:fixed \
                 film_curr:results/ncaltech101/film_rate_curriculum/best.ckpt:fixed
"""
import argparse
import csv
import sys
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader as PyGDataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cegnn.utils import load_config, results_dir, load_checkpoint, get_lambda_train
from cegnn.data.dataset import EventGraphDataset
from cegnn.data.dataset import RebuildGraph


# ============================================================================
# TEMPORAL ABLATIONS  (operate on raw pos BEFORE the graph is rebuilt)
# ============================================================================

class TemporalAblation:
    """Destroy a specific channel of information in the event stream.

    pos is (N, 3) = (x, y, t); x (features) carries polarity.
    """

    def __init__(self, mode: str, seed: int = 0):
        assert mode in {"identity", "time_shuffle", "time_reverse",
                        "time_constant", "polarity_shuffle"}
        self.mode = mode
        self.seed = seed

    def __call__(self, data):
        if self.mode == "identity":
            return data

        # Per-sample deterministic generator so the ablation is reproducible
        # and identical across the models being compared.
        g = torch.Generator().manual_seed(
            self.seed + int(abs(hash(str(data.pos.shape))) % (2 ** 31))
        )
        pos = data.pos.clone()
        n = pos.size(0)

        if self.mode == "time_shuffle":
            perm = torch.randperm(n, generator=g)
            pos[:, 2] = pos[perm, 2]
        elif self.mode == "time_reverse":
            pos[:, 2] = pos[:, 2].max() - pos[:, 2]
        elif self.mode == "time_constant":
            pos[:, 2] = pos[:, 2].mean()
        elif self.mode == "polarity_shuffle":
            perm = torch.randperm(n, generator=g)
            data.x = data.x[perm]

        data.pos = pos
        return data


class AblateThenRebuild:
    """Ablation must happen BEFORE graph construction: destroying time has to
    change the graph's connectivity, otherwise the original temporal structure
    leaks in through the stored edges."""

    def __init__(self, ablation, graph_mode, cfg, lambda_train):
        self.ablation = ablation
        self.rebuild = RebuildGraph(graph_mode, cfg, lambda_train)

    def __call__(self, data):
        return self.rebuild(self.ablation(data))


# ============================================================================
# EVAL
# ============================================================================

@torch.no_grad()
def evaluate(model, loader, device):
    correct = total = 0
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        pred = logits.argmax(dim=-1).view(-1)
        y = batch.y.view(-1)
        correct += int((pred == y).sum())
        total += int(y.numel())
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--name", default="temporal_reality_check")
    ap.add_argument("--models", nargs="+", required=True,
                    help="entries name:ckpt:graph_mode")
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ablations", nargs="+",
                    default=["identity", "time_shuffle", "time_reverse",
                             "time_constant", "polarity_shuffle"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = results_dir(cfg, args.name)
    Path(outdir).mkdir(parents=True, exist_ok=True)

    processed_dir = cfg["paths"]["processed"]
    lambda_train = get_lambda_train(cfg)

    specs = []
    for m in args.models:
        name, ckpt, gmode = m.split(":")
        specs.append({"name": name, "ckpt": ckpt, "graph_mode": gmode})

    rows = []
    for spec in specs:
        model = load_checkpoint(spec["ckpt"], cfg).to(device).eval()
        base_acc = None
        for ab in args.ablations:
            ds = EventGraphDataset(
                processed_dir, args.split, cfg,
                augment=False, rebuild=True, rate_factor=1.0,
                graph_mode=spec["graph_mode"], lambda_train=lambda_train,
            )
            # Swap in ablate-then-rebuild in place of the stock rate pipeline.
            ds._rebuild_pipeline = AblateThenRebuild(
                TemporalAblation(ab, seed=args.seed),
                spec["graph_mode"], cfg, lambda_train,
            )
            loader = PyGDataLoader(ds, batch_size=args.batch_size,
                                   shuffle=False, num_workers=args.num_workers)
            acc = evaluate(model, loader, device)
            if ab == "identity":
                base_acc = acc
            drop = None if base_acc is None else (base_acc - acc) * 100
            rows.append({"model": spec["name"], "ablation": ab,
                         "acc": round(acc * 100, 2),
                         "drop_pp": None if drop is None else round(drop, 2)})
            print(f"[{spec['name']:>14s}] {ab:<18s} acc={acc*100:6.2f}%"
                  + ("" if drop is None else f"   drop={drop:+6.2f} pp"), flush=True)

    out = Path(outdir, "temporal_reality_check.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "ablation", "acc", "drop_pp"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")

    print("\n=== READ THIS ===")
    print("If drop(time_constant) is within eval noise (~0.8 pp on this pipeline),")
    print("the benchmark cannot distinguish temporal modelling from spatial")
    print("accumulation, and every temporal claim measured on it is unfalsifiable.")


if __name__ == "__main__":
    main()
