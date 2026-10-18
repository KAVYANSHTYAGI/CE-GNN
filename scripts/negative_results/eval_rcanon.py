"""eval_rcanon.py

Evaluates a trained CanonicalizedClassifier in two modes:
  (1) Synthetic-rate grid (back-compat with prior tables)
  (2) Natural-rate buckets (no synthetic shift; bucket by log_lambda_raw)

Run:
    python scripts/negative_results/eval_rcanon.py \
        --config configs/ncaltech101.yaml \
        --base   results/ncaltech101/ra_aug_only/best.ckpt \
        --rcanon results/ncaltech101/rcanon_on_v2_ra_aug_only/best.ckpt \
        --name   rcanon_eval
"""
import argparse
import csv
import sys
import math
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader as PyGDataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cegnn.utils import load_config, make_loader, results_dir
from cegnn.data.dataset import EventGraphDataset
from cegnn.adapt.rcanon import CanonicalizedClassifier


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval().to(device)
    n_correct = 0
    n_total = 0
    log_rho_sum = 0.0
    for batch in loader:
        batch = batch.to(device)
        logits, log_rho = model(batch)
        y = batch.y.long().view(-1)
        n_correct += (logits.argmax(1) == y).sum().item()
        n_total += y.numel()
        log_rho_sum += log_rho.sum().item()
    return {
        "acc": n_correct / max(1, n_total),
        "n": n_total,
        "mean_log_rho": log_rho_sum / max(1, n_total),
    }


def collect_per_sample_meta(processed_dir, split):
    files = sorted(Path(processed_dir, split).glob("*.pt"))
    meta = []
    for f in files:
        try:
            d = torch.load(f, weights_only=False)
        except TypeError:
            d = torch.load(f)
        meta.append({
            "file": str(f),
            "label": int(d.y.view(-1)[0].item()),
            "log_lambda": float(d.log_lambda_raw.view(-1)[0].item()),
        })
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--rcanon", required=True)
    ap.add_argument("--name", default="rcanon_eval")
    ap.add_argument("--rates", type=float, nargs="+",
                    default=[0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
    ap.add_argument("--n_buckets", type=int, default=7)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = results_dir(cfg, args.name)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_state = torch.load(args.base, map_location="cpu")
    base_state = base_state.get("state_dict", base_state)
    model = CanonicalizedClassifier(cfg, base_state_dict=base_state)
    rc_state = torch.load(args.rcanon, map_location="cpu")
    rc_state = rc_state.get("state_dict", rc_state)
    model.load_state_dict(rc_state, strict=True)
    print(f"[rcanon_eval] params trainable={sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ---- (1) synthetic-rate grid (for comparison with prior tables) ----
    print(f"\n[rcanon_eval] synthetic-rate eval:")
    sr_rows = []
    for rf in args.rates:
        loader = make_loader(
            cfg, "test", augment=False, rebuild=True,
            rate_factor=float(rf), graph_mode="fixed", shuffle=False,
        )
        r = evaluate(model, loader, device)
        sr_rows.append({"mode": "synthetic", "rate_factor": float(rf),
                        "test_acc": round(r["acc"], 4),
                        "mean_log_rho": round(r["mean_log_rho"], 4),
                        "n": r["n"]})
        print(f"  rate={rf:>5.3f}x  acc={r['acc']*100:5.2f}%  "
              f"<log_rho>={r['mean_log_rho']:+.3f}")

    # ---- (2) natural-rate buckets ----
    print(f"\n[rcanon_eval] natural-rate buckets:")
    meta = collect_per_sample_meta(cfg["paths"]["processed"], "test")
    meta = sorted(meta, key=lambda m: m["log_lambda"])
    n = len(meta)
    bs = math.ceil(n / args.n_buckets)
    nr_rows = []
    for b in range(args.n_buckets):
        lo_i, hi_i = b * bs, min((b + 1) * bs, n)
        if lo_i >= hi_i:
            break
        bucket_files = [m["file"] for m in meta[lo_i:hi_i]]
        lo_lam, hi_lam = meta[lo_i]["log_lambda"], meta[hi_i - 1]["log_lambda"]

        ds = EventGraphDataset(
            processed_dir=cfg["paths"]["processed"], split="test", cfg=cfg,
            augment=False, rebuild=False,
        )
        ds.files = bucket_files
        loader = PyGDataLoader(ds, batch_size=cfg["train"]["batch_size"],
                               shuffle=False, num_workers=cfg["train"]["num_workers"])
        r = evaluate(model, loader, device)
        row = {"mode": "natural_bucket", "bucket": b,
               "log_lambda_lo": round(lo_lam, 4),
               "log_lambda_hi": round(hi_lam, 4),
               "n": r["n"],
               "test_acc": round(r["acc"], 4),
               "mean_log_rho": round(r["mean_log_rho"], 4)}
        nr_rows.append(row)
        print(f"  bucket {b} log_lambda in [{lo_lam:.3f},{hi_lam:.3f}] "
              f"n={r['n']}  acc={r['acc']*100:5.2f}%  "
              f"<log_rho>={r['mean_log_rho']:+.3f}")

    # save both
    with open(out / "rcanon_synthetic.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sr_rows[0].keys()))
        w.writeheader(); w.writerows(sr_rows)
    with open(out / "rcanon_natural.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(nr_rows[0].keys()))
        w.writeheader(); w.writerows(nr_rows)
    print(f"\n[rcanon_eval] wrote {out}/rcanon_synthetic.csv  and  rcanon_natural.csv")

    accs_s = [r["test_acc"] for r in sr_rows]
    accs_n = [r["test_acc"] for r in nr_rows]
    print(f"\nsynthetic summary: mean={sum(accs_s)/len(accs_s)*100:.2f}%  "
          f"worst={min(accs_s)*100:.2f}%  spread={(max(accs_s)-min(accs_s))*100:.2f}pp")
    print(f"natural   summary: mean={sum(accs_n)/len(accs_n)*100:.2f}%  "
          f"worst={min(accs_n)*100:.2f}%  spread={(max(accs_n)-min(accs_n))*100:.2f}pp")


if __name__ == "__main__":
    main()
