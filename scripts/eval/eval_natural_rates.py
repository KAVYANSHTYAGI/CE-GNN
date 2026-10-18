"""eval_natural_rates.py

Evaluates model accuracy as a function of *natural* per-sample event rate,
WITHOUT applying any synthetic rate factor. This answers the key question:

    "Do rate-robust methods actually generalize to the natural rate variation
    that exists in the wild, or are they only robust to our synthetic shifts?"

For each test sample we use the lambda_global field that was computed during
preprocessing on the raw (pre-time-normalization) event stream. Samples are
bucketed by quantile of log(lambda_global). Per-bucket accuracy is reported.

Run:
    python scripts/eval/eval_natural_rates.py \
        --config configs/ncaltech101.yaml \
        --name natural_rate_eval \
        --models \
            baseline:results/ncaltech101/baseline/best.ckpt:fixed \
            ra_aug_only:results/ncaltech101/ra_aug_only/best.ckpt:fixed \
            ra_full:results/ncaltech101/ra_full/best.ckpt:adapt_both
        --rptta_base results/ncaltech101/ra_aug_only/best.ckpt \
        --rptta_ckpt results/ncaltech101/rptta_on_ra_aug_only/best.ckpt \
        --n_buckets 7
"""
import argparse
import csv
import sys
import math
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader as PyGDataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cegnn.utils import load_config, results_dir, load_checkpoint, get_lambda_train
from cegnn.data.dataset import EventGraphDataset
from cegnn.models import EventClassifier
from cegnn.adapt.rp_tta import RPTTAClassifier


# ============================================================================
# COLLECT PER-SAMPLE LAMBDA AND LABELS
# ============================================================================

def collect_per_sample_metadata(processed_dir: str, split: str):
    """Walk the .pt files; pull (file, label, lambda_global) for each.

    lambda_global was stored at preprocessing time as log(events / time_extent)
    on the *raw* (pre-normalization) timestamps.
    """
    files = sorted(Path(processed_dir, split).glob("*.pt"))
    meta = []
    for f in files:
        try:
            d = torch.load(f, weights_only=False)
        except TypeError:
            d = torch.load(f)
        label = int(d.y.view(-1)[0].item())
        # lambda_global stored as torch.log(lambda_global + eps), scalar
        lam_log = float(d.lambda_global.view(-1)[0].item())
        meta.append({"file": str(f), "label": label, "log_lambda": lam_log})
    return meta


def bucket_by_quantile(meta, n_buckets: int):
    """Assign each sample to a quantile bucket of log_lambda. Returns:
       (sorted-by-log-lambda meta, bucket_ids per sample, bucket_ranges)."""
    # Equal-COUNT buckets (quantiles), not equal-width: every bucket has about the same number of
    # test samples, so per-bucket accuracies are comparably noisy.
    sorted_meta = sorted(meta, key=lambda m: m["log_lambda"])
    n = len(sorted_meta)
    bucket_size = math.ceil(n / n_buckets)
    bucket_ids = []
    bucket_ranges = []
    for b in range(n_buckets):
        lo = b * bucket_size
        hi = min((b + 1) * bucket_size, n)
        if lo >= hi:
            break
        bucket_ranges.append((
            sorted_meta[lo]["log_lambda"],
            sorted_meta[hi - 1]["log_lambda"],
            hi - lo,
        ))
        for _ in range(lo, hi):
            bucket_ids.append(b)
    return sorted_meta, bucket_ids, bucket_ranges


# ============================================================================
# PER-MODEL EVAL OVER A FILE LIST
# ============================================================================

@torch.no_grad()
def eval_file_list(model, file_list, cfg, device, repeats: int = 1):
    """Accuracy on a specific list of files, averaged over ``repeats`` passes.

    The backbone's farthest-point sampling starts from a random point, so a single
    pass is stochastic (a few points on small test sets); ``repeats > 1`` averages
    that noise out.
    """
    # Subclass EventGraphDataset by overriding the file list
    ds = EventGraphDataset(
        processed_dir=cfg["paths"]["processed"], split="test", cfg=cfg,
        augment=False, rebuild=False,
    )
    ds.files = file_list
    loader = PyGDataLoader(
        ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
        num_workers=cfg["train"]["num_workers"], pin_memory=True,
    )
    n_correct = 0
    n_total = 0
    for _ in range(max(1, repeats)):      # one pass by default; see --repeats
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            y = batch.y.long().view(-1)
            n_correct += (logits.argmax(1) == y).sum().item()
            n_total += y.numel()
    return n_correct / max(1, n_total), n_total // max(1, repeats)


def parse_model_spec(spec: str):
    name, ckpt, mode = spec.split(":")
    return {"name": name, "ckpt": ckpt, "mode": mode}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--name", default="natural_rate_eval")
    ap.add_argument("--models", nargs="*", default=[],
                    help="entries name:ckpt:graph_mode for EventClassifier ckpts")
    ap.add_argument("--rptta_base", default=None,
                    help="base ckpt for an RP-TTA model (optional)")
    ap.add_argument("--rptta_ckpt", default=None,
                    help="trained RP-TTA ckpt (optional)")
    ap.add_argument("--n_buckets", type=int, default=7)
    ap.add_argument("--repeats", type=int, default=1,
                    help="evaluation passes to average (FPS has a random start; the paper "
                         "protocol is 1, use >=5 for stable numbers)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = results_dir(cfg, args.name)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Walk the test set and collect natural log-lambda per file
    print(f"[natrate] scanning test set for lambda_global ...")
    meta = collect_per_sample_metadata(cfg["paths"]["processed"], "test")
    log_lams = np.array([m["log_lambda"] for m in meta])
    print(f"[natrate] {len(meta)} samples, log_lambda: "
          f"min={log_lams.min():.3f}  median={np.median(log_lams):.3f}  "
          f"max={log_lams.max():.3f}  std={log_lams.std():.3f}")

    # 2) Bucket
    sorted_meta, bucket_ids, bucket_ranges = bucket_by_quantile(meta, args.n_buckets)
    print(f"\n[natrate] {len(bucket_ranges)} buckets:")
    for i, (lo, hi, n) in enumerate(bucket_ranges):
        print(f"  bucket {i}: log_lambda in [{lo:.3f}, {hi:.3f}]  n={n}")
    # Group files per bucket
    files_per_bucket = [[] for _ in range(len(bucket_ranges))]
    for sm, b in zip(sorted_meta, bucket_ids):
        files_per_bucket[b].append(sm["file"])

    # 3) Evaluate every model on every bucket
    rows = []
    # ---- EventClassifier models ----
    for spec in args.models:
        m = parse_model_spec(spec)
        print(f"\n[natrate] model: {m['name']}  ckpt: {m['ckpt']}")
        # build model
        model = load_checkpoint(m["ckpt"], cfg).to(device).eval()
        for b, fl in enumerate(files_per_bucket):
            acc, n = eval_file_list(model, fl, cfg, device, args.repeats)
            lo, hi, _ = bucket_ranges[b]
            row = {
                "model": m["name"],
                "bucket": b,
                "log_lambda_lo": round(lo, 4),
                "log_lambda_hi": round(hi, 4),
                "n": n,
                "test_acc": round(acc, 4),
            }
            rows.append(row)
            print(f"  bucket {b} [{lo:.3f},{hi:.3f}] n={n}  acc={acc*100:5.2f}%")

    # ---- RP-TTA (optional) ----
    if args.rptta_base and args.rptta_ckpt:
        print(f"\n[natrate] model: rptta  base={args.rptta_base}  rptta={args.rptta_ckpt}")
        base_state = torch.load(args.rptta_base, map_location="cpu")
        base_state = base_state.get("state_dict", base_state)
        model = RPTTAClassifier(cfg, base_state_dict=base_state)
        rptta_state = torch.load(args.rptta_ckpt, map_location="cpu")
        rptta_state = rptta_state.get("state_dict", rptta_state)
        model.load_state_dict(rptta_state, strict=True)
        model = model.to(device).eval()
        for b, fl in enumerate(files_per_bucket):
            acc, n = eval_file_list(model, fl, cfg, device, args.repeats)
            lo, hi, _ = bucket_ranges[b]
            row = {
                "model": "rptta",
                "bucket": b,
                "log_lambda_lo": round(lo, 4),
                "log_lambda_hi": round(hi, 4),
                "n": n,
                "test_acc": round(acc, 4),
            }
            rows.append(row)
            print(f"  bucket {b} [{lo:.3f},{hi:.3f}] n={n}  acc={acc*100:5.2f}%")

    # 4) Save CSV + summary
    csv_path = out / "natural_rate_per_bucket.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[natrate] wrote {csv_path}")

    # Summary: per-model mean / worst / spread across buckets
    print(f"\n[natrate] summary (mean / worst / spread over {len(bucket_ranges)} natural-rate buckets):")
    print(f"  {'model':<16}  {'mean_acc':>10}  {'worst_acc':>10}  {'spread_pp':>10}")
    seen = []
    for r in rows:
        if r["model"] in seen:
            continue
        seen.append(r["model"])
        accs = [x["test_acc"] for x in rows if x["model"] == r["model"]]
        mean = sum(accs) / len(accs)
        worst = min(accs)
        spread = (max(accs) - min(accs)) * 100
        print(f"  {r['model']:<16}  {mean*100:>9.2f}%  {worst*100:>9.2f}%  {spread:>9.2f}")


if __name__ == "__main__":
    main()
