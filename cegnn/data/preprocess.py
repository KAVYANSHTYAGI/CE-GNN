"""Dataset-agnostic preprocessing driver: raw event files -> CEGNN ``.pt`` graphs.

    python -m cegnn.data.preprocess --dataset ncaltech101 --config configs/ncaltech101.yaml --workers 16
    python -m cegnn.data.preprocess --dataset dvslip --config configs/dvslip.yaml --limit 50   # smoke test

The dataset-specific part (file layout, binary format, split) lives in
``cegnn/data/readers/<dataset>.py``; the dataset-independent part (rate
statistics, subsampling, radius graph) lives in ``cegnn/data/sample.py``.
The job is resumable: files that already exist in the output folder are skipped.
"""
import argparse
import multiprocessing as mp
import os
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

from ..config import load_config
from .readers import SampleRef, available_datasets, get_adapter
from .sample import events_to_sample


def graph_params(cfg: Dict) -> Dict:
    g = cfg["graph"]
    return {
        "t_unit": float(cfg["dataset"].get("t_unit", 1.0)),
        "r": float(g["r"]),
        "d_max": int(g["d_max"]),
        "subsample_K": int(g["subsample_K"]),
        "n_max": int(g["n_samples"]),
    }


def process_one(adapter, ref: SampleRef, out_dir: str, params: Dict) -> str:
    """Process a single sample. Returns "ok" | "skip" | "degenerate" | "error: ..."."""
    out_path = os.path.join(out_dir, ref.split, f"{ref.out_name}.pt")
    if os.path.exists(out_path):
        return "skip"                       # resumable: finished samples are never recomputed
    try:
        (x, y, t_ms, p), label = adapter.load(ref)          # dataset-specific part
        data = events_to_sample(x, y, t_ms, p, label, ref.file_id, **params)
        if data is None:
            return "degenerate"
        torch.save(data, out_path)
        return "ok"
    except Exception as e:  # keep going; one corrupt file must not kill a 20k-sample job
        return f"error: {ref.file_id}: {e}"


# --- multiprocessing plumbing -----------------------------------------------------
# Each worker builds its own adapter once (in _init_worker) instead of pickling it per task;
# this matters for DVS128 Gesture, whose adapter holds a tonic dataset handle.
_W: Dict = {}


def _init_worker(name, raw_root, cfg, out_dir, params):
    _W["adapter"] = get_adapter(name)(raw_root, cfg)
    _W["out_dir"], _W["params"] = out_dir, params


def _work(ref: SampleRef) -> str:
    return process_one(_W["adapter"], ref, _W["out_dir"], _W["params"])


def run(dataset: str, cfg: Dict, workers: int = 1, limit: Optional[int] = None,
        raw_root: Optional[str] = None, out_dir: Optional[str] = None) -> Dict[str, int]:
    raw_root = raw_root or cfg["paths"]["raw_root"]
    out_dir = out_dir or cfg["paths"]["processed"]
    params = graph_params(cfg)

    adapter = get_adapter(dataset)(raw_root, cfg)
    refs: List[SampleRef] = adapter.list_samples()
    if limit:                                  # smoke test: first `limit` samples of each split
        keep = {"train": 0, "test": 0}
        sub = []
        for r in refs:
            if keep[r.split] < limit:
                sub.append(r)
                keep[r.split] += 1
        refs = sub
    for split in ("train", "test"):
        os.makedirs(os.path.join(out_dir, split), exist_ok=True)

    n = {s: sum(r.split == s for r in refs) for s in ("train", "test")}
    print(f"[preprocess] {dataset}: raw={raw_root}\n             out={out_dir}\n"
          f"             train={n['train']} test={n['test']} params={params}")

    counts: Dict[str, int] = {}
    errors: List[str] = []

    def _tally(res: str):
        key = "error" if res.startswith("error") else res
        counts[key] = counts.get(key, 0) + 1
        if key == "error":
            errors.append(res)

    if workers <= 1:
        for ref in tqdm(refs):
            _tally(process_one(adapter, ref, out_dir, params))
    else:
        with mp.Pool(workers, initializer=_init_worker,
                     initargs=(dataset, raw_root, cfg, out_dir, params)) as pool:
            for res in tqdm(pool.imap_unordered(_work, refs, chunksize=8), total=len(refs)):
                _tally(res)

    print(f"[preprocess] done: {counts}")
    for e in errors[:10]:
        print("  ", e)
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=available_datasets())
    ap.add_argument("--config", required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N samples per split (smoke test)")
    ap.add_argument("--raw_root", default=None, help="override paths.raw_root")
    ap.add_argument("--out_dir", default=None, help="override paths.processed")
    args = ap.parse_args()
    cfg = load_config(args.config, make_results_dir=False)
    run(args.dataset, cfg, args.workers, args.limit, args.raw_root, args.out_dir)


if __name__ == "__main__":
    main()
