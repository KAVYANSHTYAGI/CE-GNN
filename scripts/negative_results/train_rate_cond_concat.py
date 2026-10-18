"""train_rate_cond_concat.py

Uses RateConditionedClassifier: baseline PointNet++ backbone, but
data.log_lambda_raw is concatenated into the global feature before the
classification head. Rate is treated as an INPUT FEATURE, not erased
(ra_aug_only) or canonicalized away (R-Canon).

Requires v2-preprocessed .pt files (which embed log_lambda_raw).

Two training variants:
    (A) --no_rate_aug  : natural rates only (matches baseline data distribution)
    (B) (default)      : rate augmentation from cfg.rate_adaptive.train_rates,
                         so the model sees a wider rate range during training.

Run:
    # variant A (no rate aug)
    python scripts/negative_results/train_rate_cond_concat.py \
        --config configs/ncaltech101.yaml \
        --name rate_cond_natural \
        --no_rate_aug

    # variant B (with rate aug)
    python scripts/negative_results/train_rate_cond_concat.py \
        --config configs/ncaltech101.yaml \
        --name rate_cond_rateaug
"""
import argparse
import random
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cegnn.data.dataset import (
    EventGraphDataset, RateResample, RebuildGraph,
)
from cegnn.utils import load_config, results_dir, get_lambda_train, trainer_device_kwargs
from cegnn.models import RateConditionedClassifier


# ============================================================================
# DATASET: rate-aug + metadata-preserving (so log_lambda_raw flows to model)
# ============================================================================

_META_KEYS = ("log_lambda_raw", "n_events_raw", "duration_ms")


def _strip_keep_meta(data: Data) -> Data:
    """Drop stored edges but preserve v2 per-sample rate metadata."""
    out = Data(x=data.x, pos=data.pos, y=data.y)
    for k in _META_KEYS:
        if hasattr(data, k):
            setattr(out, k, getattr(data, k))
    return out


class RateAugMetaDataset(EventGraphDataset):
    """Like 04_train_ra.RateAugmentedDataset but preserves v2 metadata."""

    def __init__(self, train_rates, **kwargs):
        super().__init__(rebuild=False, **kwargs)
        self.train_rates = list(train_rates)
        self._graph_mode = kwargs.get("graph_mode", "fixed")
        self._cfg = kwargs["cfg"]
        self._lambda_train = kwargs.get("lambda_train", None)

    def __getitem__(self, idx: int) -> Data:
        import os as _os
        try:
            data = torch.load(self.files[idx], weights_only=False)
        except TypeError:
            data = torch.load(self.files[idx])

        has_y = (hasattr(data, "y") and data.y is not None
                 and torch.is_tensor(data.y) and data.y.numel() > 0)
        if has_y:
            data.y = data.y.long().view(-1)[:1]
        else:
            data.y = torch.tensor(
                [int("cars" in _os.path.basename(self.files[idx]))],
                dtype=torch.long,
            )

        data = _strip_keep_meta(data)
        rf = random.choice(self.train_rates)
        data = RateResample(rf)(data)        # updates log_lambda_raw too
        data = RebuildGraph(self._graph_mode, self._cfg, self._lambda_train)(data)
        if self.augment_fn is not None:
            data = self.augment_fn(data)
        return data


# ============================================================================
# DRIVER
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--graph_mode", default="fixed",
                    choices=["fixed", "adapt_radius", "adapt_subsample", "adapt_both"])
    ap.add_argument("--rate_aug", action="store_true", default=True,
                    help="enable rate augmentation (default: yes)")
    ap.add_argument("--no_rate_aug", dest="rate_aug", action="store_false")
    ap.add_argument("--batch_size", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    pl.seed_everything(cfg["train"]["seed"], workers=True)
    out = results_dir(cfg, args.name)
    print(f"[rate_cond] outputs -> {out}")
    print(f"[rate_cond] graph_mode={args.graph_mode}  rate_aug={args.rate_aug}")

    lam_train = None
    if args.graph_mode in ("adapt_radius", "adapt_both"):
        lam_train = get_lambda_train(cfg)

    train_rates = cfg["rate_adaptive"]["train_rates"] if args.rate_aug else [1.0]

    train_ds = RateAugMetaDataset(
        train_rates=train_rates,
        processed_dir=cfg["paths"]["processed"], split="train",
        cfg=cfg, augment=True, graph_mode=args.graph_mode,
        lambda_train=lam_train,
    )
    # val at native rate; rebuild=False so .pt metadata flows through unchanged
    val_ds = EventGraphDataset(
        processed_dir=cfg["paths"]["processed"], split="test",
        cfg=cfg, augment=False, rebuild=False,
    )
    print(f"[rate_cond] |train|={len(train_ds)}  |val|={len(val_ds)}")
    print(f"[rate_cond] train rate set: {train_rates}")

    bs = args.batch_size or cfg["train"]["batch_size"]
    print(f"[rate_cond] batch_size = {bs}")
    train_loader = PyGDataLoader(train_ds, batch_size=bs, shuffle=True,
        num_workers=cfg["train"]["num_workers"], pin_memory=True)
    val_loader = PyGDataLoader(val_ds, batch_size=bs, shuffle=False,
        num_workers=cfg["train"]["num_workers"], pin_memory=True)

    model = RateConditionedClassifier(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[rate_cond] model params: {n_params/1e6:.2f}M")

    ckpt = pl.callbacks.ModelCheckpoint(
        dirpath=out, filename="best",
        monitor="val_acc", mode="max", save_top_k=1, save_last=True,
    )
    early = pl.callbacks.EarlyStopping(
        monitor="val_acc", mode="max", patience=cfg["train"]["patience"],
    )
    trainer = pl.Trainer(
        max_epochs=cfg["train"]["epochs"],
        **trainer_device_kwargs(),
        gradient_clip_val=cfg["train"]["grad_clip"],
        callbacks=[ckpt, early],
        default_root_dir=out,
        progress_bar_refresh_rate=1,
    )
    trainer.fit(model, train_loader, val_loader)
    print(f"\n[rate_cond] best ckpt: {ckpt.best_model_path}")
    print(f"[rate_cond] best val_acc: {ckpt.best_model_score:.4f}")


if __name__ == "__main__":
    main()
