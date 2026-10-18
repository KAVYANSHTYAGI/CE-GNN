"""train_rate_aug_adaptive_graph.py

Trains a model with:
  - rate augmentation: sample a random rate factor per training sample
  - adaptive graph construction at every step (graph_mode flag)

Crucially this REBUILDS the graph on the fly using the chosen graph_mode,
so the network sees graphs whose structure adapts to rate.

Run:
    # full RA-GNN
    python scripts/negative_results/train_rate_aug_adaptive_graph.py --config configs/ncars.yaml \
        --graph_mode adapt_both --name ra_full

    # ablations: rate-augmented training but with FIXED graphs
    # (this is the "honest control" - if this matches RA-GNN, the
    #  adaptive graph isn't doing anything beyond augmentation)
    python scripts/negative_results/train_rate_aug_adaptive_graph.py --config configs/ncars.yaml \
        --graph_mode fixed --name ra_aug_only
"""
import argparse
import random
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cegnn.data.dataset import (
    EventGraphDataset, RateResample, RebuildGraph, TrainAugment, Compose,
)
from cegnn.utils import load_config, results_dir, get_lambda_train, trainer_device_kwargs
from cegnn.models import EventClassifier
from torch_geometric.loader import DataLoader as PyGDataLoader


# ============================================================================
# RATE-AUGMENTED DATASET
# ============================================================================

class RateAugmentedDataset(EventGraphDataset):
    """Per-sample random rate factor, chosen from a list each __getitem__."""

    def __init__(self, train_rates, **kwargs):
        # we manage rebuild ourselves, force base class off
        super().__init__(rebuild=False, **kwargs)
        self.train_rates = list(train_rates)
        self._graph_mode = kwargs.get("graph_mode", "fixed")
        self._cfg = kwargs["cfg"]
        self._lambda_train = kwargs.get("lambda_train", None)

    def __getitem__(self, idx: int) -> Data:
        try:
            data = torch.load(self.files[idx], weights_only=False)
        except TypeError:
            data = torch.load(self.files[idx])

        # Label: prefer embedded data.y; fall back to filename rule for legacy N-Cars.
        import os as _os
        has_embedded_y = (
            hasattr(data, "y") and data.y is not None
            and torch.is_tensor(data.y) and data.y.numel() > 0
        )
        if has_embedded_y:
            data.y = data.y.long().view(-1)[:1]
        else:
            label = int("cars" in _os.path.basename(self.files[idx]))
            data.y = torch.tensor([label], dtype=torch.long)

        # strip stored edges
        data = Data(x=data.x, pos=data.pos, y=data.y)

        # pick a rate, resample, rebuild
        rf = random.choice(self.train_rates)
        data = RateResample(rf)(data)
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
    ap.add_argument("--name", required=True,
                    help="results subdir (e.g. ra_full, ra_aug_only)")
    ap.add_argument("--graph_mode", required=True,
                    choices=["fixed", "adapt_radius", "adapt_subsample", "adapt_both"])
    ap.add_argument("--rate_aug", action="store_true", default=True,
                    help="enable rate augmentation (default: yes)")
    ap.add_argument("--no_rate_aug", dest="rate_aug", action="store_false")
    args = ap.parse_args()

    cfg = load_config(args.config)
    pl.seed_everything(cfg["train"]["seed"], workers=True)
    out = results_dir(cfg, args.name)
    print(f"[train_ra] outputs -> {out}")
    print(f"[train_ra] graph_mode={args.graph_mode}  rate_aug={args.rate_aug}")

    # need lambda_train for adaptive radius modes
    lam_train = None
    if args.graph_mode in ("adapt_radius", "adapt_both"):
        lam_train = get_lambda_train(cfg)

    train_rates = cfg["rate_adaptive"]["train_rates"] if args.rate_aug else [1.0]

    train_ds = RateAugmentedDataset(
        train_rates=train_rates,
        processed_dir=cfg["paths"]["processed"], split="train",
        cfg=cfg, augment=True, graph_mode=args.graph_mode,
        lambda_train=lam_train,
    )
    val_ds = EventGraphDataset(
        processed_dir=cfg["paths"]["processed"], split="test",
        cfg=cfg, augment=False, rebuild=False,  # eval at native rate during training
    )
    print(f"[train_ra] |train|={len(train_ds)}  |val|={len(val_ds)}")
    print(f"[train_ra] train rate set: {train_rates}")

    train_loader = PyGDataLoader(train_ds,
                                 batch_size=cfg["train"]["batch_size"],
                                 shuffle=True,
                                 num_workers=cfg["train"]["num_workers"],
                                 pin_memory=True)
    val_loader = PyGDataLoader(val_ds,
                               batch_size=cfg["train"]["batch_size"],
                               shuffle=False,
                               num_workers=cfg["train"]["num_workers"],
                               pin_memory=True)

    model = EventClassifier(cfg)

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
    print(f"\n[train_ra] best ckpt: {ckpt.best_model_path}")
    print(f"[train_ra] best val_acc: {ckpt.best_model_score:.4f}")


if __name__ == "__main__":
    main()