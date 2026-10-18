"""train_rptta.py

Loads a pretrained EventClassifier (e.g. ra_aug_only), freezes it, and trains
only the rate-prompt module on samples drawn from an EXTENDED rate range that
the base model never saw (default: {0.125, 0.25, 0.5, 1, 2, 4, 8}).

The point of training on extreme rates is so the prompt learns to *correct*
the base model's bottleneck features when the input distribution shifts.

Run:
    python scripts/train/train_rptta.py \
        --config configs/ncaltech101.yaml \
        --base results/ncaltech101/ra_aug_only/best.ckpt \
        --name rptta_on_ra_aug_only \
        --rates 0.125 0.25 0.5 1.0 2.0 4.0 8.0
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
    EventGraphDataset, RateResample, RebuildGraph, TrainAugment, Compose,
)
from cegnn.utils import load_config, results_dir, get_lambda_train, trainer_device_kwargs
from cegnn.models import EventClassifier
from cegnn.adapt.rp_tta import RPTTAClassifier


_META_KEYS = ("log_lambda_raw", "n_events_raw", "duration_ms")


class RateAugmentedDataset(EventGraphDataset):
    """Same shape as scripts/negative_results/train_rate_aug_adaptive_graph.py's, but takes any list of rates.

    PRESERVES v2 per-sample rate metadata (log_lambda_raw, etc.) through the
    strip-and-rebuild pipeline. This is REQUIRED when RP-TTA wraps a base that
    needs log_lambda_raw (FiLMRateClassifier). Without this, the base's _rate()
    falls back to zeros, and FiLM modulates on a constant rate -> garbage.
    """

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
            y = data.y.long().view(-1)[:1]
        else:
            y = torch.tensor([int("cars" in _os.path.basename(self.files[idx]))],
                             dtype=torch.long)

        slim = Data(x=data.x, pos=data.pos, y=y)
        for k in _META_KEYS:
            if hasattr(data, k):
                setattr(slim, k, getattr(data, k))
        data = slim

        rf = random.choice(self.train_rates)
        data = RateResample(rf)(data)              # also updates log_lambda_raw
        data = RebuildGraph(self._graph_mode, self._cfg, self._lambda_train)(data)
        if self.augment_fn is not None:
            data = self.augment_fn(data)
        return data


def load_base_state(ckpt_path: str) -> dict:
    """Strip the 'model.' prefix Lightning prepends if needed."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)
    return sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--base", required=True, help="path to base .ckpt")
    ap.add_argument("--name", required=True, help="results subdir name")
    ap.add_argument("--graph_mode", default="fixed",
                    choices=["fixed", "adapt_radius", "adapt_subsample", "adapt_both"])
    ap.add_argument("--rates", type=float, nargs="+",
                    default=[0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
                    help="rate factors sampled during prompt training")
    ap.add_argument("--prompt_hidden", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=30,
                    help="prompt-only fits fast; default 30")
    ap.add_argument("--lr", type=float, default=1.0e-3)
    args = ap.parse_args()

    cfg = load_config(args.config)
    pl.seed_everything(cfg["train"]["seed"], workers=True)
    # override LR for prompt training
    cfg = dict(cfg)
    cfg["train"] = dict(cfg["train"])
    cfg["train"]["lr"] = args.lr
    cfg["train"]["epochs"] = args.epochs

    out = results_dir(cfg, args.name)
    print(f"[rptta] outputs -> {out}")
    print(f"[rptta] base = {args.base}")
    print(f"[rptta] train rates = {args.rates}")

    lam_train = None
    if args.graph_mode in ("adapt_radius", "adapt_both"):
        lam_train = get_lambda_train(cfg)

    train_ds = RateAugmentedDataset(
        train_rates=args.rates,
        processed_dir=cfg["paths"]["processed"], split="train",
        cfg=cfg, augment=True, graph_mode=args.graph_mode,
        lambda_train=lam_train,
    )
    # validate at the SAME extended rate set so val_acc reflects multi-rate perf
    val_ds = RateAugmentedDataset(
        train_rates=args.rates,
        processed_dir=cfg["paths"]["processed"], split="test",
        cfg=cfg, augment=False, graph_mode=args.graph_mode,
        lambda_train=lam_train,
    )
    print(f"[rptta] |train|={len(train_ds)}  |val|={len(val_ds)}")

    train_loader = PyGDataLoader(train_ds,
        batch_size=cfg["train"]["batch_size"], shuffle=True,
        num_workers=cfg["train"]["num_workers"], pin_memory=True)
    val_loader = PyGDataLoader(val_ds,
        batch_size=cfg["train"]["batch_size"], shuffle=False,
        num_workers=cfg["train"]["num_workers"], pin_memory=True)

    base_state = load_base_state(args.base)
    model = RPTTAClassifier(cfg, base_state_dict=base_state,
                            prompt_hidden=args.prompt_hidden)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[rptta] trainable params: {n_train:,} / {n_total:,} total "
          f"({100.0 * n_train / n_total:.2f}%)")

    ckpt = pl.callbacks.ModelCheckpoint(
        dirpath=out, filename="best",
        monitor="val_acc", mode="max", save_top_k=1, save_last=True,
    )
    early = pl.callbacks.EarlyStopping(
        monitor="val_acc", mode="max", patience=cfg["train"]["patience"],
    )
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        **trainer_device_kwargs(),
        gradient_clip_val=cfg["train"]["grad_clip"],
        callbacks=[ckpt, early],
        default_root_dir=out,
        progress_bar_refresh_rate=1,
    )
    trainer.fit(model, train_loader, val_loader)
    print(f"\n[rptta] best ckpt: {ckpt.best_model_path}")
    print(f"[rptta] best val_acc: {ckpt.best_model_score:.4f}")


if __name__ == "__main__":
    main()
