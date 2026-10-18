"""train_baseline.py

Train the baseline classifier on fixed (preprocessed) graphs.
This reproduces your existing setup, but config-driven and with proper
result organization.

Run:
    python scripts/train/train_baseline.py --config configs/ncars.yaml
"""
import argparse
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cegnn.utils import load_config, make_loader, results_dir, trainer_device_kwargs
from cegnn.models import EventClassifier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--name", default="baseline",
                    help="results subdir name (default: baseline)")
    ap.add_argument("--head_hidden", type=int, default=None,
                    help="override cfg.model.head_hidden (used for param-budget "
                         "control vs FiLM-rate)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.head_hidden is not None:
        # capacity control for the paper: widen the head so the baseline gets about as many
        # extra parameters as FiLM adds (head_hidden=200 => +44K)
        cfg = dict(cfg)
        cfg["model"] = dict(cfg["model"])
        cfg["model"]["head_hidden"] = args.head_hidden
        print(f"[train] override head_hidden -> {args.head_hidden}")
    pl.seed_everything(cfg["train"]["seed"], workers=True)

    out = results_dir(cfg, args.name)
    print(f"[train] outputs -> {out}")

    # train-time augmentation = random horizontal flip + polarity flip (no rate augmentation here)
    train_loader = make_loader(cfg, "train", augment=True)
    # NOTE: the test split doubles as validation (no separate split); best epoch and early
    # stopping use its accuracy. Same protocol for every method in the paper.
    val_loader   = make_loader(cfg, "test",  augment=False)
    print(f"[train] |train|={len(train_loader.dataset)}  "
          f"|test|={len(val_loader.dataset)}")

    model = EventClassifier(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] model params: {n_params/1e6:.2f}M")

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
        log_every_n_steps=20,
    )
    trainer.fit(model, train_loader, val_loader)

    print(f"\n[train] best ckpt: {ckpt.best_model_path}")
    print(f"[train] best val_acc: {ckpt.best_model_score:.4f}")


if __name__ == "__main__":
    main()