"""train_rcanon.py

Loads a pretrained EventClassifier (e.g. v2 ra_aug_only), freezes it, and
trains a small RCanon module to predict per-scene log(rho*) that the base
classifier consumes.

Requires v2-preprocessed .pt files which embed scene metadata
(log_lambda_raw, n_events_raw, duration_ms).

Run:
    python scripts/negative_results/train_rcanon.py \
        --config configs/ncaltech101.yaml \
        --base   results/ncaltech101/ra_aug_only/best.ckpt \
        --name   rcanon_on_v2_ra_aug_only
"""
import argparse
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch_geometric.loader import DataLoader as PyGDataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cegnn.data.dataset import EventGraphDataset
from cegnn.utils import load_config, results_dir, trainer_device_kwargs
from cegnn.adapt.rcanon import CanonicalizedClassifier


def load_base_state(ckpt_path: str) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    return ckpt.get("state_dict", ckpt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--base", required=True, help="frozen base ckpt (v2-trained)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1.0e-3)
    ap.add_argument("--hidden", type=int, default=256,
                    help="R-Canon MLP hidden width (default 256)")
    ap.add_argument("--depth", type=int, default=4,
                    help="R-Canon MLP depth (default 4 -- ~200K params)")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lambda_cons", type=float, default=0.5)
    ap.add_argument("--lambda_reg", type=float, default=0.01)
    ap.add_argument("--batch_size", type=int, default=None,
                    help="override config batch_size (for memory control)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    pl.seed_everything(cfg["train"]["seed"], workers=True)
    cfg = dict(cfg)
    cfg["train"] = dict(cfg["train"])
    cfg["train"]["lr"] = args.lr
    cfg["train"]["epochs"] = args.epochs

    out = results_dir(cfg, args.name)
    print(f"[rcanon] outputs -> {out}")
    print(f"[rcanon] base = {args.base}")
    print(f"[rcanon] lambda_cons={args.lambda_cons}  lambda_reg={args.lambda_reg}")

    # Datasets: read v2 .pt files (which embed scene metadata).
    # We do NOT rebuild graphs here -- the base does its own neighbor queries.
    train_ds = EventGraphDataset(
        processed_dir=cfg["paths"]["processed"], split="train",
        cfg=cfg, augment=True, rebuild=False,
    )
    val_ds = EventGraphDataset(
        processed_dir=cfg["paths"]["processed"], split="test",
        cfg=cfg, augment=False, rebuild=False,
    )
    print(f"[rcanon] |train|={len(train_ds)}  |val|={len(val_ds)}")

    batch_sz = args.batch_size if args.batch_size is not None else cfg["train"]["batch_size"]
    print(f"[rcanon] batch_size = {batch_sz}")
    train_loader = PyGDataLoader(
        train_ds, batch_size=batch_sz, shuffle=True,
        num_workers=cfg["train"]["num_workers"], pin_memory=True,
    )
    val_loader = PyGDataLoader(
        val_ds, batch_size=batch_sz, shuffle=False,
        num_workers=cfg["train"]["num_workers"], pin_memory=True,
    )

    base_state = load_base_state(args.base)
    model = CanonicalizedClassifier(
        cfg, base_state_dict=base_state,
        hidden=args.hidden, depth=args.depth, dropout=args.dropout,
        lambda_cons=args.lambda_cons,
        lambda_reg=args.lambda_reg,
    )
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[rcanon] trainable params: {n_train:,} / {n_total:,} "
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
    print(f"\n[rcanon] best ckpt: {ckpt.best_model_path}")
    print(f"[rcanon] best val_acc: {ckpt.best_model_score:.4f}")


if __name__ == "__main__":
    main()
