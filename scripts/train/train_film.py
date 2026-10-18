"""train_film.py - train the CEGNN rate-conditioned classifier (Rate-FiLM).

Same PointNet++ backbone as the baseline, but the per-sample log event rate
(``data.log_lambda_raw``) modulates the features through identity-initialised
FiLM layers after every set-abstraction stage and after the global pool
(``--variant global_only`` keeps only the last one, as an ablation).

Training recipes (the paper's headline method is the CURRICULUM):

    # 1) FiLM, natural event rates only
    python scripts/train/train_film.py --config configs/ncaltech101.yaml \\
        --name film_rate_natural --no_rate_aug

    # 2) CURRICULUM: warm-start from (1), then fine-tune WITH rate augmentation
    python scripts/train/train_film.py --config configs/ncaltech101.yaml \\
        --name film_rate_curriculum --init_ckpt results/ncaltech101/film_rate_natural/best.ckpt

    # (control) FiLM trained from scratch with rate augmentation
    python scripts/train/train_film.py --config configs/ncaltech101.yaml --name film_rate_rateaug

Requires preprocessed ``.pt`` files (they embed ``log_lambda_raw``).
"""
import argparse
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch_geometric.loader import DataLoader as PyGDataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cegnn.data.dataset import EventGraphDataset, RateAugmentedDataset
from cegnn.utils import load_config, results_dir, get_lambda_train, trainer_device_kwargs
from cegnn.models import FiLMRateClassifier, FiLMGlobalOnlyClassifier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--graph_mode", default="fixed",
                    choices=["fixed", "adapt_radius", "adapt_subsample", "adapt_both"])
    ap.add_argument("--rate_aug", action="store_true", default=True)
    ap.add_argument("--no_rate_aug", dest="rate_aug", action="store_false")
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--variant", default="full",
                    choices=["full", "global_only"],
                    help="full = FiLM at every SA stage; "
                         "global_only = FiLM only at global pool output")
    ap.add_argument("--train_rates", type=float, nargs="+", default=None,
                    help="override cfg.rate_adaptive.train_rates (used for "
                         "mild rate-aug experiments)")
    ap.add_argument("--init_ckpt", default=None,
                    help="warm-start from a previously trained .ckpt (e.g. "
                         "film_rate_natural for curriculum)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    pl.seed_everything(cfg["train"]["seed"], workers=True)   # fixed seed (42 in the configs)
    out = results_dir(cfg, args.name)
    print(f"[film_rate] outputs -> {out}")
    print(f"[film_rate] graph_mode={args.graph_mode}  rate_aug={args.rate_aug}")

    # Only the adaptive-graph modes (negative result) need a reference rate; the paper's
    # method uses graph_mode="fixed", so this stays None.
    lam_train = None
    if args.graph_mode in ("adapt_radius", "adapt_both"):
        lam_train = get_lambda_train(cfg)

    # Rate factors sampled per training sample. [1.0] = natural-rate training (stage 1);
    # the config default {0.5, 1, 2} = rate augmentation (stage 2 of the curriculum).
    if args.train_rates is not None:
        train_rates = list(args.train_rates)
    else:
        train_rates = cfg["rate_adaptive"]["train_rates"] if args.rate_aug else [1.0]

    train_ds = RateAugmentedDataset(
        train_rates=train_rates,
        processed_dir=cfg["paths"]["processed"], split="train",
        cfg=cfg, augment=True, graph_mode=args.graph_mode,
        lambda_train=lam_train,
    )
    # NOTE: there is no separate validation split. The test split doubles as "val": the best
    # epoch and early stopping are chosen on its accuracy (same protocol for every method).
    val_ds = EventGraphDataset(
        processed_dir=cfg["paths"]["processed"], split="test",
        cfg=cfg, augment=False, rebuild=False,
    )
    print(f"[film_rate] |train|={len(train_ds)}  |val|={len(val_ds)}")
    print(f"[film_rate] train rate set: {train_rates}")

    bs = args.batch_size or cfg["train"]["batch_size"]
    print(f"[film_rate] batch_size = {bs}")
    train_loader = PyGDataLoader(train_ds, batch_size=bs, shuffle=True,
        num_workers=cfg["train"]["num_workers"], pin_memory=True)
    val_loader = PyGDataLoader(val_ds, batch_size=bs, shuffle=False,
        num_workers=cfg["train"]["num_workers"], pin_memory=True)

    if args.variant == "full":
        model = FiLMRateClassifier(cfg)
    else:
        model = FiLMGlobalOnlyClassifier(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    n_film = sum(p.numel() for n, p in model.named_parameters() if "film" in n)
    print(f"[film_rate] variant={args.variant}  params: {n_params:,}  "
          f"({n_film:,} in FiLM layers)")

    # Curriculum stage 2: start from stage-1 weights. strict=False because a *baseline* checkpoint
    # has no FiLM keys (they stay at their identity init); a stage-1 FiLM checkpoint loads fully.
    if args.init_ckpt is not None:
        print(f"[film_rate] warm-start from {args.init_ckpt}")
        ckpt_data = torch.load(args.init_ckpt, map_location="cpu")
        sd = ckpt_data.get("state_dict", ckpt_data)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"[film_rate]   warm-start missing={len(missing)}  "
                  f"unexpected={len(unexpected)}")
            if missing:
                print(f"[film_rate]     missing[0:5]: {missing[:5]}")
            if unexpected:
                print(f"[film_rate]     unexpected[0:5]: {unexpected[:5]}")

    # keep the single best epoch by val_acc as best.ckpt (+ last.ckpt); stop after `patience` flat epochs
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
    print(f"\n[film_rate] best ckpt: {ckpt.best_model_path}")
    print(f"[film_rate] best val_acc: {ckpt.best_model_score:.4f}")


if __name__ == "__main__":
    main()
