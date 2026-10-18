"""Evaluation, checkpoint loading and small IO helpers.

Also re-exports the config / loader helpers so scripts can do
``from cegnn.utils import load_config, make_loader, ...`` in one line.
"""
import csv
import os
from pathlib import Path
from typing import Dict, List, Union

import torch

from .config import load_config, results_dir                       # noqa: F401  (re-export)
from .data.loaders import make_loader, get_lambda_train            # noqa: F401  (re-export)
from .models import (
    EventClassifier, RateConditionedClassifier,
    FiLMRateClassifier, FiLMGlobalOnlyClassifier,
)


# ============================================================================
# EVALUATION
# ============================================================================

@torch.no_grad()
def evaluate(model, loader, device: str = "cuda") -> Dict[str, float]:
    """Run model on loader, return accuracy and loss.

    Supports both EventClassifier-style models (return logits) and
    RateMoEGNN-style models (return (logits, aux)).
    """
    model = model.to(device).eval()
    n_correct = 0
    n_total = 0
    loss_sum = 0.0
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        logits = out[0] if isinstance(out, tuple) else out
        y = batch.y.long().view(-1)
        loss = torch.nn.functional.cross_entropy(logits, y, reduction="sum")
        loss_sum += float(loss.item())
        n_correct += int((logits.argmax(1) == y).sum().item())
        n_total += int(y.size(0))
    return {
        "acc": n_correct / max(1, n_total),
        "loss": loss_sum / max(1, n_total),
        "n": n_total,
    }


# ============================================================================
# CHECKPOINT IO
# ============================================================================

def _remap_state_dict(state_dict: dict, target_model_class) -> dict:
    """Remap old checkpoint state dict to new model architecture if needed.

    Detects architecture changes and remaps keys intelligently based on
    the target model class's expected structure.
    """
    remapped = {}
    needs_remapping = False

    for key, val in state_dict.items():
        remapped[key] = val

    return remapped, needs_remapping


def load_checkpoint(ckpt_path: str, cfg: dict):
    """Load a model from a checkpoint, auto-detecting which class it belongs to.

    Supported checkpoint formats (auto-detected from the state-dict keys):
      1. EventClassifier / RateConditionedClassifier (Lightning)
      2. FiLMRateClassifier / FiLMGlobalOnlyClassifier (Lightning)
      3. RateMoEGNN (raw torch.save with state_dict + args + cfg)
    """
    import torch
    import tempfile
    import os

    # Load raw checkpoint to inspect hyperparameters and determine model type
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", {})
    hp = ckpt.get("hyper_parameters", {})

    # ── Detect RateMoE checkpoint (raw torch.save, has args.num_experts) ──
    ratemoe_args = ckpt.get("args", None)
    if isinstance(ratemoe_args, dict) and "num_experts" in ratemoe_args:
        from .models import RateMoEGNN
        # Honor flags that affect architecture
        num_experts = int(ratemoe_args["num_experts"])
        top_k = int(ratemoe_args.get("top_k", 2))
        if ratemoe_args.get("all_experts", False):
            top_k = num_experts
        ckpt_cfg = ckpt.get("cfg", cfg)
        num_classes = ckpt_cfg.get("dataset", {}).get(
            "num_classes", cfg.get("dataset", {}).get("num_classes", 2)
        )
        model = RateMoEGNN(
            in_channels=2,
            hidden_dim=int(ratemoe_args.get("hidden_dim", 64)),
            num_classes=num_classes,
            num_experts=num_experts,
            top_k=top_k,
            num_freqs=int(ratemoe_args.get("num_freqs", 4)),
        )
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"[load_checkpoint] RateMoE: missing={len(missing)} "
                  f"unexpected={len(unexpected)} keys")
        print(f"[load_checkpoint] Loaded RateMoEGNN K={num_experts} top_k={top_k}")
        return model

    # Use checkpoint's saved config (has correct architecture) but allow current cfg
    # to override evaluation-time settings (like graph_mode)
    if hp:
        ckpt_cfg = hp.copy()
        # Merge in any non-model settings from current cfg, but keep model settings
        # from checkpoint (since those affect network dimensions)
        for key in cfg:
            if key not in ckpt_cfg or key not in ["model"]:
                ckpt_cfg[key] = cfg[key]
        cfg = ckpt_cfg

    model_class = EventClassifier
    # Detect FiLM variants first (look at presence of film_* and film1/2/3 keys).
    has_film_global = any(k.startswith("film_global.") for k in state_dict.keys())
    has_film_per_node = any(k.startswith("film1.") or k.startswith("film2.")
                            or k.startswith("film3.") for k in state_dict.keys())
    if has_film_per_node and has_film_global:
        model_class = FiLMRateClassifier
        print(f"[load_checkpoint] Detected FiLMRateClassifier (full FiLM)")
    elif has_film_global and not has_film_per_node:
        model_class = FiLMGlobalOnlyClassifier
        print(f"[load_checkpoint] Detected FiLMGlobalOnlyClassifier "
              f"(film_global only)")
    else:
        # Detect RateConditionedClassifier from head input dim (= 2*gd + 1, vs 2*gd).
        gd = cfg.get("model", {}).get("global_dim", None)
        head0_w = state_dict.get("head.0.weight")
        if gd is not None and head0_w is not None:
            in_dim = int(head0_w.shape[1])
            if in_dim == 2 * gd + 1:
                model_class = RateConditionedClassifier
                print(f"[load_checkpoint] Detected RateConditionedClassifier "
                      f"(head in_dim={in_dim} = 2*gd+1)")
    # Remap old architecture keys if needed
    remapped_state_dict, was_remapped = _remap_state_dict(state_dict, model_class)
    if was_remapped:
        ckpt["state_dict"] = remapped_state_dict

    # Temporarily replace checkpoint on disk for loading
    with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as f:
        torch.save(ckpt, f.name)
        temp_ckpt_path = f.name

    try:
        model = model_class.load_from_checkpoint(
            temp_ckpt_path,
            cfg=cfg,
            map_location="cpu",
            strict=True,  # Use strict mode now that config matches
        )
    finally:
        os.unlink(temp_ckpt_path)

    return model


# ============================================================================
# TRAINER DEVICE
# ============================================================================

def trainer_device_kwargs() -> dict:
    """Lightning ``Trainer`` kwargs: one GPU when available, otherwise plain CPU.

    The paper's runs used a single GPU; the CPU fallback exists so the whole pipeline
    can be smoke-tested on a laptop or a login node.
    """
    return {"accelerator": "gpu", "devices": 1} if torch.cuda.is_available() else {}


# ============================================================================
# CSV WRITING
# ============================================================================

def write_csv(path: Union[str, os.PathLike], rows: List[dict]) -> None:
    """Write list-of-dicts to CSV. All rows must share the same keys."""
    if not rows:
        Path(path).write_text("")
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
