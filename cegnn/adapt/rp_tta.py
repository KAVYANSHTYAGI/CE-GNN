"""Rate-Prompt Test-Time Adaptation (RP-TTA; called "RA-TTA" in the paper).

Wraps a frozen EventClassifier with a small (~17k param) rate-prompt module
that injects a learned correction at the global-feature bottleneck conditioned
on the per-sample event rate.

Pipeline:
    SA blocks (frozen)  ->  sa_global (frozen)  ->  features (B, 2*gd)
                                                       |
                       log(lambda_hat) -> RatePrompt -> shift
                                                       |
                                                       v
                                       features + alpha * shift  ->  head (frozen)

Training: freeze base, train prompt on EXTREME-rate batches the base never saw.
Inference: optionally do one entropy-minimization step over the prompt per batch
           (TENT-style) before producing predictions.
"""
from __future__ import annotations

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_scatter import scatter_add, scatter_min, scatter_max

from ..models import EventClassifier, FiLMRateClassifier


# ============================================================================
# PER-SAMPLE RATE FROM A BATCHED PyG DATA
# ============================================================================

def per_sample_log_rate(data, eps: float = 1e-6) -> torch.Tensor:
    """Compute log(events/time_extent) per sample in a PyG Batch.

    Returns: (B, 1) float tensor on the same device as data.pos.
    """
    # Unlike FiLM (which reads the stored raw-stream log_lambda_raw), the prompt estimates the rate
    # from the graph it is actually given: nodes / time-extent. At test time under a synthetic
    # shift this is all the adapter can see.
    t = data.pos[:, 2]
    batch = data.batch
    B = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
    n_per = scatter_add(torch.ones_like(t), batch, dim=0, dim_size=B)     # nodes per graph
    t_min, _ = scatter_min(t, batch, dim=0, dim_size=B)
    t_max, _ = scatter_max(t, batch, dim=0, dim_size=B)
    extent = (t_max - t_min).clamp(min=eps)
    lam = n_per / extent
    return torch.log(lam.clamp(min=eps)).unsqueeze(-1)


# ============================================================================
# RATE PROMPT MODULE
# ============================================================================

class RatePrompt(nn.Module):
    """Tiny MLP: log(lambda) -> shift vector for the global feature bottleneck.

    Residual-init: alpha starts at 0, so the wrapped model defaults to the
    base model's behavior at the start of training.
    """

    def __init__(self, out_dim: int, hidden: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, out_dim),
        )
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, lam_log: torch.Tensor) -> torch.Tensor:
        return self.alpha * self.mlp(lam_log)


# ============================================================================
# WRAPPED CLASSIFIER (BASE FROZEN, PROMPT TRAINABLE)
# ============================================================================

def _detect_base_class(state_dict: dict):
    """Detect which classifier produced this state_dict."""
    if any(k.startswith("film1.") or k.startswith("film_global.")
           for k in state_dict.keys()):
        return FiLMRateClassifier
    return EventClassifier


class RPTTAClassifier(pl.LightningModule):
    """Any-base classifier + RatePrompt at the global-feature bottleneck.

    The base must implement forward_to_bottleneck(data) -> (B, 2*gd) and
    have a `head` attribute. Both EventClassifier and FiLMRateClassifier do.
    """

    def __init__(self, cfg: dict, base_state_dict: dict | None = None,
                 prompt_hidden: int = 32, base_class=None):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(ignore=["base_state_dict", "base_class"])

        if base_class is None:
            base_class = (_detect_base_class(base_state_dict)
                          if base_state_dict is not None else EventClassifier)
        self.base = base_class(cfg)
        if base_state_dict is not None:
            self.base.load_state_dict(base_state_dict, strict=True)
        for p in self.base.parameters():
            p.requires_grad = False

        gd = cfg["model"]["global_dim"]
        self.prompt = RatePrompt(out_dim=gd * 2, hidden=prompt_hidden)

    def forward(self, data):
        feat = self.base.forward_to_bottleneck(data)            # (B, 2*gd)
        lam_log = per_sample_log_rate(data).to(feat.dtype)
        feat = feat + self.prompt(lam_log)
        return self.base.head(feat)

    def _step(self, batch, stage: str):
        y = batch.y.long().view(-1)
        logits = self(batch)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        self.log(f"{stage}_loss", loss, prog_bar=True, batch_size=y.size(0))
        self.log(f"{stage}_acc",  acc,  prog_bar=True, batch_size=y.size(0))
        return loss

    def training_step(self, batch, _):
        return self._step(batch, "train")

    def validation_step(self, batch, _):
        return self._step(batch, "val")

    def configure_optimizers(self):
        # Only prompt + alpha train. Base is frozen.
        trainable = [p for p in self.prompt.parameters()]
        opt = torch.optim.Adam(trainable, lr=self.cfg["train"]["lr"])
        return opt


# ============================================================================
# TEST-TIME ADAPTATION (TENT-STYLE ENTROPY MINIMIZATION)
# ============================================================================

def tent_step(model: RPTTAClassifier, batch, lr: float = 1e-3) -> None:
    """Single entropy-minimization gradient step on prompt params per batch.

    Updates `model.prompt` in-place. Base is untouched.
    Call BEFORE the prediction; predictions for this batch reflect the update.
    """
    optim = torch.optim.Adam(model.prompt.parameters(), lr=lr)
    model.train()  # need grad through prompt
    for p in model.base.parameters():
        p.requires_grad_(False)
    optim.zero_grad()
    logits = model(batch)
    probs = logits.softmax(dim=-1).clamp(min=1e-8)
    ent = -(probs * probs.log()).sum(dim=-1).mean()
    ent.backward()
    optim.step()
    model.eval()


def snapshot_prompt(model: RPTTAClassifier) -> dict:
    """Save prompt state for restore between batches (reset TENT)."""
    return copy.deepcopy(model.prompt.state_dict())


def restore_prompt(model: RPTTAClassifier, state: dict) -> None:
    model.prompt.load_state_dict(state)
