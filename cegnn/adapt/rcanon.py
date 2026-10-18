"""Rate-Canonicalizer (R-Canon) — v2 design.

A learnable, content-aware per-scene rate canonicalizer.  Two-pass forward:

    Pass 1 (frozen base, no grad):  data  ->  SA modules  ->  global_feat (B, 2*gd)
    R-Canon predicts log(rho*) from [global_feat || scene_stats]   (B, 1)
    Pass 2 (frozen base, with grad through canonicalization):
                                    data + rho*  ->  SA modules  ->  head  ->  logits

The base classifier remains frozen.  Only R-Canon trains.

Why two passes:
    With only 3 scalar scene stats, R-Canon's MLP has nothing to learn from -- a
    bigger MLP on 3 features wastes parameters.  Conditioning on the base's
    global features gives R-Canon true scene-content awareness and lets us scale
    the MLP to a size comparable to the base (~200K params).

Math:
    For a sample E with raw rate lambda and duration T:
        new_pos[:, 2] = pos[:, 2] * rho*        (time scale)
        new_lambda    = lambda * rho*
        new_duration  = T / rho*

Training objective (combined):
    SUP   : cross-entropy on canonicalized original sample
    CONS  : KL between two independent random rate-perturbations of the same E,
            forcing the canonicalizer to absorb perturbations.
    REG   : (mean log rho*)^2  -- keep canonicalizer near identity in expectation
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.data import Data

from ..models import EventClassifier


# ============================================================================
# SCENE FEATURE EXTRACTION  (scalar per-sample fields embedded by v2 preprocess)
# ============================================================================

def scene_features(data) -> torch.Tensor:
    """Returns (B, 3): log lambda_raw, log N, log T_ms."""
    log_lam = data.log_lambda_raw.view(-1, 1).float()
    log_N   = torch.log(data.n_events_raw.view(-1, 1).float().clamp(min=1.0))
    log_T   = torch.log(data.duration_ms.view(-1, 1).float().clamp(min=1.0))
    return torch.cat([log_lam, log_N, log_T], dim=-1)


# ============================================================================
# RCANON MODULE — bigger and content-aware
# ============================================================================

class RCanon(nn.Module):
    """Predicts log(rho*) from [global_feat || scene_stats].

    Default architecture (in=515, hidden=256, depth=4) ~= 200K params.
    Last layer is zero-init so the canonicalizer starts as identity (log rho* = 0).
    """

    def __init__(self, in_features: int, hidden: int = 256, depth: int = 4,
                 dropout: float = 0.1, last_init_std: float = 1e-3):
        super().__init__()
        layers = [nn.Linear(in_features, hidden), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(depth - 2):
            layers += [nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout)]
        last = nn.Linear(hidden, 1)
        # near-identity init: small Gaussian on weights, zero bias.
        # zero-init weights would BLOCK gradient through all earlier layers
        # (d_loss/d_hidden = grad @ W_last.T = 0 when W_last is exactly 0).
        # std=1e-3 keeps output magnitude ~0.01 (rho ~ 1.01) while letting
        # gradients flow normally on the very first step.
        nn.init.normal_(last.weight, std=last_init_std)
        nn.init.zeros_(last.bias)
        layers.append(last)
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


# ============================================================================
# DATA UTILITY  (in-place-safe per-sample time rescale)
# ============================================================================

def apply_rho_per_sample(data, log_rho: torch.Tensor) -> Data:
    """Rescale data.pos[:, 2] per-sample by exp(log_rho). Returns NEW Data.

    Uses torch.cat instead of slice-assignment so autograd is happy across
    multiple successive applications in one training step.
    """
    log_rho_per_node = log_rho.view(-1)[data.batch]            # (N,)
    rho_per_node     = log_rho_per_node.exp()                  # (N,)

    new_t = data.pos[:, 2] * rho_per_node                      # (N,)
    new_pos = torch.cat(
        [data.pos[:, :2], new_t.unsqueeze(-1)], dim=-1
    )                                                          # (N, 3)

    out = Data(x=data.x, pos=new_pos, y=data.y)
    out.batch = data.batch
    # update per-sample metadata so a second call (or downstream scene_features)
    # sees consistent state
    out.log_lambda_raw = data.log_lambda_raw.view(-1, 1) + log_rho
    out.duration_ms    = data.duration_ms.view(-1, 1) / log_rho.exp()
    out.n_events_raw   = data.n_events_raw
    return out


# ============================================================================
# CANONICALIZED CLASSIFIER (base frozen, R-Canon trainable)
# ============================================================================

class CanonicalizedClassifier(pl.LightningModule):
    """Two-pass forward = (extract scene features) -> R-Canon -> (classify rescaled)."""

    def __init__(self, cfg: dict, base_state_dict: dict | None = None,
                 hidden: int = 256, depth: int = 4, dropout: float = 0.1,
                 lambda_cons: float = 0.5, lambda_reg: float = 0.01,
                 aug_max_log_rho: float = math.log(2.0)):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(ignore=["base_state_dict"])

        self.base = EventClassifier(cfg)
        if base_state_dict is not None:
            self.base.load_state_dict(base_state_dict, strict=True)
        for p in self.base.parameters():
            p.requires_grad = False

        gd = cfg["model"]["global_dim"]
        in_features = 2 * gd + 3                      # global_feat || scene_stats
        self.canon = RCanon(in_features=in_features,
                            hidden=hidden, depth=depth, dropout=dropout)

        self.lambda_cons     = lambda_cons
        self.lambda_reg      = lambda_reg
        self.aug_max_log_rho = aug_max_log_rho

    # ------------------------------------------------------------------ pass 1

    @torch.no_grad()
    def _extract_global_feat(self, data) -> torch.Tensor:
        """Run base SA modules + global pool, no grad.  Returns (B, 2*gd)."""
        # base is in eval mode by default; explicit to be sure
        was_training = self.base.training
        self.base.eval()
        try:
            x = data.x.float()
            pos = data.pos.float()
            batch = data.batch
            x, pos, batch = self.base.sa1(x, pos, batch)
            x, pos, batch = self.base.sa2(x, pos, batch)
            x, pos, batch = self.base.sa3(x, pos, batch)
            feat = self.base.sa_global(x, pos, batch)
        finally:
            if was_training:
                self.base.train()
        return feat

    # ------------------------------------------------------------------ forward

    def forward(self, data):
        global_feat = self._extract_global_feat(data)         # (B, 2*gd)
        stats       = scene_features(data)                    # (B, 3)
        canon_in    = torch.cat([global_feat, stats], dim=-1) # (B, 2*gd + 3)
        log_rho     = self.canon(canon_in)                    # (B, 1)
        canon_data  = apply_rho_per_sample(data, log_rho)
        logits      = self.base(canon_data)
        return logits, log_rho

    # ------------------------------------------------------------------ training

    def _consistency_pair(self, batch):
        """Return (logits_a, logits_b) from two independent rate-perturbations."""
        B = int(batch.batch.max().item()) + 1 if batch.batch.numel() > 0 else 1
        dev = batch.pos.device
        L = self.aug_max_log_rho
        rho_a = (torch.rand(B, 1, device=dev) * 2 - 1) * L
        rho_b = (torch.rand(B, 1, device=dev) * 2 - 1) * L
        batch_a = apply_rho_per_sample(batch, rho_a)
        batch_b = apply_rho_per_sample(batch, rho_b)
        logits_a, _ = self(batch_a)
        logits_b, _ = self(batch_b)
        return logits_a, logits_b

    def training_step(self, batch, _):
        y = batch.y.long().view(-1)
        # 1) supervised on canonicalized original
        logits_orig, log_rho_orig = self(batch)
        sup = F.cross_entropy(logits_orig, y)

        # 2) consistency between two independent rate-perturbations
        logits_a, logits_b = self._consistency_pair(batch)
        log_pa = F.log_softmax(logits_a, dim=-1)
        log_pb = F.log_softmax(logits_b, dim=-1)
        cons = 0.5 * (
            F.kl_div(log_pa, log_pb.exp(), reduction="batchmean") +
            F.kl_div(log_pb, log_pa.exp(), reduction="batchmean")
        )

        # 3) reg: keep log rho* near 0 in expectation
        reg = (log_rho_orig.mean()) ** 2

        loss = sup + self.lambda_cons * cons + self.lambda_reg * reg
        acc = (logits_orig.argmax(1) == y).float().mean()
        B = int(batch.batch.max().item()) + 1
        self.log("train_loss", loss, prog_bar=True, batch_size=B)
        self.log("train_acc",  acc,  prog_bar=True, batch_size=B)
        self.log("train_sup",  sup.detach(),  batch_size=B)
        self.log("train_cons", cons.detach(), batch_size=B)
        self.log("train_reg",  reg.detach(),  batch_size=B)
        self.log("train_log_rho_mean", log_rho_orig.mean().detach(), batch_size=B)
        self.log("train_log_rho_std",  log_rho_orig.std().detach(),  batch_size=B)
        return loss

    def validation_step(self, batch, _):
        y = batch.y.long().view(-1)
        logits_orig, log_rho_orig = self(batch)
        sup = F.cross_entropy(logits_orig, y)
        acc = (logits_orig.argmax(1) == y).float().mean()
        B = int(batch.batch.max().item()) + 1
        self.log("val_loss",  sup, prog_bar=True, batch_size=B)
        self.log("val_acc",   acc, prog_bar=True, batch_size=B)
        self.log("val_log_rho_mean", log_rho_orig.mean(), batch_size=B)
        return sup

    def configure_optimizers(self):
        # Only the canon trains.  Base is frozen.
        return torch.optim.Adam(self.canon.parameters(), lr=self.cfg["train"]["lr"])
