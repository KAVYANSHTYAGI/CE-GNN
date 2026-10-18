"""Rate-FiLM: identity-initialised feature-wise modulation conditioned on the
per-sample log event rate (the CEGNN conditioning mechanism).

The last linear layer of every FiLM gating MLP is zero-initialised, so at step 0
gamma = beta = 0 and the network is *exactly* the unconditioned baseline.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import EventClassifier

# Why FiLM at *every* stage rather than one concatenated scalar at the head?
# A scalar at the head is easy for the network to ignore (see rate_conditioned.py and the
# ablations in the paper). Modulating the features after each set-abstraction stage lets the
# rate change how local neighbourhoods are summarised, before they are pooled away.
# Because the gating MLPs start as the identity, training begins at the baseline and can only
# move away from it if the rate turns out to be useful.


class FiLM(nn.Module):
    """Feature-wise linear modulation: y = (1 + gamma(cond)) * x + beta(cond).

    Last linear is zero-init -> identity transform at init.
    """

    def __init__(self, feat_dim: int, cond_dim: int = 1, hidden: int = 32):
        super().__init__()
        # tiny gating MLP: scalar condition -> hidden -> (gamma, beta) for every feature channel
        self.fc1 = nn.Linear(cond_dim, hidden)
        self.fc2 = nn.Linear(hidden, 2 * feat_dim)
        # zero init => gamma = beta = 0 => y = x (identity) at step 0
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        self.feat_dim = feat_dim

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gb = self.fc2(F.gelu(self.fc1(cond)))                # (..., 2*feat_dim)
        gamma, beta = gb.chunk(2, dim=-1)                    # per-channel scale and shift
        return (1.0 + gamma) * x + beta                      # "1 +" keeps the identity at init


class FiLMRateClassifier(EventClassifier):
    """EventClassifier + FiLM modulation after every SA stage, cond on log_lambda_raw.

    All FiLM layers identity-init -> equivalent to baseline at step 0.
    """

    # The rate is centred before it enters FiLM. 12.5 is the middle of N-Caltech101's
    # log-rate range; the same constant is used for every dataset (the checkpoints were
    # trained that way). Other datasets sit off-centre (e.g. DVS Gesture ~10.7); the first
    # FiLM linear layer absorbs the offset, so changing the constant is not needed.
    LAMBDA_CENTER = 12.5

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        m = cfg["model"]
        h1, h2, h3 = m["hidden"]
        gd = m["global_dim"]
        # one FiLM per feature width: after SA1, SA2, SA3 and after the global pool
        self.film1 = FiLM(h1)
        self.film2 = FiLM(h2)
        self.film3 = FiLM(h3)
        self.film_global = FiLM(2 * gd)

    def _rate(self, data) -> torch.Tensor:
        """Per-graph conditioning scalar: centred log event rate of the raw recording."""
        if hasattr(data, "log_lambda_raw") and data.log_lambda_raw is not None:
            r = data.log_lambda_raw.view(-1, 1).float()
        else:
            # legacy data without rate metadata: condition on a constant (model degrades to baseline)
            B = int(data.batch.max().item()) + 1 if data.batch.numel() > 0 else 1
            r = torch.zeros(B, 1, device=data.x.device)
        return r - self.LAMBDA_CENTER                           # (B, 1)

    def forward_to_bottleneck(self, data):
        rate = self._rate(data)                                 # (B, 1)
        x = data.x.float()
        pos = data.pos.float()
        batch = data.batch
        # rate[batch] broadcasts each graph's scalar to all of that graph's nodes
        x, pos, batch = self.sa1(x, pos, batch)
        x = self.film1(x, rate[batch])                          # (N1, h1)
        x, pos, batch = self.sa2(x, pos, batch)
        x = self.film2(x, rate[batch])                          # (N2, h2)
        x, pos, batch = self.sa3(x, pos, batch)
        x = self.film3(x, rate[batch])                          # (N3, h3)
        x = self.sa_global(x, pos, batch)                       # (B, 2*gd)
        return self.film_global(x, rate)                        # (B, 2*gd), one rate per graph


class FiLMGlobalOnlyClassifier(EventClassifier):
    """Ablation: FiLM ONLY at the global pool output, not at per-node stages.

    Tests whether a single FiLM at the bottleneck recovers full FiLM-rate's
    benefit (it does not: 60.3 vs 64.4 on N-Caltech101). Adds ~34K parameters.
    """

    LAMBDA_CENTER = 12.5

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        gd = cfg["model"]["global_dim"]
        self.film_global = FiLM(2 * gd)

    def _rate(self, data) -> torch.Tensor:
        if hasattr(data, "log_lambda_raw") and data.log_lambda_raw is not None:
            r = data.log_lambda_raw.view(-1, 1).float()
        else:
            B = int(data.batch.max().item()) + 1 if data.batch.numel() > 0 else 1
            r = torch.zeros(B, 1, device=data.x.device)
        return r - self.LAMBDA_CENTER

    def forward_to_bottleneck(self, data):
        rate = self._rate(data)
        x = data.x.float()
        pos = data.pos.float()
        batch = data.batch
        # plain baseline path ...
        x, pos, batch = self.sa1(x, pos, batch)
        x, pos, batch = self.sa2(x, pos, batch)
        x, pos, batch = self.sa3(x, pos, batch)
        x = self.sa_global(x, pos, batch)
        # ... with the rate entering only after pooling
        return self.film_global(x, rate)
