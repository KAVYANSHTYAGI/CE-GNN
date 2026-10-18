"""Control: concatenate the (centred) log-rate to the pooled descriptor.

Used as an ablation against FiLM - rate is an *input feature* the head may
ignore, instead of modulating every stage.
"""
import torch
import torch.nn as nn

from .backbone import EventClassifier

class RateConditionedClassifier(EventClassifier):
    """EventClassifier + per-sample rate feature concat into head.

    Architecture delta vs EventClassifier:
        head input: 2*gd + 1   (was 2*gd; +1 = normalized log_lambda_raw)

    All other parameters and hooks are inherited.
    """

    # Normalization constants for log_lambda_raw on N-Caltech101 v2.
    # Empirical range observed in test set: ~[10.8, 14.0]; center ~12.5.
    # We center and divide by 1.0 (range already small ~3 nats).
    LAMBDA_CENTER = 12.5

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        m = cfg["model"]
        gd = m["global_dim"]
        # Replace head with one that takes the extra rate-feature dim.
        self.head = nn.Sequential(
            nn.Linear(gd * 2 + 1, m["head_hidden"]),
            nn.BatchNorm1d(m["head_hidden"]),
            nn.ReLU(),
            nn.Dropout(m["dropout"]),
            nn.Linear(m["head_hidden"], cfg["dataset"]["num_classes"]),
        )

    def _rate_feat(self, data) -> torch.Tensor:
        # (B, 1): centered log_lambda_raw. Falls back to zeros if absent
        # (e.g. legacy data without v2 metadata) so the model still runs.
        if hasattr(data, "log_lambda_raw") and data.log_lambda_raw is not None:
            r = data.log_lambda_raw.view(-1, 1).float()
        else:
            B = int(data.batch.max().item()) + 1 if data.batch.numel() > 0 else 1
            r = torch.zeros(B, 1, device=data.x.device)
        return r - self.LAMBDA_CENTER

    def forward(self, data):
        x = data.x.float()
        pos = data.pos.float()
        batch = data.batch
        x, pos, batch = self.sa1(x, pos, batch)
        x, pos, batch = self.sa2(x, pos, batch)
        x, pos, batch = self.sa3(x, pos, batch)
        x = self.sa_global(x, pos, batch)        # (B, 2*gd)
        r = self._rate_feat(data)                # (B, 1)
        return self.head(torch.cat([x, r], dim=-1))
