"""PointNet++ backbone and the baseline event classifier.

Three set-abstraction (SA) stages followed by a global max+mean pool and an MLP
head. No time re-scaling happens inside the model: temporal scale is fixed once
at preprocessing time, so event-rate effects are *not* hidden from the network.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.nn import (
    PointNetConv, fps, radius, global_max_pool, global_mean_pool,
)

# ============================================================================
# PointNet++ blocks
# ============================================================================

class SAModule(nn.Module):
    """Set-abstraction block: FPS sampling + ball-query + PointNetConv."""

    def __init__(self, ratio: float, r: float, nn_mlp: nn.Module):
        super().__init__()
        self.ratio = ratio          # fraction of points kept as centroids
        self.r = r                  # ball-query radius (in x/y/t graph units)
        # PointNetConv applies nn_mlp to [neighbour features, relative position] and max-pools
        self.conv = PointNetConv(nn_mlp, add_self_loops=False)

    def forward(self, x, pos, batch):
        # Farthest-point sampling picks well-spread centroids. NOTE: torch_cluster's fps starts
        # from a RANDOM point by default, which makes every forward pass (also at eval time)
        # slightly stochastic - see docs/REPRODUCING.md, "Evaluation noise".
        idx = fps(pos, batch, ratio=self.ratio)
        # For each centroid, gather up to 64 neighbours within radius r (same graph only).
        row, col = radius(
            pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=64
        )
        # radius() returns (centroid, neighbour); message passing wants (source, target)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        # the coarser point set (features, positions, graph ids) feeds the next stage
        return x, pos[idx], batch[idx]


class GlobalSAModule(nn.Module):
    """Last stage: one MLP over all remaining points, then a graph-level pooling."""

    def __init__(self, nn_mlp: nn.Module):
        super().__init__()
        self.nn = nn_mlp

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        # max captures the strongest local response, mean the overall activity; concatenate both
        x_max = global_max_pool(x, batch)
        x_mean = global_mean_pool(x, batch)
        return torch.cat([x_max, x_mean], dim=1)


def _mlp(in_dim: int, hidden: int, out: int) -> nn.Sequential:
    """Two-layer MLP with BatchNorm + ReLU, the building block of every SA stage."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
        nn.Linear(hidden, out),    nn.BatchNorm1d(out),    nn.ReLU(),
    )


# ============================================================================
# Lightning module
# ============================================================================

class EventClassifier(pl.LightningModule):
    """Baseline PointNet++ event classifier (num_classes from the config)."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.save_hyperparameters(cfg)   # stored in the checkpoint => load_checkpoint() can rebuild the model
        self.cfg = cfg

        m = cfg["model"]
        h1, h2, h3 = m["hidden"]         # feature widths of the three SA stages (32, 64, 128)
        gd = m["global_dim"]             # width before the max+mean pooling (256 => 512 pooled)

        # Input to SA1 is the 1-d polarity (+3 relative-position coordinates added by PointNetConv).
        self.sa1 = SAModule(m["sa1_ratio"], m["sa1_radius"],
                            _mlp(1 + 3, h1, h1))
        self.sa2 = SAModule(m["sa2_ratio"], m["sa2_radius"],
                            _mlp(h1 + 3, h2, h2))
        self.sa3 = SAModule(m["sa3_ratio"], m["sa3_radius"],
                            _mlp(h2 + 3, h3, h3))
        self.sa_global = GlobalSAModule(
            nn.Sequential(
                nn.Linear(h3 + 3, h3), nn.BatchNorm1d(h3), nn.ReLU(),
                nn.Linear(h3, gd),     nn.BatchNorm1d(gd), nn.ReLU(),
            )
        )

        # Classification head on the pooled (max || mean) descriptor.
        self.head = nn.Sequential(
            nn.Linear(gd * 2, m["head_hidden"]),
            nn.BatchNorm1d(m["head_hidden"]),
            nn.ReLU(),
            nn.Dropout(m["dropout"]),
            nn.Linear(m["head_hidden"], cfg["dataset"]["num_classes"]),
        )

    def forward_to_bottleneck(self, data):
        """Run all SA blocks + global pool. Returns pooled features (B, 2*gd).

        This is the input to self.head. Exposed so wrappers (RA-TTA) can tap in
        without re-implementing forward.
        """
        x = data.x.float()       # (N, 1) polarity
        pos = data.pos.float()   # (N, 3) x, y, t
        batch = data.batch       # graph id of every node
        x, pos, batch = self.sa1(x, pos, batch)
        x, pos, batch = self.sa2(x, pos, batch)
        x, pos, batch = self.sa3(x, pos, batch)
        return self.sa_global(x, pos, batch)

    def forward(self, data):
        feat = self.forward_to_bottleneck(data)
        return self.head(feat)

    # --- Lightning hooks ---

    def _step(self, batch, stage: str):
        y = batch.y.long().view(-1)
        logits = self(batch)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        # batch_size is passed explicitly because PyG batches confuse Lightning's size inference
        self.log(f"{stage}_loss", loss, prog_bar=True, batch_size=y.size(0))
        self.log(f"{stage}_acc", acc, prog_bar=True, batch_size=y.size(0))
        return loss

    def training_step(self, batch, _):
        return self._step(batch, "train")

    def validation_step(self, batch, _):
        return self._step(batch, "val")

    def configure_optimizers(self):
        # Adam with a step decay (x0.1 every `lr_step` epochs); values come from the config.
        opt = torch.optim.Adam(self.parameters(), lr=self.cfg["train"]["lr"])
        sch = torch.optim.lr_scheduler.StepLR(
            opt,
            step_size=self.cfg["train"]["lr_step"],
            gamma=self.cfg["train"]["lr_gamma"],
        )
        return [opt], [sch]
