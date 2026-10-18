"""graph_res_film.py -- FiLM-Rate conditioning applied to AEGNN's GraphRes.

Third architecture family for the robustness-across-backbones story
(PointNet++ point-cloud -> EventMamba point-cloud+SSM -> this: pure graph-
conv). Unlike EventMamba's per-window normalized point clouds, AEGNN's
edge_attr is rebuilt fresh each forward from raw (beta-rescaled) positions
(_build_edge_attr_cartesian), so a real timestamp rescale genuinely changes
the geometric input the same way it does in the original PointNet++ paper --
no window-size workaround needed here.

FiLM injection points mirror the paper's SA1-3 granularity: after block 1-2
(n[2] channels), after block 5 / before pool5 (n[5] channels), and after
block 6-7 / before pool7 (n[7] channels) -- three stages of increasing
receptive field, same as SA1/SA2/SA3.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch.nn import Linear
from torch_geometric.nn.conv import SplineConv

from aegnn.models.layer import MaxPooling, MaxPoolingX


_EDGE_ATTR_R = 5.0  # BUGFIX: see graph_res.py -- must be the fixed preprocessing
                     # radius, not a per-graph max, or rate-shift rescaling of
                     # pos[:,2] has no effect on the geometry the model sees.


def _build_edge_attr_cartesian(pos, edge_index):
    if edge_index.size(1) == 0:
        return torch.zeros((0, pos.size(1)), dtype=torch.float32, device=pos.device)
    src, dst = edge_index
    delta = pos[dst] - pos[src]
    delta = (delta / (2 * _EDGE_ATTR_R)) + 0.5
    return delta.clamp(0.0, 1.0).float()


def _cartesian_transform(data):
    data.edge_attr = _build_edge_attr_cartesian(data.pos, data.edge_index)
    return data


class FiLMGenerator(nn.Module):
    """centered log-rate scalar -> (gamma_k, beta_k) per stage. Zero-init
    final layer: FiLM = identity at init, model == unconditioned baseline."""
    def __init__(self, stage_dims, hidden=32):
        super().__init__()
        self.stage_dims = stage_dims
        total_out = 2 * sum(stage_dims)
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.GELU(), nn.Linear(hidden, total_out)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, c):
        out = self.net(c.unsqueeze(-1))
        chunks = torch.split(out, [2 * d for d in self.stage_dims], dim=-1)
        return [torch.split(ch, [d, d], dim=-1) for d, ch in zip(self.stage_dims, chunks)]


class GraphResFiLM(nn.Module):
    def __init__(self, dataset, input_shape: torch.Tensor, num_outputs: int,
                 pooling_size=(16, 12), bias: bool = False, root_weight: bool = False,
                 dropout: float = 0.0, edge_dropout: float = 0.0, edge_dim: int = None):
        super().__init__()
        assert len(input_shape) == 3
        dim = 3
        img_w, img_h = float(input_shape[0]), float(input_shape[1])

        if dataset == "ncars":
            kernel_size = 2
            n = [1, 8, 16, 16, 16, 32, 32, 32, 32]
            pooling_outputs = 32
            pooling_size = (8.0, 8.0)
            pool7_voxel = [img_w / 2.0, img_h / 2.0]
            pool7_size = 16
        elif dataset in ("ncaltech101", "gen1"):
            kernel_size = 8
            n = [1, 16, 32, 32, 32, 128, 128, 128]
            pooling_outputs = 128
            pool7_voxel = [img_w / 4.0, img_h / 4.0]
            pool7_size = 16
        else:
            raise NotImplementedError(f"No model parameters for dataset {dataset}")

        self.dropout_p = dropout

        self.conv1 = SplineConv(n[0], n[1], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.norm1 = torch.nn.LayerNorm(n[1])
        self.conv2 = SplineConv(n[1], n[2], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.norm2 = torch.nn.LayerNorm(n[2])
        self.conv3 = SplineConv(n[2], n[3], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.norm3 = torch.nn.LayerNorm(n[3])
        self.conv4 = SplineConv(n[3], n[4], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.norm4 = torch.nn.LayerNorm(n[4])
        self.conv5 = SplineConv(n[4], n[5], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.norm5 = torch.nn.LayerNorm(n[5])
        self.pool5 = MaxPooling(pooling_size, transform=_cartesian_transform)
        self.conv6 = SplineConv(n[5], n[6], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.norm6 = torch.nn.LayerNorm(n[6])
        self.conv7 = SplineConv(n[6], n[7], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.norm7 = torch.nn.LayerNorm(n[7])
        self.pool7 = MaxPoolingX(voxel_size=pool7_voxel, size=pool7_size)
        self.fc = Linear(pooling_outputs * pool7_size, num_outputs, bias=True)

        # FiLM stages: after block1-2 (n[2]), after block5 (n[5]), after block7 (n[7])
        self.film_generator = FiLMGenerator([n[2], n[5], n[7]])

    def forward(self, data: torch_geometric.data.Batch, c: torch.Tensor) -> torch.Tensor:
        # c: [B] centered log-rate, one scalar per graph in the batch
        data.x = data.x.float()
        data.pos = data.pos.float()
        data.edge_attr = _build_edge_attr_cartesian(data.pos, data.edge_index)
        (g1, b1), (g2, b2), (g3, b3) = self.film_generator(c)

        def _film(x, gamma, beta, batch):
            # broadcast per-graph (gamma,beta) to each node via data.batch index
            return (1 + gamma[batch]) * x + beta[batch]

        data.x = F.elu(self.norm1(self.conv1(data.x, data.edge_index, data.edge_attr)))
        data.x = F.elu(self.norm2(self.conv2(data.x, data.edge_index, data.edge_attr)))
        data.x = _film(data.x, g1, b1, data.batch)

        x_sc = data.x.clone()
        data.x = F.elu(self.norm3(self.conv3(data.x, data.edge_index, data.edge_attr)))
        data.x = F.elu(self.norm4(self.conv4(data.x, data.edge_index, data.edge_attr)))
        data.x = data.x + x_sc

        data.x = F.elu(self.norm5(self.conv5(data.x, data.edge_index, data.edge_attr)))
        data.x = _film(data.x, g2, b2, data.batch)
        data.x = torch.nan_to_num(data.x, nan=0.0)

        batch_before_pool = data.batch
        data = self.pool5(data.x, pos=data.pos, batch=data.batch,
                           edge_index=data.edge_index, return_data_obj=True)
        data.num_nodes = data.x.size(0)
        data.edge_attr = _build_edge_attr_cartesian(data.pos, data.edge_index)

        x_sc = data.x.clone()
        data.x = F.elu(self.norm6(self.conv6(data.x, data.edge_index, data.edge_attr)))
        data.x = F.elu(self.norm7(self.conv7(data.x, data.edge_index, data.edge_attr)))
        data.x = data.x + x_sc
        data.x = _film(data.x, g3, b3, data.batch)
        data.x = torch.nan_to_num(data.x, nan=0.0)

        x = self.pool7(data.x, pos=data.pos[:, :2], batch=data.batch)
        x = x.view(-1, self.fc.in_features)
        if self.dropout_p > 0:
            x = F.dropout(x, p=self.dropout_p, training=self.training)
        return self.fc(x)
