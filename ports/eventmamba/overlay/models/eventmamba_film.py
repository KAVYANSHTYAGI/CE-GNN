"""eventmamba_film.py -- FiLM-Rate conditioning applied to EventMamba (V2).

Directly mirrors the recipe from the PointNet++ paper: a per-sample centered
log-rate scalar modulates the backbone's per-stage features via FiLM, with
identity initialization so the network starts equal to the unconditioned
EventMamba baseline. Injection points are the exact structural analogs of
our SA1-3 outputs: EventMamba's local_grouper -> local_conv -> attention ->
mamba -> global_conv per stage; we apply FiLM right after each stage's
global_conv (i.e. after Mamba has mixed the stage's tokens), matching where
our own FiLM sits (after each PointNet++ set-abstraction stage).
"""
import torch
import torch.nn as nn
from mamba_layer import MambaBlock
from modules import LocalGrouper
from eventmamba_v2 import Attention, Linear1Layer, Linear2Layer


class FiLMGenerator(nn.Module):
    """Shared MLP: centered log-rate scalar -> (gamma_k, beta_k) per stage.
    Zero-initialized final layer so (gamma,beta) = (0,0) at init, i.e. FiLM
    reduces to the identity map and the model equals the unconditioned
    EventMamba baseline before any training happens."""
    def __init__(self, stage_dims, hidden=32):
        super().__init__()
        self.stage_dims = stage_dims
        total_out = 2 * sum(stage_dims)
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.GELU(),
            nn.Linear(hidden, total_out),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, c):
        # c: [B] centered log-rate -> list of (gamma_i, beta_i), each [B, dim_i]
        out = self.net(c.unsqueeze(-1))
        chunks = torch.split(out, [2 * d for d in self.stage_dims], dim=-1)
        params = []
        for d, chunk in zip(self.stage_dims, chunks):
            gamma, beta = torch.split(chunk, [d, d], dim=-1)
            params.append((gamma, beta))
        return params


class EventMambaFiLM(nn.Module):
    def __init__(self, num_classes=11, feature_list=(16, 32, 64, 128),
                 group_number=(512, 256, 128), neighbors=(24, 24, 24)):
        super().__init__()
        self.feature_list = list(feature_list)
        self.group_number = list(group_number)
        self.neighbors = list(neighbors)
        self.stages = len(group_number)
        self.bimamba_type = 'v2'
        self.local_grouper_list = nn.ModuleList()
        self.local_conv_list = nn.ModuleList()
        self.aggregation_list = nn.ModuleList()
        self.mamba_list = nn.ModuleList()
        self.global_conv_list = nn.ModuleList()
        self.embed_dim = Linear1Layer(3, self.feature_list[0], 1)
        self.attention = Attention(self.feature_list[-1])

        for i in range(self.stages):
            self.local_grouper_list.append(LocalGrouper(
                self.feature_list[i], self.group_number[i], self.neighbors[i],
                use_xyz=False, normalize="anchor"))
            self.local_conv_list.append(Linear2Layer(self.feature_list[i + 1], 1, 1))
            self.aggregation_list.append(Attention(self.feature_list[i + 1]))
            self.mamba_list.append(MambaBlock(dim=self.feature_list[i + 1], layer_idx=0,
                                               bimamba_type=self.bimamba_type))
            self.global_conv_list.append(Linear2Layer(self.feature_list[i + 1], 1, 1))

        # FiLM: one (gamma,beta) pair per stage output, dims = feature_list[1:]
        self.film_generator = FiLMGenerator(self.feature_list[1:])

        self.classifier = nn.Sequential(
            nn.Linear(self.feature_list[-1], 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        # x: [B, 3, N] raw (t,x,y); c: [B] centered log-rate
        xyz = x.permute(0, 2, 1)
        x = self.embed_dim(x)
        x = x.permute(0, 2, 1)
        film_params = self.film_generator(c)
        for i in range(self.stages):
            xyz, x = self.local_grouper_list[i](xyz, x)
            x = x.permute(0, 1, 3, 2)
            b, n, d, s = x.size()
            x = x.reshape(-1, d, s)
            x = self.local_conv_list[i](x)
            x = x.permute(0, 2, 1)
            att = self.aggregation_list[i](x)
            x = torch.bmm(att.unsqueeze(1), x).squeeze(1)
            x = x.reshape(b, n, -1)
            x, _ = self.mamba_list[i](x)
            x = x.permute(0, 2, 1)
            x = self.global_conv_list[i](x)
            x = x.permute(0, 2, 1)
            # ---- FiLM: (1+gamma) * x + beta, broadcast over the n tokens ----
            gamma, beta = film_params[i]
            x = (1 + gamma.unsqueeze(1)) * x + beta.unsqueeze(1)
        attn = self.attention(x)
        x = torch.bmm(attn.unsqueeze(1), x).squeeze(1)
        x = self.classifier(x)
        return x
