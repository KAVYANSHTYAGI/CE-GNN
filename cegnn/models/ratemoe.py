"""Negative result: RateMoE-GNN (rate-routed mixture of EdgeConv experts).

Kept for completeness / reproducibility of the negative result reported in the
paper; it is not part of the CEGNN method.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_max_pool, global_mean_pool

# ============================================================================
# RateMoE-GNN: Rate-Conditioned Mixture-of-Experts Event GNN
# ============================================================================
#
# Reads the pre-built radius graph (edge_index, edge_attr) from the .pt file
# rather than rebuilding internally via fps/radius (as EventClassifier does).
# This is a different graph paradigm: EdgeConv-style message passing where
# each event is routed to top-K rate-specialist experts.
#
# Inputs expected on the Data object:
#   data.x            : (N, 1)  polarity in {-1, +1}
#   data.lambda_local : (N, 1)  log(λ̂_local / λ_median), clipped to [-4, 4]
#   data.pos          : (N, 3)  [x, y, t]
#   data.edge_index   : (2, E)
#   data.edge_attr    : (E, 3)  [Δx/r, Δy/r, Δt/r]
#   data.batch        : (N,)
# ============================================================================


class EdgeConv(MessagePassing):
    """Single GNN layer with edge-feature-conditioned message passing.

    msg(i←j) = MLP([h_j || edge_attr_ij])
    upd(i)   = LN(MLP([h_i || mean_j msg]))
    """

    def __init__(self, in_channels: int, out_channels: int, edge_dim: int):
        super().__init__(aggr="mean")
        self.msg_mlp = nn.Sequential(
            nn.Linear(in_channels + edge_dim, out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels),
        )
        self.upd_mlp = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels),
        )
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x, edge_index, edge_attr):
        agg = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = self.upd_mlp(torch.cat([x, agg], dim=-1))
        return self.norm(out)

    def message(self, x_j, edge_attr):
        return self.msg_mlp(torch.cat([x_j, edge_attr], dim=-1))


class MultiScaleEdgeAttr(nn.Module):
    """Spatial pass-through + log-spaced Fourier features on Δt.

    in:  (E, 3) [Δx/r, Δy/r, Δt/r]
    out: (E, 3 + 2 * num_freqs)  spatial(2) + raw Δt(1) + [sin, cos] at K freqs
    """

    def __init__(self, num_freqs: int = 4):
        super().__init__()
        freqs = torch.logspace(-1, 2, num_freqs)  # 0.1 ... 100
        self.register_buffer("freqs", freqs)
        self.out_dim = 3 + 2 * num_freqs

    def forward(self, edge_attr):
        spatial = edge_attr[:, :2]
        dt = edge_attr[:, 2:3]
        dt_scaled = dt * self.freqs.unsqueeze(0)
        dt_enc = torch.cat([torch.sin(dt_scaled), torch.cos(dt_scaled)], dim=-1)
        return torch.cat([spatial, dt, dt_enc], dim=-1)


class ExpertLayer(nn.Module):
    """Rate-specialist GNN layer with learnable temporal decay τ_k.

    The expert downweights temporally distant neighbors by exp(-|Δt|/τ_k),
    so different experts can specialise on different temporal scales.
    """

    def __init__(self, hidden_dim: int, edge_dim: int, tau_init: float):
        super().__init__()
        # tau_raw is unconstrained; tau = softplus(tau_raw) > 0
        # Inverse softplus: tau_raw = log(exp(tau_init) - 1). For tau_init > 20
        # this overflows, so use the identity tau_raw ≈ tau_init when large.
        tau_init = float(max(tau_init, 1e-3))
        if tau_init > 20.0:
            tau_raw_init = tau_init
        else:
            tau_raw_init = float(torch.log(torch.expm1(torch.tensor(tau_init))).item())
        self.tau_raw = nn.Parameter(torch.tensor(tau_raw_init, dtype=torch.float32))
        # The conv layer sees edge_dim + 1 (extra channel = decay weight)
        self.gnn = EdgeConv(hidden_dim, hidden_dim, edge_dim=edge_dim + 1)

    @property
    def tau(self):
        return F.softplus(self.tau_raw)

    def forward(self, x, edge_index, edge_attr, dt_raw):
        tau = self.tau
        decay = torch.exp(-dt_raw.abs() / (tau + 1e-6))
        decay = decay.clamp(max=1.0)  # safety
        edge_feat = torch.cat([edge_attr, decay.unsqueeze(-1)], dim=-1)
        return self.gnn(x, edge_index, edge_feat)


class RateRouter(nn.Module):
    """Top-K gating over experts conditioned on local rate + backbone features.

    Bottleneck design (16+16 dims) keeps the feature-routing path narrow
    so the rate signal dominates, not class shortcuts.
    """

    def __init__(self, hidden_dim: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.rate_proj = nn.Sequential(
            nn.Linear(1, 16), nn.GELU(), nn.Linear(16, 16),
        )
        self.feat_proj = nn.Sequential(
            nn.Linear(hidden_dim, 16), nn.GELU(),
        )
        self.gate_head = nn.Linear(32, num_experts, bias=False)
        self.temperature = 1.0

    def forward(self, lambda_log, h):
        rate_emb = self.rate_proj(lambda_log)             # (N, 16)
        feat_emb = self.feat_proj(h)                      # (N, 16)
        logits = self.gate_head(torch.cat([rate_emb, feat_emb], dim=-1))
        logits = logits / self.temperature

        # Guard against top_k > num_experts when N is tiny or K is misset
        k = min(self.top_k, logits.size(1))
        top_k_logits, top_k_indices = logits.topk(k, dim=-1)

        sparse_logits = torch.full_like(logits, float("-inf"))
        sparse_logits.scatter_(1, top_k_indices, top_k_logits)
        gates = torch.softmax(sparse_logits, dim=-1)      # top_k nonzero per row
        return gates, top_k_indices


def load_balance_loss(gates: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Shazeer (2017) load-balance loss: K * Σ_k f_k * P_k.

    f_k = fraction routed to expert k, P_k = mean gate prob for expert k.
    """
    routing_mask = (gates > 0).float()
    f_k = routing_mask.mean(dim=0)
    P_k = gates.mean(dim=0)
    return num_experts * (f_k * P_k).sum()


def tau_diversity_loss(expert_layers: nn.ModuleList) -> torch.Tensor:
    """Push τ values apart: minimise -mean_{j≠k}(τ_j − τ_k)²."""
    taus = torch.stack([e.tau for e in expert_layers])
    diff = taus.unsqueeze(0) - taus.unsqueeze(1)
    sq = diff.pow(2).clamp(min=1e-8)
    mask = ~torch.eye(len(taus), dtype=torch.bool, device=taus.device)
    return -sq[mask].mean()


class RateMoEGNN(nn.Module):
    """Rate-Conditioned MoE Event GNN.

    Reads pre-built radius graph from data.edge_index/edge_attr.
    Node input is [polarity, log_lambda_local] (assembled in forward).
    """

    def __init__(
        self,
        in_channels: int = 2,
        hidden_dim: int = 64,
        num_classes: int = 2,
        num_experts: int = 4,
        top_k: int = 2,
        num_freqs: int = 4,
        tau_inits=None,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        if tau_inits is None:
            tau_inits = [40.0, 15.0, 6.0, 2.0][:num_experts]
            # if K > 4, geometrically extend toward 1.0
            while len(tau_inits) < num_experts:
                tau_inits.append(tau_inits[-1] * 0.5)
        assert len(tau_inits) == num_experts

        self.edge_encoder = MultiScaleEdgeAttr(num_freqs=num_freqs)
        edge_dim = self.edge_encoder.out_dim

        self.node_embed = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.backbone = nn.ModuleList([
            EdgeConv(hidden_dim, hidden_dim, edge_dim=edge_dim),
            EdgeConv(hidden_dim, hidden_dim, edge_dim=edge_dim),
        ])

        self.router = RateRouter(hidden_dim, num_experts, top_k)
        self.experts = nn.ModuleList([
            ExpertLayer(hidden_dim, edge_dim, tau_init=tau_inits[k])
            for k in range(num_experts)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, data):
        # Assemble node input: [polarity, log_lambda_local]
        # If lambda_local is missing (e.g. legacy .pt file), default to zeros.
        polarity = data.x.float()                         # (N, 1)
        if hasattr(data, "lambda_local") and data.lambda_local is not None:
            lambda_log = data.lambda_local.float()        # (N, 1)
        else:
            lambda_log = torch.zeros_like(polarity)
        node_in = torch.cat([polarity, lambda_log], dim=-1)  # (N, 2)

        edge_index = data.edge_index
        edge_attr = data.edge_attr.float()
        batch = data.batch

        edge_feat = self.edge_encoder(edge_attr)          # (E, edge_dim)
        dt_raw = edge_attr[:, 2]                          # (E,)

        h = self.node_embed(node_in)
        for layer in self.backbone:
            h = h + layer(h, edge_index, edge_feat)       # residual

        gates, active_indices = self.router(lambda_log, h)

        # Only compute experts that are actually used in this batch
        unique_experts = active_indices.unique().tolist()
        expert_out = torch.zeros_like(h)
        for k in unique_experts:
            gate_k = gates[:, k]
            if (gate_k > 0).any():
                h_k = self.experts[k](h, edge_index, edge_feat, dt_raw)
                expert_out = expert_out + gate_k.unsqueeze(-1) * h_k

        h = h + expert_out                                # residual mix

        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        h_pool = torch.cat([h_mean, h_max], dim=-1)
        logits = self.classifier(h_pool)

        aux = {
            "lb_loss": load_balance_loss(gates, self.num_experts),
            "div_loss": tau_diversity_loss(self.experts),
            "gates": gates.detach(),
        }
        return logits, aux

    def get_tau_values(self):
        return [float(e.tau.item()) for e in self.experts]
