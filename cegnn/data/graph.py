"""Graph construction: fixed (baseline) and rate-adaptive variants.

Two operations are performed:
  1. SUBSAMPLING: pick which events become nodes
  2. EDGE CONSTRUCTION: connect nodes within radius r

Each can be FIXED (baseline) or ADAPTIVE (rate-aware).
"""
from typing import Optional

import torch
from torch_geometric.data import Data
from torch_geometric.nn.pool import radius_graph

from .rate import sample_rate, local_density


# ============================================================================
# SUBSAMPLING
# ============================================================================

def subsample_uniform(data: Data, K: int, n_max: int) -> Data:
    """Baseline: keep every K-th event after time-sorting, cap at n_max.

    This is what AEGNN / NVS-S / your current pipeline do.
    """
    n = data.pos.size(0)
    if n <= 1:
        return data

    # ensure sorted by time
    order = data.pos[:, 2].argsort()
    pos = data.pos[order]
    x = data.x[order]

    if n > n_max:
        idx = torch.arange(0, n, step=K)
        if len(idx) > n_max:
            idx = idx[:n_max]
        pos = pos[idx]
        x = x[idx]

    out = Data(x=x, pos=pos)
    for attr in ("y", "file_id", "log_lambda_raw", "n_events_raw", "duration_ms"):
        if hasattr(data, attr):
            setattr(out, attr, getattr(data, attr))
    return out


def subsample_density_aware(
    data: Data,
    target_count: int,
    k_neighbors: int = 8,
    seed: Optional[int] = None,
) -> Data:
    """Density-aware: drop more events from over-sampled regions.

    Computes per-event local density (inverse kNN distance) and keeps each
    event with probability inversely proportional to density, normalized so
    the expected kept count is `target_count`.

    Args:
        data: PyG Data with pos (N, 3), x (N, F).
        target_count: desired number of nodes after subsampling.
        k_neighbors: k for kNN density estimate.
        seed: RNG seed for reproducible eval; None = random.
    """
    n = data.pos.size(0)
    if n <= target_count or n <= k_neighbors + 2:
        return data

    # ensure sorted by time (for consistency with other code paths)
    order = data.pos[:, 2].argsort()
    pos = data.pos[order]
    x = data.x[order]

    density = local_density(pos, k=k_neighbors)
    # weight = 1 / density, but clamped to avoid nuking sparse regions entirely
    weight = 1.0 / (density + 1e-3)
    # normalize so expected kept count = target_count
    p_keep = weight * (target_count / weight.sum())
    p_keep = p_keep.clamp(0.0, 1.0)

    if seed is not None:
        g = torch.Generator(device=pos.device).manual_seed(seed)
        keep = torch.rand(n, generator=g, device=pos.device) < p_keep
    else:
        keep = torch.rand(n, device=pos.device) < p_keep

    # safety: if we got too few or zero, fall back to top-target_count by p_keep
    if int(keep.sum()) < max(64, target_count // 4):
        topk = torch.topk(p_keep, k=target_count).indices
        keep = torch.zeros(n, dtype=torch.bool, device=pos.device)
        keep[topk] = True

    out = Data(x=x[keep], pos=pos[keep])
    for attr in ("y", "file_id", "log_lambda_raw", "n_events_raw", "duration_ms"):
        if hasattr(data, attr):
            setattr(out, attr, getattr(data, attr))
    return out


# ============================================================================
# EDGE CONSTRUCTION
# ============================================================================

def build_edges_fixed(data: Data, r: float, d_max: int) -> Data:
    """Baseline: radius_graph with fixed r."""
    data.edge_index = radius_graph(
        data.pos, r=r, max_num_neighbors=d_max, loop=False
    )
    return data


def build_edges_adaptive(
    data: Data,
    r0: float,
    d_max: int,
    lambda_train: float,
    alpha_R: float = -1.0 / 3.0,
    eps: float = 1e-6,
) -> Data:
    """Rate-adaptive radius.

    R(sample) = r0 * (lambda_hat / lambda_train) ** alpha_R

    The default alpha_R = -1/3 holds expected neighbor count approximately
    constant under uniform 3D rescaling of the event cloud, since events
    fill a 3D volume in (x, y, t).

    NOTE: We adapt at the SAMPLE level here (one r per sample) rather than
    per-node, because for short clips like N-Cars samples, sample-level
    adaptation is the dominant effect and is much cheaper than per-node.
    Per-node adaptation is left for future work / N-Caltech101.
    """
    lam = sample_rate(data)
    if lam <= eps or lambda_train <= eps:
        r_eff = r0
    else:
        ratio = lam / lambda_train
        r_eff = r0 * (ratio ** alpha_R)
        # safety clamp - never let r blow up or vanish
        r_eff = float(max(0.25 * r0, min(4.0 * r0, r_eff)))

    data.edge_index = radius_graph(
        data.pos, r=r_eff, max_num_neighbors=d_max, loop=False
    )
    # store the effective radius for diagnostics
    data.r_eff = torch.tensor([r_eff])
    return data


# ============================================================================
# EDGE FEATURES
# ============================================================================

def add_edge_attr(data: Data, r_norm: float) -> Data:
    """Compute edge attributes as normalized relative position."""
    if data.edge_index.size(1) == 0:
        data.edge_attr = torch.zeros((0, 3), dtype=torch.float32)
        return data
    src, dst = data.edge_index
    delta = data.pos[dst] - data.pos[src]
    data.edge_attr = (delta / r_norm).float()
    return data


# ============================================================================
# UNIFIED INTERFACE
# ============================================================================

def build_graph(
    data: Data,
    mode: str,
    cfg: dict,
    lambda_train: Optional[float] = None,
) -> Data:
    """Single entry point.

    mode: one of
        'fixed'              - baseline: uniform subsample + fixed-r edges
        'adapt_radius'       - uniform subsample + adaptive-r edges
        'adapt_subsample'    - density-aware subsample + fixed-r edges
        'adapt_both'         - density-aware subsample + adaptive-r edges  (RA-GNN)
    """
    K = cfg["graph"]["subsample_K"]
    n_max = cfg["graph"]["n_samples"]
    r0 = cfg["graph"]["r"]
    d_max = cfg["graph"]["d_max"]

    # subsample
    if mode in ("fixed", "adapt_radius"):
        data = subsample_uniform(data, K=K, n_max=n_max)
    elif mode in ("adapt_subsample", "adapt_both"):
        # target the same expected count as uniform subsampling for fairness
        n_orig = data.pos.size(0)
        target = min(n_max, max(64, n_orig // K))
        data = subsample_density_aware(data, target_count=target)
    else:
        raise ValueError(f"Unknown graph mode: {mode}")

    # edges
    if mode in ("fixed", "adapt_subsample"):
        data = build_edges_fixed(data, r=r0, d_max=d_max)
    elif mode in ("adapt_radius", "adapt_both"):
        if lambda_train is None:
            raise ValueError("adaptive-radius modes require lambda_train")
        alpha_R = cfg.get("rate_adaptive", {}).get("alpha_R", -1.0 / 3.0)
        data = build_edges_adaptive(
            data, r0=r0, d_max=d_max,
            lambda_train=lambda_train, alpha_R=alpha_R,
        )
    return add_edge_attr(data, r_norm=r0)