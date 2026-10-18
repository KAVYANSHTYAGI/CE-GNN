"""Event-rate utilities.

Two notions of rate are used throughout the project:

* **natural (global) rate** - ``n_events / duration`` of the *raw* recording,
  stored at preprocessing time as ``log_lambda_raw``. This is the scalar that
  conditions every FiLM layer.
* **local rate** - a k-event sliding-window rate (``lambda_local``), kept per
  event for the (negative-result) RateMoE experiments.
"""
import numpy as np
import torch
from torch_geometric.data import Data


# ----------------------------------------------------------------------------
# Preprocessing-time statistics (numpy)
# ----------------------------------------------------------------------------

def compute_local_rate(t_sorted_sec: np.ndarray, k: int = 20) -> np.ndarray:
    """k-event backward-gap rate (events / second) for time-sorted timestamps."""
    n = len(t_sorted_sec)
    rates = np.zeros(n, dtype=np.float64)
    if n < k + 1:
        if n > 1:
            rates[:] = n / (t_sorted_sec[-1] - t_sorted_sec[0] + 1e-9)
        return rates.astype(np.float32)
    for i in range(k):
        rates[i] = k / (t_sorted_sec[i + k] - t_sorted_sec[i] + 1e-9)
    rates[k:] = k / (t_sorted_sec[k:] - t_sorted_sec[:-k] + 1e-9)
    return rates.astype(np.float32)


def normalized_log_local_rate(t_ms: np.ndarray, k: int = 20) -> np.ndarray:
    """log(lambda_local / median), clipped to [-4, 4] - the per-event rate feature."""
    lam = compute_local_rate(t_ms.astype(np.float64) / 1000.0, k=k)
    med = float(np.median(lam))
    out = np.log(lam / (med + 1e-9) + 1e-9)
    return np.clip(out, -4.0, 4.0).astype(np.float32)


def natural_rate_stats(t_ms: np.ndarray) -> dict:
    """Raw-stream rate statistics, computed *before* any subsampling or rescaling."""
    n_raw = len(t_ms)
    duration_ms = float(t_ms[-1] - t_ms[0] + 1e-6)
    lambda_raw = n_raw * 1000.0 / duration_ms                     # events / second
    return {
        "n_raw": n_raw,
        "duration_ms": duration_ms,
        "lambda_raw": lambda_raw,
        "log_lambda_raw": float(np.log(lambda_raw + 1e-9)),
    }


# ----------------------------------------------------------------------------
# Run-time estimators (torch)
# ----------------------------------------------------------------------------

def sample_rate(data: Data, eps: float = 1e-6) -> float:
    """Events per unit time for a whole sample.

    Args:
        data: torch_geometric Data with pos = (x, y, t).
    Returns:
        Scalar float: number_of_events / time_extent.
    """
    if data.pos.numel() == 0:
        return 0.0
    t = data.pos[:, 2]
    extent = float(t.max() - t.min())
    if extent < eps:
        return 0.0
    return float(data.pos.size(0)) / extent


def dataset_reference_rate(loader, max_batches: int = 50) -> float:
    """Compute the median per-sample rate across a dataset.

    Used to set lambda_train, the reference rate the model was calibrated to.
    Median (not mean) because event-rate distributions are heavy-tailed.
    """
    rates = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        # batch is a PyG Batch object; iterate samples
        for j in range(batch.num_graphs):
            sample = batch.get_example(j)
            r = sample_rate(sample)
            if r > 0:
                rates.append(r)
    if not rates:
        return 1.0
    rates_t = torch.tensor(rates)
    return float(rates_t.median())


def local_density(pos: torch.Tensor, k: int = 8) -> torch.Tensor:
    """Per-node local event density via k-nearest-neighbor distance.

    Higher density (smaller mean kNN distance) -> more redundant region
    (good candidate for subsampling).
    """
    from torch_geometric.nn import knn

    n = pos.size(0)
    if n <= k + 1:
        return torch.ones(n, device=pos.device)

    edge = knn(pos, pos, k=k + 1)
    src, dst = edge[0], edge[1]
    mask = src != dst
    src, dst = src[mask], dst[mask]

    d = (pos[src] - pos[dst]).norm(dim=1)

    # mean kNN distance per src, computed via sum/count (works on old torch)
    sum_d = torch.zeros(n, device=pos.device).scatter_add_(0, src, d)
    cnt = torch.zeros(n, device=pos.device).scatter_add_(
        0, src, torch.ones_like(d)
    )
    mean_d = sum_d / cnt.clamp(min=1.0)

    density = 1.0 / (mean_d + 1e-3)
    return density