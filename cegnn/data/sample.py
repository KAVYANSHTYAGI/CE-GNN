"""The CEGNN sample schema and the events -> graph conversion shared by all datasets.

Every dataset adapter ends up here: it hands over four aligned arrays
``(x, y, t_ms, polarity)`` and this module produces *one* PyG ``Data`` object
with the following fields (identical for every dataset):

==================  ========  ==================================================
field               shape     meaning
==================  ========  ==================================================
``x``               (N, 1)    polarity in {-1, +1}
``pos``             (N, 3)    [x_px, y_px, t] with t in graph-time units
                              (``t_unit`` graph units per ms; default 1 = 1 ms)
``y``               (1,)      class label
``log_lambda_raw``  (1,)      ln(events/sec) of the *raw* recording  (FiLM input)
``lambda_raw``      (1,)      events/sec of the raw recording
``n_events_raw``    (1,)      raw event count (before subsampling)
``duration_ms``     (1,)      raw recording duration in ms
``lambda_global``   (1,)      alias of ``log_lambda_raw`` (kept for old code)
``lambda_local``    (N, 1)    per-event log-rate relative to the clip median
``edge_index``      (2, E)    spatio-temporal radius graph (r, d_max from config)
``edge_attr``       (E, 3)    (dx, dy, dt) / r
``file_id``         str       source file / sample id
==================  ========  ==================================================

Crucially the *natural duration is preserved* (no per-sample time
normalisation), so ``log_lambda_raw`` carries real information about the
recording instead of being erased by preprocessing.
"""
from typing import Optional

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn.pool import radius_graph

from .rate import natural_rate_stats, normalized_log_local_rate


def events_to_sample(
    x: np.ndarray,
    y: np.ndarray,
    t_ms: np.ndarray,
    pol: np.ndarray,
    label: int,
    file_id: str,
    *,
    t_unit: float,
    r: float,
    d_max: int,
    subsample_K: int,
    n_max: int,
) -> Optional[Data]:
    """Convert one raw event stream into a CEGNN sample.

    Returns ``None`` for degenerate recordings (fewer than 2 events, or a graph
    with no edges); callers simply skip those.
    """
    if len(t_ms) < 2:
        return None

    stats = natural_rate_stats(t_ms)                       # BEFORE any subsampling

    t_graph = t_ms * float(t_unit)
    events = torch.from_numpy(np.column_stack((x, y, t_graph, pol))).float()

    data = Data(x=events[:, 3:4], pos=events[:, :3])
    data.y = torch.tensor([label], dtype=torch.long)
    data.file_id = file_id

    data.duration_ms = torch.tensor([stats["duration_ms"]], dtype=torch.float32)
    data.n_events_raw = torch.tensor([stats["n_raw"]], dtype=torch.long)
    data.lambda_raw = torch.tensor([stats["lambda_raw"]], dtype=torch.float32)
    data.log_lambda_raw = torch.tensor([stats["log_lambda_raw"]], dtype=torch.float32)
    data.lambda_global = torch.tensor([stats["log_lambda_raw"]], dtype=torch.float32)
    data.lambda_local = torch.from_numpy(normalized_log_local_rate(t_ms)).unsqueeze(1)

    # uniform K-step subsample, capped at n_max nodes
    n = data.pos.size(0)
    if n > n_max:
        idx = torch.arange(0, n, step=int(subsample_K))
        if len(idx) > n_max:
            idx = idx[:n_max]
        data.pos = data.pos[idx]
        data.x = data.x[idx]
        data.lambda_local = data.lambda_local[idx]

    # spatio-temporal radius graph
    data.edge_index = radius_graph(data.pos, r=r, max_num_neighbors=d_max, loop=False)
    if data.edge_index.size(1) == 0:
        return None
    src, dst = data.edge_index
    data.edge_attr = ((data.pos[dst] - data.pos[src]) / r).float()

    assert data.x.shape[0] == data.pos.shape[0] == data.lambda_local.shape[0]
    return data
