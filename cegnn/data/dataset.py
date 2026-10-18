"""Runtime dataset over preprocessed ``.pt`` graphs (any dataset).

Two key features:
1. Loads the CEGNN ``.pt`` samples written by ``cegnn.data.preprocess``.
2. Supports rate resampling: stretch/compress event timestamps to simulate a
   different effective event rate, then rebuild the graph on the fly.

Rate resampling is a transform applied in ``__getitem__`` - NOT a
re-preprocessing step - so a full rate audit takes hours, not days.
"""
import glob
import os
from typing import Callable, Optional

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from .graph import build_graph


# ============================================================================
# RATE RESAMPLING TRANSFORMS
# ============================================================================

class RateResample:
    """Simulate a different event rate by stretching/compressing timestamps.

    rate_factor > 1.0 -> events come faster (compress time axis)
    rate_factor < 1.0 -> events come slower (stretch time axis)

    Important: the SPATIAL distribution is unchanged. Only the time axis
    is rescaled. This isolates the effect of rate on graph structure.

    We do this BEFORE graph construction so the graph reflects the new rate.
    """

    def __init__(self, rate_factor: float):
        self.rate_factor = float(rate_factor)

    def __call__(self, data: Data) -> Data:
        if self.rate_factor == 1.0:
            return data                      # native rate: nothing to do
        # rate_factor > 1 means more events per unit time => compress time
        new_pos = data.pos.clone()           # clone: never modify the cached/loaded sample in place
        new_pos[:, 2] = new_pos[:, 2] / self.rate_factor   # column 2 is time; x/y are untouched
        # Start from a slim Data (no edges): edges depend on the time axis and must be rebuilt
        out = Data(x=data.x, pos=new_pos)
        for attr in ("y", "file_id"):
            if hasattr(data, attr):
                setattr(out, attr, getattr(data, attr))
        # Carry the per-sample rate metadata through the shift so that FiLM sees the *shifted*
        # rate: compressing time by 1/f multiplies the event rate by f, i.e. adds ln f to the log-rate.
        import math
        if hasattr(data, "log_lambda_raw"):
            out.log_lambda_raw = data.log_lambda_raw + math.log(self.rate_factor)
        if hasattr(data, "duration_ms"):
            out.duration_ms = data.duration_ms / self.rate_factor
        if hasattr(data, "n_events_raw"):
            out.n_events_raw = data.n_events_raw
        return out


class RebuildGraph:
    """Rebuild the graph for a sample (after rate resampling, etc.)."""

    def __init__(self, mode: str, cfg: dict, lambda_train: Optional[float] = None):
        self.mode = mode
        self.cfg = cfg
        self.lambda_train = lambda_train

    def __call__(self, data: Data) -> Data:
        return build_graph(data, mode=self.mode, cfg=self.cfg,
                           lambda_train=self.lambda_train)


class Compose:
    """Standard transform composition."""

    def __init__(self, transforms):
        self.transforms = [t for t in transforms if t is not None]

    def __call__(self, data: Data) -> Data:
        for t in self.transforms:
            data = t(data)
        return data


# ============================================================================
# AUGMENTATION (training only)
# ============================================================================

class TrainAugment:
    """Horizontal flip + polarity flip. Matches your existing training code."""

    def __init__(self, hflip_prob: float = 0.5, polflip_prob: float = 0.3,
                 width: float = 119.0):
        self.hflip_prob = hflip_prob
        self.polflip_prob = polflip_prob
        self.width = width

    def __call__(self, data: Data) -> Data:
        if torch.rand(1).item() < self.hflip_prob:
            data.pos[:, 0] = self.width - data.pos[:, 0]
        if torch.rand(1).item() < self.polflip_prob:
            data.x = -data.x
        return data


# ============================================================================
# DATASET
# ============================================================================

class EventGraphDataset(Dataset):
    """Loads preprocessed .pt graphs (formerly ``NCarsGraphDataset``).

    Two modes of operation:

    1. PRE-BUILT GRAPH MODE (default, fast):
       Returns the graph as-stored in the .pt file.
       Use this for baseline training / eval at native rate.

    2. RAW-EVENTS MODE (rate audit):
       Strips edge_index from the loaded sample, applies rate resampling,
       then rebuilds the graph using the configured graph_mode.
       This is what the audit script uses.

    Set rebuild=True and provide rate_factor + graph_mode + cfg to use mode 2.
    """

    def __init__(
        self,
        processed_dir: str,
        split: str,
        cfg: dict,
        # transforms
        augment: bool = False,
        # rate audit
        rebuild: bool = False,
        rate_factor: float = 1.0,
        graph_mode: str = "fixed",
        lambda_train: Optional[float] = None,
    ):
        super().__init__()
        folder = os.path.join(processed_dir, split)
        self.files = sorted(glob.glob(os.path.join(folder, "*.pt")))
        if not self.files:
            raise RuntimeError(f"No .pt files found in {folder}")

        self.cfg = cfg
        self.augment_fn = (
            TrainAugment(
                hflip_prob=cfg["train"]["augment"]["hflip_prob"],
                polflip_prob=cfg["train"]["augment"]["polflip_prob"],
                width=cfg["dataset"]["spatial_dims"][0] - 1,
            ) if augment else None
        )

        self.rebuild = rebuild
        self.rate_factor = rate_factor
        self.graph_mode = graph_mode
        self.lambda_train = lambda_train

        if rebuild:
            self._rebuild_pipeline = Compose([
                RateResample(rate_factor),
                RebuildGraph(graph_mode, cfg, lambda_train),
            ])
        else:
            self._rebuild_pipeline = None

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Data:
        # The .pt files hold PyG Data objects, not bare tensors, so newer torch versions need
        # weights_only=False (older versions do not have the argument at all).
        try:
            data = torch.load(self.files[idx], weights_only=False)
        except TypeError:
            # older torch without weights_only kwarg
            data = torch.load(self.files[idx])

        # Label: every file written by cegnn.data.preprocess embeds `y`. The filename rule below
        # is only a fallback for very old N-Cars files that were written without a label.
        has_embedded_y = (
            hasattr(data, "y")
            and data.y is not None
            and torch.is_tensor(data.y)
            and data.y.numel() > 0
        )
        if not has_embedded_y:
            label = int("cars" in os.path.basename(self.files[idx]))
            data.y = torch.tensor([label], dtype=torch.long)
        else:
            data.y = data.y.long().view(-1)[:1]

        if self._rebuild_pipeline is not None:
            # Rate-shifted evaluation: drop the stored edges (they were built at the native
            # rate), keep the rate metadata, then resample time and rebuild the graph.
            slim = Data(x=data.x, pos=data.pos, y=data.y)
            for attr in ("log_lambda_raw", "n_events_raw", "duration_ms", "file_id"):
                if hasattr(data, attr):
                    setattr(slim, attr, getattr(data, attr))
            data = self._rebuild_pipeline(slim)

        if self.augment_fn is not None:
            data = self.augment_fn(data)

        return data


# Backwards-compatible alias (the class was first written for N-Cars).
NCarsGraphDataset = EventGraphDataset


class RateAugmentedDataset(EventGraphDataset):
    """Training set with a random rate factor per sample (rate augmentation).

    Keeps the per-sample rate metadata (``log_lambda_raw`` ...) and updates it
    consistently with the time rescaling, so FiLM sees the *augmented* rate.
    With ``train_rates=[1.0]`` this is plain natural-rate training.
    """

    _META_KEYS = ("log_lambda_raw", "n_events_raw", "duration_ms")

    def __init__(self, train_rates, **kwargs):
        super().__init__(rebuild=False, **kwargs)
        self.train_rates = list(train_rates)
        self._graph_mode = kwargs.get("graph_mode", "fixed")
        self._cfg = kwargs["cfg"]
        self._lambda_train = kwargs.get("lambda_train", None)

    def __getitem__(self, idx: int) -> Data:
        import random
        try:
            data = torch.load(self.files[idx], weights_only=False)
        except TypeError:
            data = torch.load(self.files[idx])

        has_y = (hasattr(data, "y") and data.y is not None
                 and torch.is_tensor(data.y) and data.y.numel() > 0)
        if has_y:
            data.y = data.y.long().view(-1)[:1]
        else:  # legacy N-Cars files without an embedded label
            data.y = torch.tensor([int("cars" in os.path.basename(self.files[idx]))],
                                  dtype=torch.long)

        slim = Data(x=data.x, pos=data.pos, y=data.y)
        for k in self._META_KEYS:
            if hasattr(data, k):
                setattr(slim, k, getattr(data, k))
        data = RateResample(random.choice(self.train_rates))(slim)   # updates log_lambda_raw
        data = RebuildGraph(self._graph_mode, self._cfg, self._lambda_train)(data)
        if self.augment_fn is not None:
            data = self.augment_fn(data)
        return data
