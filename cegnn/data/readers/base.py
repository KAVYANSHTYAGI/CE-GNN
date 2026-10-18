"""Adapter interface that every dataset implements.

To add a new dataset you write ONE file in this folder that subclasses
:class:`DatasetAdapter`, implements ``list_samples`` and ``load_events``, and
registers itself with ``@register``. Nothing else in the pipeline changes - see
``docs/ADDING_A_DATASET.md``.
"""
from typing import Dict, List, NamedTuple, Tuple

import numpy as np

# (x_px, y_px, t_ms, polarity) - float32 arrays of equal length, t_ms starting at 0
Events = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def empty_events() -> Events:
    return (np.empty(0, np.float32),) * 4


def to_pm1(pol: np.ndarray) -> np.ndarray:
    """Map polarity {0,1} -> {-1,+1}; leave data already in {-1,+1} untouched."""
    pol = pol.astype(np.int32)
    return (pol * 2 - 1).astype(np.float32) if pol.min() >= 0 else pol.astype(np.float32)


class SampleRef(NamedTuple):
    """One raw sample to process."""
    split: str          # "train" | "test"
    out_name: str       # output file stem, e.g. "airplane_cifar10_airplane_0"
    label: int
    file_id: str        # stored in Data.file_id
    source: object      # path (str) or dataset-specific handle


class DatasetAdapter:
    """Base class. Subclasses are stateless apart from ``raw_root`` / ``cfg``."""

    name: str = ""
    description: str = ""
    num_classes: int = 0
    sensor: Tuple[int, int] = (0, 0)            # (W, H)
    has_official_split: bool = False

    def __init__(self, raw_root: str, cfg: Dict):
        self.raw_root = raw_root
        self.cfg = cfg

    def list_samples(self) -> List[SampleRef]:
        raise NotImplementedError

    def load_events(self, ref: SampleRef) -> Events:
        raise NotImplementedError

    def load(self, ref: SampleRef) -> Tuple[Events, int]:
        """Return ``(events, label)``. Override when the label is only known after reading."""
        return self.load_events(ref), ref.label


_REGISTRY: Dict[str, type] = {}


def register(cls):
    """Class decorator: make an adapter discoverable by name."""
    if not cls.name:
        raise ValueError(f"{cls.__name__} must define a non-empty `name`")
    _REGISTRY[cls.name] = cls
    return cls


def get_adapter(name: str) -> type:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown dataset '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def available_datasets() -> List[str]:
    return sorted(_REGISTRY)
