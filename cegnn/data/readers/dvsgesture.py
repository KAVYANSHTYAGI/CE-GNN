"""DVS128 Gesture (Amir et al., 2017) - 11 gestures, DVS128 128x128, official split.

Raw layout (what ``scripts/data/download_datasets.py`` produces; the layout
``tonic`` expects)::

    <raw_root>/DVSGesture/ibmGestureTrain/<user_lighting>/<n>.npy
    <raw_root>/DVSGesture/ibmGestureTest/<user_lighting>/<n>.npy

Reading goes through ``tonic.datasets.DVSGesture`` (the sample order defines the
output file names ``sample_000000.pt`` ...), so results match the tonic split.
"""
from typing import List

import numpy as np

from .base import DatasetAdapter, Events, SampleRef, empty_events, register, to_pm1


def tonic_to_events(ev: np.ndarray, sensor_w: int, sensor_h: int) -> Events:
    """Tonic structured array (fields x/y/t/p, t in us) -> (x, y, t_ms, p)."""
    if len(ev) == 0:
        return empty_events()
    t_us = ev["t"].astype(np.float64)
    x = ev["x"].astype(np.float32)
    y = ev["y"].astype(np.float32)
    pol = to_pm1(ev["p"])
    mask = (x < sensor_w) & (y < sensor_h)
    x, y, t_us, pol = x[mask], y[mask], t_us[mask], pol[mask]
    if len(t_us) <= 1:
        return x, y, (t_us / 1000.0).astype(np.float32), pol
    order = np.argsort(t_us)
    x, y, t_us, pol = x[order], y[order], t_us[order], pol[order]
    t_ms = (t_us - t_us[0]) / 1000.0
    return x, y, t_ms.astype(np.float32), pol


@register
class DVSGesture(DatasetAdapter):
    name = "dvsgesture"
    description = "DVS128 Gesture: 1,341 recordings of 11 hand/arm gestures (1,077 train / 264 test)."
    num_classes = 11
    sensor = (128, 128)
    has_official_split = True

    def __init__(self, raw_root, cfg):
        super().__init__(raw_root, cfg)
        self._tonic = {}

    def _ds(self, split: str):
        if split not in self._tonic:
            import tonic
            self._tonic[split] = tonic.datasets.DVSGesture(
                save_to=self.raw_root, train=(split == "train"))
        return self._tonic[split]

    def list_samples(self) -> List[SampleRef]:
        refs = []
        for split in ("train", "test"):
            n = len(self._ds(split))
            for i in range(n):
                refs.append(SampleRef(split, f"sample_{i:06d}", -1,
                                      f"sample_{i:06d}", i))   # label comes from load()
        return refs

    def load(self, ref: SampleRef):
        ev, label = self._ds(ref.split)[ref.source]
        return tonic_to_events(ev, *self.sensor), int(label)

    def load_events(self, ref: SampleRef) -> Events:
        return self.load(ref)[0]
