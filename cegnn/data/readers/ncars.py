"""N-Cars (Sironi et al., 2018) - car vs. background, ATIS 120x100, official split.

Raw layout (as distributed by Prophesee)::

    <raw_root>/n-cars_train/{cars,background}/*.dat
    <raw_root>/n-cars_test/{cars,background}/*.dat

Requires the ``expelliarmus`` package for the Prophesee ``.dat`` container.
"""
import os
from typing import List

import numpy as np

from .base import DatasetAdapter, Events, SampleRef, empty_events, register, to_pm1


def read_dat(path: str) -> Events:
    """Read a Prophesee ``.dat`` file -> (x, y, t_ms, p)."""
    from expelliarmus import Wizard

    arr = Wizard(encoding="dat").read(path)
    if arr is None or len(arr) == 0:
        return empty_events()
    t_us = arr["t"].astype(np.float64)
    x = arr["x"].astype(np.float32)
    y = arr["y"].astype(np.float32)
    pol = to_pm1(arr["p"])
    order = np.argsort(t_us)
    x, y, t_us, pol = x[order], y[order], t_us[order], pol[order]
    t_ms = (t_us - t_us[0]) / 1000.0
    return x, y, t_ms.astype(np.float32), pol


@register
class NCars(DatasetAdapter):
    name = "ncars"
    description = "N-Cars: 24,029 short recordings, car vs. background (ATIS)."
    num_classes = 2
    sensor = (120, 100)
    has_official_split = True

    def list_samples(self) -> List[SampleRef]:
        refs = []
        for split, sub in (("train", "n-cars_train"), ("test", "n-cars_test")):
            for cls_name, label in (("cars", 1), ("background", 0)):
                cls_root = os.path.join(self.raw_root, sub, cls_name)
                if not os.path.isdir(cls_root):
                    continue
                for fn in sorted(os.listdir(cls_root)):
                    if fn.endswith(".dat"):
                        base = os.path.splitext(fn)[0]
                        refs.append(SampleRef(split, f"{cls_name}_{base}", label, fn,
                                              os.path.join(cls_root, fn)))
        return refs

    def load_events(self, ref: SampleRef) -> Events:
        return read_dat(ref.source)
