"""CIFAR10-DVS (Li et al., 2017) - 10 classes, DVS128 128x128, no official split.

Raw layout (legacy jAER AEDAT-2.0 files, one folder per class)::

    <raw_root>/CIFAR10DVS/<airplane|automobile|...|truck>/*.aedat

We parse the AEDAT-2.0 container directly (no ``tonic``/``dv`` dependency).
Split: deterministic stratified 80/20 per class.
"""
import os
from typing import List

import numpy as np

from .base import DatasetAdapter, Events, SampleRef, empty_events, register
from ..splits import per_class_split

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]


def read_aedat2(path: str) -> Events:
    """Read an AEDAT-2.0 file -> (x, y, t_ms, p).

    ASCII ``#``-prefixed header, then 8-byte big-endian records
    ``(address: uint32, timestamp_us: uint32)``. DVS128 address decoding:
    ``x = (addr & 0xFE) >> 1``, ``y = (addr & 0x7F00) >> 8``, ``p = addr & 1``.
    """
    with open(path, "rb") as f:
        p = 0
        line = f.readline()
        while line and line[0] == ord("#"):
            p += len(line)
            line = f.readline()
        f.seek(p)
        raw = np.frombuffer(f.read(), dtype=">u4")
    raw = raw[: (len(raw) // 2) * 2]
    if len(raw) == 0:
        return empty_events()
    addr = raw[0::2]
    t_us = raw[1::2].astype(np.int64)
    x = ((addr & 0xFE) >> 1).astype(np.float32)
    y = ((addr & 0x7F00) >> 8).astype(np.float32)
    pol = (addr & 0x1).astype(np.float32) * 2 - 1
    order = np.argsort(t_us)
    x, y, t_us, pol = x[order], y[order], t_us[order], pol[order]
    t_ms = (t_us - t_us[0]).astype(np.float64) / 1000.0
    return x, y, t_ms.astype(np.float32), pol


@register
class CIFAR10DVS(DatasetAdapter):
    name = "cifar10dvs"
    description = "CIFAR10-DVS: 10,000 DVS recordings of CIFAR-10 images moved in front of a DVS128."
    num_classes = 10
    sensor = (128, 128)
    has_official_split = False

    def list_samples(self) -> List[SampleRef]:
        base = os.path.join(self.raw_root, "CIFAR10DVS")
        files = {}
        for cls in CLASSES:
            d = os.path.join(base, cls)
            if os.path.isdir(d):
                files[cls] = sorted(fn for fn in os.listdir(d) if fn.endswith(".aedat"))
        d = self.cfg["dataset"]
        split = per_class_split(files, int(d["split_seed"]), float(d["split_train_frac"]))
        refs = []
        for cls, (train_files, test_files) in split.items():
            for split_name, fs in (("train", train_files), ("test", test_files)):
                for fn in fs:
                    refs.append(SampleRef(split_name, f"{cls}_{os.path.splitext(fn)[0]}",
                                          CLASSES.index(cls), fn,
                                          os.path.join(base, cls, fn)))
        return refs

    def load_events(self, ref: SampleRef) -> Events:
        return read_aedat2(ref.source)
