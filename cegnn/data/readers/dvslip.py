"""DVS-Lip (Tan et al., 2022) - 100-word lip reading, DVS346 128x128 crops, official split.

Raw layout::

    <raw_root>/DVS-Lip/{train,test}/<word>/<idx>.npy

Each ``.npy`` is a structured array with fields ``t`` (us), ``x``, ``y``, ``p`` in {0, 1}.
Class ids are the alphabetical order of the 100 word folders in ``train/``.
"""
import os
from typing import List

import numpy as np

from .base import DatasetAdapter, Events, SampleRef, empty_events, register, to_pm1


def read_npy(path: str) -> Events:
    a = np.load(path, allow_pickle=True)
    if len(a) == 0:
        return empty_events()
    t_us = a["t"].astype(np.int64)
    x = a["x"].astype(np.float32)
    y = a["y"].astype(np.float32)
    pol = to_pm1(a["p"])
    order = np.argsort(t_us)
    x, y, t_us, pol = x[order], y[order], t_us[order], pol[order]
    t_ms = (t_us - t_us[0]).astype(np.float64) / 1000.0
    return x, y, t_ms.astype(np.float32), pol


@register
class DVSLip(DatasetAdapter):
    name = "dvslip"
    description = "DVS-Lip: ~20k event recordings of 100 spoken words."
    num_classes = 100
    sensor = (128, 128)
    has_official_split = True

    def list_samples(self) -> List[SampleRef]:
        base = os.path.join(self.raw_root, "DVS-Lip")
        words = sorted(os.listdir(os.path.join(base, "train")))
        word2label = {w: i for i, w in enumerate(words)}
        refs = []
        for split in ("train", "test"):
            split_root = os.path.join(base, split)
            for w in sorted(os.listdir(split_root)):
                if w not in word2label:
                    continue
                wdir = os.path.join(split_root, w)
                for fn in sorted(os.listdir(wdir)):
                    if fn.endswith(".npy"):
                        refs.append(SampleRef(split, f"{w}_{os.path.splitext(fn)[0]}",
                                              word2label[w], fn, os.path.join(wdir, fn)))
        return refs

    def load_events(self, ref: SampleRef) -> Events:
        return read_npy(ref.source)
