"""N-Caltech101 (Orchard et al., 2015) - 101 classes, ATIS 240x180, no official split.

Raw layout::

    <raw_root>/<class_name>/*.bin          # e.g. .../N-Caltech101/Caltech101/accordion/image_0001.bin

Binary format: 5 bytes / event
    byte0 = x, byte1 = y, byte2 bit7 = polarity, byte2[6:0]+byte3+byte4 = t (us).
Split: deterministic stratified 80/20 per class (seed from the config).
"""
import glob
import os
from typing import List

import numpy as np

from .base import DatasetAdapter, Events, SampleRef, empty_events, register
from ..splits import per_class_split


def read_bin(path: str, sensor_w: int, sensor_h: int) -> Events:
    """Parse one ``.bin`` file -> (x, y, t_ms, p). Time is in ms, origin at 0, NOT normalised."""
    with open(path, "rb") as f:
        raw = f.read()
    arr = np.frombuffer(raw, dtype=np.uint8)
    n = len(arr) // 5
    if n == 0:
        return empty_events()

    arr = arr[:n * 5].reshape(n, 5)
    x = arr[:, 0].astype(np.float32)
    y = arr[:, 1].astype(np.float32)
    b2 = arr[:, 2].astype(np.uint32)
    b3 = arr[:, 3].astype(np.uint32)
    b4 = arr[:, 4].astype(np.uint32)
    pol = ((b2 >> 7) & 1).astype(np.float32) * 2.0 - 1.0
    t_us = (((b2 & 0x7F) << 16) | (b3 << 8) | b4).astype(np.float64)

    mask = (x < sensor_w) & (y < sensor_h)
    x, y, t_us, pol = x[mask], y[mask], t_us[mask], pol[mask]

    if len(t_us) > 1:
        order = np.argsort(t_us)
        x, y, t_us, pol = x[order], y[order], t_us[order], pol[order]
        t_ms = (t_us - t_us[0]) / 1000.0
    else:
        t_ms = t_us / 1000.0
    return x, y, t_ms.astype(np.float32), pol


@register
class NCaltech101(DatasetAdapter):
    name = "ncaltech101"
    description = "N-Caltech101: 8,709 saccade recordings of 101 object classes (ATIS)."
    num_classes = 101
    sensor = (240, 180)
    has_official_split = False

    def list_samples(self) -> List[SampleRef]:
        d = self.cfg["dataset"]
        class_dirs = sorted(c for c in os.listdir(self.raw_root)
                            if os.path.isdir(os.path.join(self.raw_root, c)))
        files = {c: sorted(glob.glob(os.path.join(self.raw_root, c, "*.bin")))
                 for c in class_dirs}
        split = per_class_split(files, int(d["split_seed"]), float(d["split_train_frac"]),
                                keep_both_nonempty=True)
        label = {c: i for i, c in enumerate(class_dirs)}
        refs = []
        for cls, (train_files, test_files) in split.items():
            for split_name, fs in (("train", train_files), ("test", test_files)):
                for f in fs:
                    base = os.path.splitext(os.path.basename(f))[0]
                    refs.append(SampleRef(split_name, f"{cls}_{base}", label[cls],
                                          os.path.basename(f), f))
        return refs

    def load_events(self, ref: SampleRef) -> Events:
        w, h = self.sensor
        return read_bin(ref.source, w, h)
