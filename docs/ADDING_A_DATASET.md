# Adding a dataset

You only write one file. Example for a hypothetical `mydvs` dataset stored as `<raw>/{train,test}/<class>/*.npy`
(structured arrays with fields `t` [µs], `x`, `y`, `p`):

```python
# cegnn/data/readers/mydvs.py
import os
import numpy as np
from .base import DatasetAdapter, SampleRef, register, to_pm1


@register
class MyDVS(DatasetAdapter):
    name = "mydvs"                     # used as --dataset mydvs
    description = "My event dataset."
    num_classes = 7
    sensor = (240, 180)                # (W, H)
    has_official_split = True

    def list_samples(self):
        classes = sorted(os.listdir(os.path.join(self.raw_root, "train")))
        refs = []
        for split in ("train", "test"):
            for label, cls in enumerate(classes):
                d = os.path.join(self.raw_root, split, cls)
                for fn in sorted(os.listdir(d)):
                    refs.append(SampleRef(split, f"{cls}_{fn[:-4]}", label, fn, os.path.join(d, fn)))
        return refs

    def load_events(self, ref):
        a = np.load(ref.source)
        order = np.argsort(a["t"])
        t_ms = (a["t"][order] - a["t"][order][0]) / 1000.0           # ms, origin at 0, NOT normalised
        return (a["x"][order].astype(np.float32), a["y"][order].astype(np.float32),
                t_ms.astype(np.float32), to_pm1(a["p"][order]))
```

Then:

1. Import it in `cegnn/data/readers/__init__.py` (`from . import mydvs`).
2. Copy a config: `cp configs/dvsgesture.yaml configs/mydvs.yaml`; set `num_classes`, `spatial_dims`,
   the paths, and the SA radii (`sa*_radius`; we used 20 / 40 / 80 for the 240×180 ATIS sensor and 10 / 20 / 40 for the 128×128 and 120×100 sensors).
3. `python scripts/data/prepare_dataset.py --dataset mydvs --config configs/mydvs.yaml --limit 20`
4. `bash scripts/run_pipeline.sh mydvs`

A few conventions: return time in milliseconds starting at 0 and do not normalise it (the method depends on
the natural rate surviving), polarity in {−1, +1} (`to_pm1` maps {0, 1}), and for datasets without an official split use
`cegnn.data.splits.per_class_split` so the split is deterministic. If the label is only known after
reading the file (like DVS128 Gesture via `tonic`), override `load()` to return `(events, label)`.
