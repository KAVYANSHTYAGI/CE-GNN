"""Deterministic train/test splitting helpers.

Datasets without an official split (N-Caltech101, CIFAR10-DVS) are split 80/20
*per class* with a fixed seed, so rare classes appear in both splits and the
split is identical on every machine.
"""
from typing import Dict, List, Sequence, Tuple

import numpy as np


def per_class_split(
    files_by_class: Dict[str, Sequence[str]],
    seed: int,
    train_frac: float,
    keep_both_nonempty: bool = False,
) -> Dict[str, Tuple[List[str], List[str]]]:
    """Split each class's (already sorted) file list into (train, test).

    The RNG is consumed class-by-class in the iteration order of
    ``files_by_class``, so pass classes in a stable (sorted) order.

    Args:
        keep_both_nonempty: if True and a class has >= 2 files, force at least one
            file into each split (used for N-Caltech101, which has classes with
            only 31 samples).
    """
    rng = np.random.default_rng(seed)
    out = {}
    for cls, files in files_by_class.items():
        files = list(files)
        if not files:
            continue
        perm = rng.permutation(len(files))
        n_train = int(round(len(files) * train_frac))
        if keep_both_nonempty and len(files) >= 2:
            n_train = max(1, min(len(files) - 1, n_train))
        out[cls] = ([files[i] for i in perm[:n_train]],
                    [files[i] for i in perm[n_train:]])
    return out
