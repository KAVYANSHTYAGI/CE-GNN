"""Unit tests for the data layer: config, rate statistics, binary readers, sample schema."""
import os
import struct

import numpy as np
import pytest
import torch

from cegnn.config import expand_vars
from cegnn.data.rate import compute_local_rate, natural_rate_stats
from cegnn.data.readers import available_datasets
from cegnn.data.readers.cifar10dvs import read_aedat2
from cegnn.data.readers.ncaltech101 import read_bin
from cegnn.data.sample import events_to_sample
from cegnn.data.splits import per_class_split

PARAMS = dict(t_unit=1.0, r=6.0, d_max=32, subsample_K=25, n_max=10000)


def _synthetic_events(n=3000, w=64, h=48, duration_ms=300.0, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, w, n).astype(np.float32)
    y = rng.integers(0, h, n).astype(np.float32)
    t = np.sort(rng.uniform(0, duration_ms, n)).astype(np.float32)
    p = rng.choice([-1.0, 1.0], n).astype(np.float32)
    return x, y, t, p


# ---------------------------------------------------------------- config
def test_expand_vars_default_and_env(monkeypatch):
    monkeypatch.delenv("CEGNN_TEST_VAR", raising=False)
    assert expand_vars("${CEGNN_TEST_VAR:-fallback}/a") == "fallback/a"
    monkeypatch.setenv("CEGNN_TEST_VAR", "/data")
    assert expand_vars("${CEGNN_TEST_VAR:-fallback}/a") == "/data/a"


def test_expand_vars_missing_without_default_raises(monkeypatch):
    monkeypatch.delenv("CEGNN_TEST_MISSING", raising=False)
    with pytest.raises(KeyError):
        expand_vars("${CEGNN_TEST_MISSING}/a")


# ---------------------------------------------------------------- registry
def test_all_paper_datasets_registered():
    assert set(available_datasets()) == {"ncaltech101", "ncars", "dvsgesture", "cifar10dvs", "dvslip"}


# ---------------------------------------------------------------- rate
def test_natural_rate_stats_known_values():
    t_ms = np.linspace(0, 1000.0, 5001)            # 5001 events over 1 s  => ~5001 ev/s
    s = natural_rate_stats(t_ms)
    assert s["n_raw"] == 5001
    assert s["lambda_raw"] == pytest.approx(5001.0, rel=1e-3)
    assert s["log_lambda_raw"] == pytest.approx(np.log(5001.0), rel=1e-3)


def test_local_rate_constant_stream_is_constant():
    t = np.arange(1000) * 1e-3                       # 1 kHz
    lam = compute_local_rate(t, k=20)
    assert np.allclose(lam[25:], 1000.0, rtol=1e-3)


# ---------------------------------------------------------------- sample schema
def test_events_to_sample_schema():
    x, y, t, p = _synthetic_events()
    d = events_to_sample(x, y, t, p, label=3, file_id="s0", **PARAMS)
    n = d.pos.size(0)
    assert d.x.shape == (n, 1) and d.pos.shape == (n, 3)
    assert d.lambda_local.shape == (n, 1)
    assert d.edge_index.size(0) == 2 and d.edge_attr.shape == (d.edge_index.size(1), 3)
    assert int(d.y) == 3 and d.file_id == "s0"
    # natural duration is PRESERVED (not normalised to a fixed window)
    assert float(d.duration_ms) == pytest.approx(300.0, abs=1.0)
    assert float(d.log_lambda_raw) == pytest.approx(np.log(3000 / 0.3), rel=1e-2)


def test_events_to_sample_preserves_rate_differences():
    """Same spatial content at 2x the speed => log-rate differs by ln 2 (the whole point)."""
    x, y, t, p = _synthetic_events()
    slow = events_to_sample(x, y, t, p, 0, "a", **PARAMS)
    fast = events_to_sample(x, y, t / 2.0, p, 0, "b", **PARAMS)
    assert float(fast.log_lambda_raw - slow.log_lambda_raw) == pytest.approx(np.log(2), abs=1e-2)


def test_events_to_sample_degenerate_returns_none():
    assert events_to_sample(*(np.zeros(1, np.float32),) * 4, label=0, file_id="x", **PARAMS) is None


# ---------------------------------------------------------------- readers (round trip)
def test_ncaltech_bin_roundtrip(tmp_path):
    xs, ys = [10, 20, 30], [5, 6, 7]
    ts, ps = [1000, 2500, 4000], [1, 0, 1]       # microseconds, polarity bit
    buf = bytearray()
    for x, y, t, p in zip(xs, ys, ts, ps):
        buf += bytes([x, y, (p << 7) | ((t >> 16) & 0x7F), (t >> 8) & 0xFF, t & 0xFF])
    f = tmp_path / "a.bin"
    f.write_bytes(bytes(buf))
    x, y, t_ms, pol = read_bin(str(f), 240, 180)
    assert x.tolist() == xs and y.tolist() == ys
    assert t_ms.tolist() == pytest.approx([0.0, 1.5, 3.0])       # origin at first event, in ms
    assert pol.tolist() == [1.0, -1.0, 1.0]


def test_aedat2_roundtrip(tmp_path):
    # DVS128 address: x = (a & 0xFE) >> 1, y = (a & 0x7F00) >> 8, p = a & 1
    def addr(x, y, p):
        return (y << 8) | (x << 1) | p
    recs = [(addr(5, 9, 1), 100), (addr(7, 2, 0), 1100), (addr(1, 1, 1), 3100)]
    body = b"".join(struct.pack(">II", a, t) for a, t in recs)
    f = tmp_path / "a.aedat"
    f.write_bytes(b"#!AER-DAT2.0\r\n# comment\r\n" + body)
    x, y, t_ms, pol = read_aedat2(str(f))
    assert x.tolist() == [5, 7, 1] and y.tolist() == [9, 2, 1]
    assert t_ms.tolist() == pytest.approx([0.0, 1.0, 3.0])
    assert pol.tolist() == [1.0, -1.0, 1.0]


# ---------------------------------------------------------------- splits
def test_per_class_split_is_deterministic_and_disjoint():
    files = {c: [f"{c}_{i}" for i in range(20)] for c in ("a", "b", "c")}
    s1 = per_class_split(files, seed=42, train_frac=0.8)
    s2 = per_class_split(files, seed=42, train_frac=0.8)
    assert s1 == s2
    for tr, te in s1.values():
        assert len(tr) == 16 and len(te) == 4 and not set(tr) & set(te)


def test_per_class_split_keeps_rare_classes_in_both_splits():
    s = per_class_split({"rare": ["x", "y"]}, seed=0, train_frac=0.99, keep_both_nonempty=True)
    tr, te = s["rare"]
    assert len(tr) == 1 and len(te) == 1
