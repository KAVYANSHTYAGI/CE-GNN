"""Model tests: the CEGNN conditioning layers behave as the paper describes."""
import os

import pytest
import torch

from cegnn.config import load_config
from cegnn.data.dataset import RateResample
from cegnn.models import EventClassifier, FiLM, FiLMGlobalOnlyClassifier, FiLMRateClassifier

CFG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "ncaltech101.yaml")


@pytest.fixture(scope="module")
def cfg():
    os.environ.setdefault("CEGNN_DATA_ROOT", "/nonexistent")
    return load_config(CFG_PATH, make_results_dir=False)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def test_film_is_identity_at_init():
    film = FiLM(feat_dim=32)
    x, cond = torch.randn(10, 32), torch.randn(10, 1)
    assert torch.allclose(film(x, cond), x)          # gamma = beta = 0  =>  y = x


def test_film_parameter_budget_matches_paper(cfg):
    base, film = EventClassifier(cfg), FiLMRateClassifier(cfg)
    n_film = sum(p.numel() for k, p in film.named_parameters() if "film" in k)
    assert n_params(base) == 163_429                  # "baseline budget"
    assert n_film == 48_832                           # "four FiLM layers add 48,832 parameters"
    assert n_params(film) == 212_261                  # "212,261 total"


def test_film_global_only_is_a_strict_subset(cfg):
    assert n_params(FiLMGlobalOnlyClassifier(cfg)) < n_params(FiLMRateClassifier(cfg))


def test_film_classifier_loads_baseline_weights(cfg):
    """Warm-start (the curriculum): a baseline state dict must load with only FiLM keys missing."""
    base, film = EventClassifier(cfg), FiLMRateClassifier(cfg)
    missing, unexpected = film.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected
    assert missing and all("film" in k for k in missing)


def test_rate_resample_updates_rate_metadata():
    from torch_geometric.data import Data
    d = Data(x=torch.ones(5, 1), pos=torch.rand(5, 3) * 100, y=torch.tensor([1]),
             log_lambda_raw=torch.tensor([11.0]), duration_ms=torch.tensor([300.0]))
    out = RateResample(2.0)(d)
    assert torch.allclose(out.pos[:, 2], d.pos[:, 2] / 2.0)         # time compressed 2x
    assert float(out.log_lambda_raw) == pytest.approx(11.0 + 0.6931, abs=1e-3)
    assert float(out.duration_ms) == pytest.approx(150.0)
