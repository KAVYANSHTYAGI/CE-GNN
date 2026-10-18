"""Integration smoke tests on REAL processed data + shipped checkpoints (CPU).

Skipped automatically unless CEGNN_DATA_ROOT points at a folder containing
``dvsgesture_v2_processed`` (see docs/DATASETS.md) and the checkpoints are present.
"""
import os

import pytest
import torch
from torch_geometric.loader import DataLoader

from cegnn.config import load_config
from cegnn.data.dataset import EventGraphDataset, RateAugmentedDataset
from cegnn.models import EventClassifier, FiLMRateClassifier
from cegnn.utils import load_checkpoint

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.environ.get("CEGNN_DATA_ROOT", "")
CKPT = os.path.join(ROOT, "checkpoints", "dvsgesture")
pytestmark = pytest.mark.skipif(
    not os.path.isdir(os.path.join(DATA, "dvsgesture_v2_processed", "train"))
    or not os.path.isdir(CKPT),
    reason="needs CEGNN_DATA_ROOT with dvsgesture_v2_processed and checkpoints/dvsgesture",
)


@pytest.fixture(scope="module")
def cfg():
    return load_config(os.path.join(ROOT, "configs", "dvsgesture.yaml"), make_results_dir=False)


def test_rate_augmented_batch_and_film_train_step(cfg):
    ds = RateAugmentedDataset(train_rates=[0.5, 1.0, 2.0], processed_dir=cfg["paths"]["processed"],
                              split="train", cfg=cfg, augment=True, graph_mode="fixed", lambda_train=None)
    batch = next(iter(DataLoader(ds, batch_size=4, shuffle=True)))
    assert batch.log_lambda_raw.shape == (4,) or batch.log_lambda_raw.numel() == 4
    model = FiLMRateClassifier(cfg).train()
    loss = torch.nn.functional.cross_entropy(model(batch), batch.y.long().view(-1))
    loss.backward()
    film_grad = sum(p.grad.abs().sum().item() for k, p in model.named_parameters()
                    if "film" in k and p.grad is not None)
    assert torch.isfinite(loss) and film_grad > 0           # FiLM actually receives gradient


def test_curriculum_warm_start_from_baseline_checkpoint(cfg):
    """The curriculum's stage-2 init: load a baseline ckpt into FiLMRateClassifier (strict=False)."""
    ck = torch.load(os.path.join(CKPT, "baseline.ckpt"), map_location="cpu")
    model = FiLMRateClassifier(cfg)
    missing, unexpected = model.load_state_dict(ck["state_dict"], strict=False)
    assert not unexpected and all("film" in k for k in missing)


def test_load_checkpoint_autodetects_film_and_runs(cfg):
    model = load_checkpoint(os.path.join(CKPT, "film_rate_curriculum.ckpt"), cfg).eval()
    assert isinstance(model, FiLMRateClassifier)
    ds = EventGraphDataset(cfg["paths"]["processed"], "test", cfg)
    batch = next(iter(DataLoader(ds, batch_size=8)))
    with torch.no_grad():
        logits = model(batch)
    assert logits.shape == (8, cfg["dataset"]["num_classes"]) and torch.isfinite(logits).all()
