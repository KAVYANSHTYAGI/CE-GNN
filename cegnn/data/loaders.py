"""DataLoader factories and the dataset reference rate."""
from pathlib import Path
from typing import Optional

from torch_geometric.loader import DataLoader as PyGDataLoader

from .dataset import EventGraphDataset
from .rate import dataset_reference_rate


def make_loader(
    cfg: dict,
    split: str,
    *,
    augment: bool = False,
    rebuild: bool = False,
    rate_factor: float = 1.0,
    graph_mode: str = "fixed",
    lambda_train: Optional[float] = None,
    batch_size: Optional[int] = None,
    shuffle: Optional[bool] = None,
) -> PyGDataLoader:
    """Build a PyG dataloader for any experimental condition."""
    ds = EventGraphDataset(
        processed_dir=cfg["paths"]["processed"],
        split=split,
        cfg=cfg,
        augment=augment,
        rebuild=rebuild,
        rate_factor=rate_factor,
        graph_mode=graph_mode,
        lambda_train=lambda_train,
    )
    bs = batch_size or cfg["train"]["batch_size"]
    sh = shuffle if shuffle is not None else (split == "train")
    return PyGDataLoader(ds, batch_size=bs, shuffle=sh,
                         num_workers=cfg["train"]["num_workers"], pin_memory=True)


def get_lambda_train(cfg: dict, cache: bool = True) -> float:
    """Median per-sample event rate on the train split (cached in ``<results>/lambda_train.txt``)."""
    cache_file = Path(cfg["paths"]["results"]) / "lambda_train.txt"
    if cache and cache_file.exists():
        return float(cache_file.read_text().strip())

    print("[get_lambda_train] computing reference rate from train split...")
    loader = make_loader(cfg, "train", augment=False, batch_size=32, shuffle=False)
    lam = dataset_reference_rate(loader, max_batches=50)
    if cache:
        cache_file.write_text(str(lam))
    print(f"[get_lambda_train] lambda_train = {lam:.4f} events/time-unit")
    return lam
