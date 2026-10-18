"""Preprocess one dataset into CEGNN graphs (thin wrapper over ``cegnn.data.preprocess``).

    python scripts/data/prepare_dataset.py --dataset ncaltech101 --config configs/ncaltech101.yaml --workers 16
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cegnn.data.preprocess import main  # noqa: E402

if __name__ == "__main__":
    main()
