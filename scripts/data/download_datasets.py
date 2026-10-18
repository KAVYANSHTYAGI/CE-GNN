"""Download DVS128 Gesture and CIFAR10-DVS (thin wrapper over ``cegnn.data.download``).

    python scripts/data/download_datasets.py --datasets dvsgesture cifar10dvs --data_root $CEGNN_DATA_ROOT

N-Caltech101, N-Cars and DVS-Lip require manual download - see docs/DATASETS.md.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cegnn.data.download import main  # noqa: E402

if __name__ == "__main__":
    main()
