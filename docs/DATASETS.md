# Datasets

CEGNN is evaluated on five event-camera benchmarks. All are processed by **one** pipeline
(`cegnn/data/`) into an identical graph format — see [PREPROCESSING.md](PREPROCESSING.md).

Set one environment variable and keep every dataset below it:

```bash
export CEGNN_DATA_ROOT=/path/to/datasets      # raw data in, processed graphs out
```

| Dataset | Task | Sensor | Classes | Split | Raw folder (under `$CEGNN_DATA_ROOT`) | Processed folder | Get it |
|---|---|---|---:|---|---|---|---|
| **N-Caltech101** | object recognition | ATIS 240×180 | 101 | stratified 80/20 per class, seed 42 | `N-Caltech101/Caltech101/<class>/*.bin` | `ncaltech101_v2_processed/` | manual (Orchard et al., 2015) |
| **N-Cars** | car vs. background | ATIS 120×100 | 2 | official | `ncars/n-cars_{train,test}/{cars,background}/*.dat` | `ncars_v2_processed/` | manual (Sironi et al., 2018; Prophesee) |
| **DVS128 Gesture** | gesture recognition | DVS128 128×128 | 11 | official (1,077 / 264) | `dvsgesture_raw/DVSGesture/ibmGesture{Train,Test}/` | `dvsgesture_v2_processed/` | `scripts/data/download_datasets.py` |
| **CIFAR10-DVS** | image-as-events | DVS128 128×128 | 10 | stratified 80/20 per class, seed 42 | `cifar10dvs_raw/CIFAR10DVS/<class>/*.aedat` | `cifar10dvs_v2_processed/` | `scripts/data/download_datasets.py` |
| **DVS-Lip** | lip reading (100 words) | DVS 128×128 | 100 | official | `dvslip_raw/DVS-Lip/{train,test}/<word>/*.npy` | `dvslip_v2_processed/` | manual (Tan et al., CVPR 2022) |

> The paper's main comparison covers DVS128 Gesture, N-Caltech101, CIFAR10-DVS and N-Cars;
> DVS-Lip was added afterwards (see the results table in the README).

## Downloading

```bash
# the two figshare-hosted sets (works around figshare's browser-only WAF by following the
# signed-S3 redirect with curl; verifies md5 for DVS128 Gesture)
python scripts/data/download_datasets.py --datasets dvsgesture cifar10dvs --data_root $CEGNN_DATA_ROOT
```

N-Caltech101, N-Cars and DVS-Lip are distributed behind registration pages / drive links and must be
downloaded by hand, then unpacked to the layout in the table. Quick sanity checks:

* N-Caltech101: 101 class folders, 8,709 `.bin` files in total (BACKGROUND_Google is kept as a class).
* N-Cars: 15,422 train / 8,607 test `.dat` recordings.
* DVS-Lip: 100 word folders under `train/`.

## Processing

```bash
python scripts/data/prepare_dataset.py --dataset ncaltech101 --config configs/ncaltech101.yaml --workers 16
python scripts/data/prepare_dataset.py --dataset dvslip     --config configs/dvslip.yaml --limit 20   # smoke test
```

The job is **resumable** (existing `.pt` files are skipped) and tolerant (a corrupt raw file is
reported, not fatal). `--limit N` processes only the first N samples per split.

## Citations

* N-Caltech101 — G. Orchard et al., *Converting static image datasets to spiking neuromorphic datasets using saccades*, Front. Neurosci. 2015.
* N-Cars — A. Sironi et al., *HATS: Histograms of Averaged Time Surfaces*, CVPR 2018.
* DVS128 Gesture — A. Amir et al., *A low power, fully event-based gesture recognition system*, CVPR 2017.
* CIFAR10-DVS — H. Li et al., *CIFAR10-DVS: an event-stream dataset for object classification*, Front. Neurosci. 2017.
* DVS-Lip — Y. Tan et al., *Multi-grained spatio-temporal features perceived network for event-based lip-reading*, CVPR 2022.
