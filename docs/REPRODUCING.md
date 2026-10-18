# Reproducing the paper

## 0. Setup

```bash
conda env create -f environment.yml && conda activate cegnn     # Python 3.8, torch 1.11 + CUDA 11.3, PyG 2.0.4
pip install -e .
export CEGNN_DATA_ROOT=/path/to/datasets                         # see docs/DATASETS.md
export CEGNN_RESULTS_ROOT=$PWD/results                           # optional, default ./results
make test                                                        # 17 CPU unit tests, no data needed
```

Versions are old on purpose: they are the ones the shipped checkpoints and numbers come from.

## 1. One command per dataset (main table)

```bash
bash scripts/run_pipeline.sh dvsgesture      # or: ncaltech101 | cifar10dvs | ncars | dvslip
```

= preprocess → train **baseline** → train **Rate-FiLM (natural rates)** → train **Rate-FiLM curriculum**
(warm-start from the previous model, then fine-tune with rate augmentation) → synthetic-rate evaluation
→ natural-rate-bucket evaluation. Stages are skipped when their output exists, so it is safe to re-run.
Training uses the schedule in `configs/<dataset>.yaml` (Adam, lr 1e-3, StepLR(30, 0.1), 80 epochs,
early-stopping patience 15, batch 16, seed 42). Wall time and GPU memory depend on your hardware.

Outputs: `$CEGNN_RESULTS_ROOT/<dataset>/{baseline,film_rate_natural,film_rate_curriculum}/best.ckpt`,
`main_compare/comparison.csv`, `main_natural_eval/natural_rate_per_bucket.csv`. Rebuild the table with
`make summary`.

The two evaluation protocols:

| | what changes | buckets / factors | aggregates |
|---|---|---|---|
| *natural-rate* (headline) | nothing — test recordings as recorded | 7 equal-count quantile buckets of `log_lambda_raw` | mean, worst bucket |
| *synthetic-rate* | timestamps rescaled by f, graph rebuilt | f ∈ {0.125, 0.25, 0.5, 1, 2, 4, 8} | mean, worst factor |

## 2. Where every paper result comes from

| Paper element | Command | Stored results |
|---|---|---|
| Main table (N-Caltech101) | `run_pipeline.sh ncaltech101`, `run_ablations.sh`, `run_ratta.sh ncaltech101 film_rate_curriculum`, `scripts/negative_results/*` | `results/ncaltech101/` |
| Cross-dataset table | `run_pipeline.sh <dataset>` for each dataset | `results/<dataset>/main_*` |
| Ablations: capacity / placement / curriculum source | `scripts/run_ablations.sh` | `results/ncaltech101/film_ablation_*` |
| RA-TTA rows | `scripts/run_ratta.sh` | `results/ncaltech101/rptta_on_*` |
| Rate/class confound analysis | `scripts/analysis/rate_label_leakage.py --config configs/<dataset>.yaml` | `results/<dataset>/rate_label_leakage/leakage.json` |
| What does FiLM compute? (γ/β, t-SNE, separability) | `scripts/analysis/visualize_film.py` | `figures/` (created on first run) |
| Teaser (rate histogram + Pareto) | `scripts/analysis/make_teaser.py` | `figures/fig_teaser.pdf` |
| Efficiency (FLOPs / event) | `scripts/eval/efficiency.py` | — |
| Is temporal structure used? (shuffle / reverse / constant time) | `scripts/analysis/temporal_reality_check.py` | `results/*/temporal_reality_check/` |
| Cross-architecture (AEGNN, EventMamba) | [`ports/README.md`](../ports/README.md) | `ports/*/results/` |

## 3. Evaluation noise

The backbone uses farthest-point sampling with a **random start point** (`torch_cluster.fps`), so every
evaluation pass is slightly stochastic. On a small test set the effect is visible: DVS128 Gesture has
264 test recordings, and re-evaluating the *same* shipped checkpoints gave bucket-mean accuracy in the
range of a few points between passes (baseline 67–70%, curriculum 83–87%; the ordering
baseline < Rate-FiLM < curriculum held in every pass). On N-Caltech101 the same baseline checkpoint scored
60.5 / 60.9 / 61.0 / 61.6 in four stored passes.

* `eval_natural_rates.py --repeats 5` averages several passes — use it when you compare against a
  number to within ~1–2 points.
* The shipped CSVs are single passes, exactly as used in the paper.
* Release check (CPU, this repo's code, shipped DVS128 Gesture checkpoints, `--repeats 5`): natural-rate mean
  **68.3 / 71.1 / 82.7** (baseline / Rate-FiLM / curriculum) vs. the paper's 68.6 / 71.2 / 82.5.

## 4. Model selection protocol

There is no separate validation split for any dataset. `best.ckpt` is the epoch with the highest accuracy
on the test split, and early stopping (patience 15) watches the same number; this is identical for the
baseline, the FiLM models and the ablations, so comparisons between them are like for like. Absolute
accuracies are therefore slightly optimistic. If you need clean numbers, carve a validation set out of
the train split in `cegnn/data/loaders.py` and monitor that instead.

## 5. Pre-trained checkpoints

`checkpoints/<dataset>/{baseline,film_rate_natural,film_rate_curriculum}.ckpt` (plus the N-Caltech101
ablation models) — ~2.5 MB each (0.16–0.21 M parameters), so they live in git directly:

```python
from cegnn.config import load_config
from cegnn.utils import load_checkpoint               # auto-detects baseline / FiLM / FiLM-global-only
cfg = load_config("configs/dvsgesture.yaml", make_results_dir=False)
model = load_checkpoint("checkpoints/dvsgesture/film_rate_curriculum.ckpt", cfg).eval()
```
