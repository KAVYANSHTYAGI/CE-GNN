#!/usr/bin/env bash
# End-to-end reproduction of the paper's main table for ONE dataset:
#   preprocess -> baseline -> Rate-FiLM (natural) -> Rate-FiLM curriculum
#   -> synthetic-rate evaluation -> natural-rate-bucket evaluation.
#
#   bash scripts/run_pipeline.sh ncaltech101            # or: ncars dvsgesture cifar10dvs dvslip
#
# Environment:
#   CEGNN_DATA_ROOT     folder holding the raw datasets (and receiving the processed graphs)
#   CEGNN_RESULTS_ROOT  where checkpoints / CSVs are written   (default: ./results)
#   WORKERS             preprocessing workers                   (default: 16)
#   PYTHON              interpreter                             (default: python)
set -euo pipefail
cd "$(dirname "$0")/.."

DS="${1:?usage: run_pipeline.sh <ncaltech101|ncars|dvsgesture|cifar10dvs|dvslip>}"
PYTHON="${PYTHON:-python}"
WORKERS="${WORKERS:-16}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export CEGNN_DATA_ROOT="${CEGNN_DATA_ROOT:-$PWD/data}"
export CEGNN_RESULTS_ROOT="${CEGNN_RESULTS_ROOT:-$PWD/results}"

CFG="configs/${DS}.yaml"
RES="$CEGNN_RESULTS_ROOT/${DS}"
RATES="0.125 0.25 0.5 1.0 2.0 4.0 8.0"

stage() {  # stage <sentinel-file> <description> <command...>   (skips finished stages)
  local sentinel="$1" desc="$2"; shift 2
  if [[ -e "$sentinel" ]]; then echo ">>> SKIP  $desc"; return 0; fi
  echo; echo ">>> RUN   $desc   ($(date +%T))"; "$@"
}

# preprocessing is resumable (existing .pt files are skipped), so it is always safe to re-run
echo ">>> RUN   [$DS] preprocess"
"$PYTHON" -u scripts/data/prepare_dataset.py --dataset "$DS" --config "$CFG" --workers "$WORKERS"

stage "$RES/baseline/best.ckpt" "[$DS] baseline (PointNet++)" \
  "$PYTHON" -u scripts/train/train_baseline.py --config "$CFG" --name baseline

stage "$RES/film_rate_natural/best.ckpt" "[$DS] Rate-FiLM, natural rates" \
  "$PYTHON" -u scripts/train/train_film.py --config "$CFG" --name film_rate_natural --no_rate_aug

stage "$RES/film_rate_curriculum/best.ckpt" "[$DS] Rate-FiLM curriculum (headline)" \
  "$PYTHON" -u scripts/train/train_film.py --config "$CFG" --name film_rate_curriculum \
      --init_ckpt "$RES/film_rate_natural/best.ckpt"

MODELS=("baseline:$RES/baseline/best.ckpt:fixed"
        "film_rate_natural:$RES/film_rate_natural/best.ckpt:fixed"
        "film_rate_curriculum:$RES/film_rate_curriculum/best.ckpt:fixed")

stage "$RES/main_compare/comparison.csv" "[$DS] synthetic-rate evaluation" \
  "$PYTHON" -u scripts/eval/eval_synthetic_rates.py --config "$CFG" --name main_compare \
      --rates $RATES --models "${MODELS[@]}"

stage "$RES/main_natural_eval/natural_rate_per_bucket.csv" "[$DS] natural-rate-bucket evaluation" \
  "$PYTHON" -u scripts/eval/eval_natural_rates.py --config "$CFG" --name main_natural_eval \
      --n_buckets 7 --models "${MODELS[@]}"

echo; echo "Done. Tables: $RES/main_compare/comparison.csv  $RES/main_natural_eval/natural_rate_per_bucket.csv"
