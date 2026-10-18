#!/usr/bin/env bash
# N-Caltech101 ablations of the paper (Tab. "Consolidated ablations"):
#   capacity control  : baseline with a wider head (+44K params)
#   placement control : FiLM only after the global pool
#   curriculum source : FiLM from scratch with mild / default rate augmentation
# Assumes `scripts/run_pipeline.sh ncaltech101` has produced baseline + film_rate_natural.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"; export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export CEGNN_DATA_ROOT="${CEGNN_DATA_ROOT:-$PWD/data}" CEGNN_RESULTS_ROOT="${CEGNN_RESULTS_ROOT:-$PWD/results}"
CFG=configs/ncaltech101.yaml; RES="$CEGNN_RESULTS_ROOT/ncaltech101"

stage() { local s="$1" d="$2"; shift 2; [[ -e "$s" ]] && { echo ">>> SKIP $d"; return; }; echo ">>> RUN  $d"; "$@"; }

stage "$RES/baseline_widehead/best.ckpt" "baseline, head_hidden=200" \
  "$PYTHON" -u scripts/train/train_baseline.py --config $CFG --name baseline_widehead --head_hidden 200
stage "$RES/film_global_only_natural/best.ckpt" "FiLM at global pool only" \
  "$PYTHON" -u scripts/train/train_film.py --config $CFG --name film_global_only_natural --no_rate_aug --variant global_only
stage "$RES/film_rate_rateaug/best.ckpt" "FiLM from scratch + default rate-aug {0.5,1,2}" \
  "$PYTHON" -u scripts/train/train_film.py --config $CFG --name film_rate_rateaug
stage "$RES/film_rate_mildaug/best.ckpt" "FiLM from scratch + mild rate-aug {0.7,1,1.5}" \
  "$PYTHON" -u scripts/train/train_film.py --config $CFG --name film_rate_mildaug --train_rates 0.7 1.0 1.5
stage "$RES/ra_aug_only/best.ckpt" "baseline + rate augmentation only (ra-aug)" \
  "$PYTHON" -u scripts/negative_results/train_rate_aug_adaptive_graph.py --config $CFG --name ra_aug_only --graph_mode fixed

stage "$RES/film_ablation_natural_eval/natural_rate_per_bucket.csv" "natural-rate evaluation of the ablations" \
  "$PYTHON" -u scripts/eval/eval_natural_rates.py --config $CFG --name film_ablation_natural_eval --n_buckets 7 --models \
    "baseline:$RES/baseline/best.ckpt:fixed" "baseline_widehead:$RES/baseline_widehead/best.ckpt:fixed" \
    "film_global_only_natural:$RES/film_global_only_natural/best.ckpt:fixed" \
    "film_rate_natural:$RES/film_rate_natural/best.ckpt:fixed" "film_rate_rateaug:$RES/film_rate_rateaug/best.ckpt:fixed" \
    "film_rate_mildaug:$RES/film_rate_mildaug/best.ckpt:fixed"
