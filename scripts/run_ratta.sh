#!/usr/bin/env bash
# RA-TTA (rate-prompt adapter, 18K params) on top of a frozen base model - trained on the extreme
# synthetic range {0.125 ... 8x} - then evaluated on the synthetic grid and on natural-rate buckets.
#   bash scripts/run_ratta.sh ncaltech101 film_rate_curriculum
set -euo pipefail
cd "$(dirname "$0")/.."
DS="${1:?dataset}"; BASE="${2:-film_rate_curriculum}"
PYTHON="${PYTHON:-python}"; export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export CEGNN_DATA_ROOT="${CEGNN_DATA_ROOT:-$PWD/data}" CEGNN_RESULTS_ROOT="${CEGNN_RESULTS_ROOT:-$PWD/results}"
CFG=configs/$DS.yaml; RES="$CEGNN_RESULTS_ROOT/$DS"; RATES="0.125 0.25 0.5 1.0 2.0 4.0 8.0"
NAME="ratta_on_$BASE"

[[ -e "$RES/$NAME/best.ckpt" ]] || "$PYTHON" -u scripts/train/train_rptta.py --config $CFG \
    --base "$RES/$BASE/best.ckpt" --name "$NAME" --rates $RATES
"$PYTHON" -u scripts/eval/eval_rptta.py --config $CFG --base "$RES/$BASE/best.ckpt" \
    --rptta "$RES/$NAME/best.ckpt" --name "${NAME}_eval" --rates $RATES
"$PYTHON" -u scripts/eval/eval_natural_rates.py --config $CFG --name "${NAME}_natural" --n_buckets 7 \
    --rptta_base "$RES/$BASE/best.ckpt" --rptta_ckpt "$RES/$NAME/best.ckpt"
