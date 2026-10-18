#!/bin/bash
#SBATCH --job-name=aegnn_film_full
#SBATCH --partition=week
#SBATCH --gres=gpu:A4000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=6-00:00:00
#SBATCH --output=logs/film_full_%j.out
#SBATCH --error=logs/film_full_%j.err
#
# Full-length run of the corrected (augmentation + LR-decay + best-checkpoint)
# AEGNN FiLM-Rate curriculum, following the verify job (2305911) that
# confirmed the fix climbs cleanly (18.86 -> 31.44 -> 33.32 natural mean at
# just 4 epochs/stage, ordering baseline < film_no_aug < film_curriculum
# already holding). At ~18 min/epoch observed in the verify run, 80+80+30
# epochs is budgeted at ~57h; requesting the week partition (7-day cap) with
# a 6-day wall time for margin.

set -euo pipefail
set -x
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate aegnn

cd "${AEGNN_REPO:?set AEGNN_REPO to the patched AEGNN checkout (see ports/README.md)}"
mkdir -p logs runs_full

PROCESSED_DIR=${CEGNN_DATA_ROOT}/main_split_claude/processed_3d_10k
OUT=./runs_full

python train_film_curriculum.py --processed_dir ${PROCESSED_DIR} --out_dir ${OUT} --run baseline --epochs_baseline 80
python train_film_curriculum.py --processed_dir ${PROCESSED_DIR} --out_dir ${OUT} --run stage1 --epochs_stage1 80
python train_film_curriculum.py --processed_dir ${PROCESSED_DIR} --out_dir ${OUT} --run stage2 --epochs_stage2 30
python train_film_curriculum.py --processed_dir ${PROCESSED_DIR} --out_dir ${OUT} --run eval_all

echo "Full run finished: $(date)"
