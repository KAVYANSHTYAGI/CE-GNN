#!/bin/bash
#SBATCH --job-name=aegnn_film_verify
#SBATCH --partition=day
#SBATCH --gres=gpu:A4000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/film_verify_%j.out
#SBATCH --error=logs/film_verify_%j.err
#
# Verify the augmentation + LR-decay + best-checkpoint fix actually closes
# the gap (prior runs without it: 28-31% test acc despite 99%+ train acc;
# this same architecture/dataset reaches up to 77% val_acc with these
# regularizers per prior train.py checkpoints). Short run (8 epochs/stage)
# just to confirm val_acc climbs well past 31% before committing to a full
# multi-day run.

set -euo pipefail
set -x
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate aegnn

cd "${AEGNN_REPO:?set AEGNN_REPO to the patched AEGNN checkout (see ports/README.md)}"
mkdir -p logs runs_verify

PROCESSED_DIR=${CEGNN_DATA_ROOT}/main_split_claude/processed_3d_10k
OUT=./runs_verify

python train_film_curriculum.py --processed_dir ${PROCESSED_DIR} --out_dir ${OUT} --run baseline --epochs_baseline 4
python train_film_curriculum.py --processed_dir ${PROCESSED_DIR} --out_dir ${OUT} --run stage1 --epochs_stage1 4
python train_film_curriculum.py --processed_dir ${PROCESSED_DIR} --out_dir ${OUT} --run stage2 --epochs_stage2 4
python train_film_curriculum.py --processed_dir ${PROCESSED_DIR} --out_dir ${OUT} --run eval_all

echo "Verify run finished: $(date)"
