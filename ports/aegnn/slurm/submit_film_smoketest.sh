#!/bin/bash
#SBATCH --job-name=aegnn_film_smoke
#SBATCH --partition=day
#SBATCH --gres=gpu:A4000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/film_smoketest_%j.out
#SBATCH --error=logs/film_smoketest_%j.err
#
# Second architecture in the cross-backbone robustness study: FiLM-Rate
# curriculum applied to AEGNN's GraphRes on N-Caltech101. Reuses the existing
# "aegnn" conda env (already has torch_geometric working). 2-epoch smoke test
# of baseline/stage1/stage2/eval to validate the pipeline before a full run.

set -euo pipefail
set -x

echo "============================================================"
echo "AEGNN FiLM-Rate setup + smoke test on $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-N/A}  Started: $(date)"
echo "============================================================"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate aegnn

nvidia-smi
python -c "import torch, torch_geometric; print('torch', torch.__version__, 'pyg', torch_geometric.__version__, 'cuda avail', torch.cuda.is_available())"

cd "${AEGNN_REPO:?set AEGNN_REPO to the patched AEGNN checkout (see ports/README.md)}"
mkdir -p logs runs_smoketest

PROCESSED_DIR=${CEGNN_DATA_ROOT}/main_split_claude/processed_3d_10k
OUT=./runs_smoketest

python train_film_curriculum.py --processed_dir ${PROCESSED_DIR} --out_dir ${OUT} --run baseline --epochs_baseline 2
python train_film_curriculum.py --processed_dir ${PROCESSED_DIR} --out_dir ${OUT} --run stage1 --epochs_stage1 2
python train_film_curriculum.py --processed_dir ${PROCESSED_DIR} --out_dir ${OUT} --run stage2 --epochs_stage2 2
python train_film_curriculum.py --processed_dir ${PROCESSED_DIR} --out_dir ${OUT} --run eval_all

echo "============================================================"
echo "Smoke test finished: $(date)"
echo "============================================================"
