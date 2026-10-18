#!/bin/bash
#SBATCH --job-name=evmamba_full
#SBATCH --partition=week
#SBATCH --gres=gpu:A4000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=6-00:00:00
#SBATCH --output=logs/full_%j.out
#SBATCH --error=logs/full_%j.err
#
# Full-length run: smoke test (2 epochs/stage) already showed a dramatic
# curriculum effect (74.2% natural vs 52.4% baseline, 67.3% vs 39.1%
# bucket-worst) on this second architecture (point-cloud+SSM). Data is
# fully cached on disk. Now run closer to a real schedule.

set -euo pipefail
set -x
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate eventmamba
export CUDA_HOME=${CONDA_PREFIX}
export PATH=${CUDA_HOME}/bin:${PATH}

cd "${EVENTMAMBA_REPO:?set EVENTMAMBA_REPO (see ports/README.md)}"
DATA_DIR=${EVENTMAMBA_REPO}/data/dvsgesture_rate
OUT=./runs_full
mkdir -p ${OUT} logs

python train_film_curriculum.py --data_dir ${DATA_DIR} --out_dir ${OUT} --num_classes 10 --run baseline --epochs_baseline 40
python train_film_curriculum.py --data_dir ${DATA_DIR} --out_dir ${OUT} --num_classes 10 --run stage1 --epochs_stage1 40
python train_film_curriculum.py --data_dir ${DATA_DIR} --out_dir ${OUT} --num_classes 10 --run stage2 --epochs_stage2 15
python train_film_curriculum.py --data_dir ${DATA_DIR} --out_dir ${OUT} --num_classes 10 --run eval_all

echo "Full run finished: $(date)"
