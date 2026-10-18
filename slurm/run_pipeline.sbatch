#!/bin/bash
# Example SLURM wrapper for the full pipeline of one dataset. Adapt partition / GPU / paths.
#   sbatch --export=ALL,DS=ncaltech101 slurm/run_pipeline.sbatch
#SBATCH --job-name=cegnn
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/cegnn_%x_%j.out
set -euo pipefail
mkdir -p logs
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cegnn
export CEGNN_DATA_ROOT="${CEGNN_DATA_ROOT:?export CEGNN_DATA_ROOT=/path/to/datasets}"
export CEGNN_RESULTS_ROOT="${CEGNN_RESULTS_ROOT:-$PWD/results}"
bash scripts/run_pipeline.sh "${DS:?pass --export=ALL,DS=<dataset>}"
