#!/bin/bash
#SBATCH --job-name=evmamba_lighting_eval
#SBATCH --partition=day
#SBATCH --gres=gpu:A4000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/lighting_eval_%j.out
#SBATCH --error=logs/lighting_eval_%j.err

set -euo pipefail
set -x
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate eventmamba

cd "${EVENTMAMBA_REPO:?set EVENTMAMBA_REPO (see ports/README.md)}"
python eval_lighting_stratified.py --data_dir ./data/dvsgesture_rate --out_dir ./runs_full
