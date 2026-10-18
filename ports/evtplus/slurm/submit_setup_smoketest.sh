#!/bin/bash
#SBATCH --job-name=evtplus_setup
#SBATCH --partition=day
#SBATCH --gres=gpu:A4000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/setup_smoketest_%j.out
#SBATCH --error=logs/setup_smoketest_%j.err
#
# Fourth architecture (attention/transformer paradigm): FiLM-Rate curriculum
# on EvT+ / DVS128 Gesture. Builds a dedicated env, generates frame data
# directly from our existing raw .npy trials (bypassing aermanager/AEDAT
# parsing), and runs a short smoke test of baseline/stage1/stage2/eval.

set -euo pipefail
set -x

echo "============================================================"
echo "EvT+ FiLM-Rate setup + smoke test on $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-N/A}  Started: $(date)"
echo "============================================================"

source "$(conda info --base)/etc/profile.d/conda.sh"

ENV_NAME=evtplus
if ! conda env list | grep -q "^${ENV_NAME} "; then
    conda create -y -n ${ENV_NAME} python=3.8
fi
conda activate ${ENV_NAME}

nvidia-smi
python -c "import torch" 2>/dev/null || \
  conda install -y pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch

python - <<'EOF'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
EOF

pip install -q tqdm scipy scikit-learn numpy pandas einops

cd "${EVT_REPO:?set EVT_REPO (see ports/README.md)}"

# ---- data prep: reuse our existing raw DVS128 Gesture archive ----
RAW_ROOT=${CEGNN_DATA_ROOT}/dvsgesture_raw/DVSGesture
DATA_ROOT=${EVT_REPO}/data/dvsgesture_rate
mkdir -p ${DATA_ROOT}

cd dataset_scripts
if [ ! -d "${DATA_ROOT}/natural/train" ]; then
    python dvs128_to_frames_direct.py --root_dir ${RAW_ROOT} --export_path ${DATA_ROOT}
fi
find ${DATA_ROOT} -maxdepth 2 -type d | sort

# ---- smoke test: short run of baseline/stage1/stage2/eval ----
cd ${EVT_REPO}/classification
OUT=../runs_smoketest
mkdir -p ${OUT}
python train_film_curriculum.py --data_root ${DATA_ROOT} --out_dir ${OUT} --run baseline --epochs_baseline 2
python train_film_curriculum.py --data_root ${DATA_ROOT} --out_dir ${OUT} --run stage1 --epochs_stage1 2
python train_film_curriculum.py --data_root ${DATA_ROOT} --out_dir ${OUT} --run stage2 --epochs_stage2 2
python train_film_curriculum.py --data_root ${DATA_ROOT} --out_dir ${OUT} --run eval_all

echo "============================================================"
echo "Smoke test finished: $(date)"
echo "============================================================"
