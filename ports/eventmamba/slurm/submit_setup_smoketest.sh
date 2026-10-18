#!/bin/bash
#SBATCH --job-name=evmamba_setup
#SBATCH --partition=day
#SBATCH --gres=gpu:A4000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/setup_smoketest_%j.out
#SBATCH --error=logs/setup_smoketest_%j.err
#
# Step 1: build a dedicated EventMamba conda env (needs GPU node for nvcc to
# compile mamba-ssm/causal-conv1d CUDA extensions), preprocess DVS128 Gesture
# with per-sample event-rate metadata + the multi-window synthetic-rate pool,
# then run a 2-epoch smoke test of baseline / stage1 / stage2 / eval to
# validate the whole FiLM-Rate-on-EventMamba pipeline before committing to a
# full training run.

set -euo pipefail
set -x

echo "============================================================"
echo "EventMamba FiLM-Rate setup + smoke test on $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-N/A}  Started: $(date)"
echo "============================================================"

source "$(conda info --base)/etc/profile.d/conda.sh"

ENV_NAME=eventmamba
if ! conda env list | grep -q "^${ENV_NAME} "; then
    conda create -y -n ${ENV_NAME} python=3.8
fi
conda activate ${ENV_NAME}

nvidia-smi
nvcc --version 2>/dev/null || echo "nvcc not found via PATH; installing conda's cuda-nvcc"

python -c "import torch" 2>/dev/null || \
  conda install -y pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia

# BUGFIX: the compute node has the NVIDIA driver but no CUDA dev toolkit on
# PATH, so causal-conv1d/mamba-ssm's setup.py can't find nvcc and fails with
# "NameError: name 'bare_metal_version' is not defined". Install the matching
# CUDA 12.1 nvcc via conda (must match pytorch-cuda=12.1 above) before
# building either extension.
nvcc --version 2>/dev/null || conda install -y -c nvidia cuda-nvcc=12.1 cuda-cudart-dev=12.1
export CUDA_HOME=${CONDA_PREFIX}
export PATH=${CUDA_HOME}/bin:${PATH}
nvcc --version

python - <<'EOF'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
EOF

pip install -q h5py tqdm scikit-learn tensorboard timm
python -c "import spikingjelly" 2>/dev/null || pip install -q spikingjelly
python -c "import causal_conv1d" 2>/dev/null || pip install -q causal-conv1d --no-build-isolation
python -c "import mamba_ssm" 2>/dev/null || pip install -q mamba-ssm --no-build-isolation
# triton backs mamba_ssm's CUDA selective-scan kernels; install as a general
# safety net (models/mamba_layer.py no longer hard-requires its layernorm
# submodule specifically -- see that file's BUGFIX comment on the
# bimamba_type/unofficial-fork mismatch and the renamed layernorm module).
python -c "import triton" 2>/dev/null || pip install -q triton==2.1.0

python - <<'EOF'
import torch, mamba_ssm, spikingjelly, causal_conv1d
print("all imports OK")
print("mamba_ssm", mamba_ssm.__file__)
EOF

# ---- data prep: reuse our existing raw DVS128 Gesture archive ----
RAW_ROOT=${CEGNN_DATA_ROOT}/dvsgesture_raw/DVSGesture
DATA_DIR=${EVENTMAMBA_REPO}/data/dvsgesture_rate
mkdir -p ${DATA_DIR}

cd ${EVENTMAMBA_REPO}/dataprocess
# BUGFIX: spikingjelly's DVS128Gesture refuses to run unless it finds the
# single official DvsGesture.tar.gz under <root>/download/. Our raw data is
# already extracted into the ibmGestureTrain/ibmGestureTest per-trial .npy
# layout instead, so read it directly (no spikingjelly dependency).
# BUGFIX: no longer gated on train_natural.h5 existing -- the script itself
# is now resumable per-split-file (see maybe_generate()), so always call it;
# it skips whatever's already on disk and only regenerates what's missing
# (e.g. after the previous run's OOM kill during w1.0 generation).
python generate_dvsgesture_rate_direct.py \
    --root_dir ${RAW_ROOT} \
    --export_path ${DATA_DIR} \
    --num_points 1024 \
    --step_size 0.25 \
    --natural_window 0.5 \
    --aug_windows 0.25 0.5 1.0
ls -la ${DATA_DIR}
cat ${DATA_DIR}/label_remap.txt
NUM_CLASSES=$(python3 -c "print(len(eval(open('${DATA_DIR}/label_remap.txt').read())))")
echo "NUM_CLASSES=${NUM_CLASSES}"

# ---- smoke test: 2 epochs each stage, just to validate correctness ----
cd "${EVENTMAMBA_REPO:?set EVENTMAMBA_REPO (see ports/README.md)}"
OUT=./runs_smoketest
mkdir -p ${OUT}
python train_film_curriculum.py --data_dir ${DATA_DIR} --out_dir ${OUT} --num_classes ${NUM_CLASSES} --run baseline --epochs_baseline 2
python train_film_curriculum.py --data_dir ${DATA_DIR} --out_dir ${OUT} --num_classes ${NUM_CLASSES} --run stage1 --epochs_stage1 2
python train_film_curriculum.py --data_dir ${DATA_DIR} --out_dir ${OUT} --num_classes ${NUM_CLASSES} --run stage2 --epochs_stage2 2
python train_film_curriculum.py --data_dir ${DATA_DIR} --out_dir ${OUT} --num_classes ${NUM_CLASSES} --run eval_all

echo "============================================================"
echo "Smoke test finished: $(date)"
echo "============================================================"
