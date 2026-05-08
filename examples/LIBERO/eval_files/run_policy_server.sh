#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_GIT_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
source "${REPO_GIT_ROOT}/.starvla.env"
cd "${REPO_ROOT}"
python3 starVLA/path_tool.py setup-links >/dev/null

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export star_vla_python="${STARVLA_PYTHON:-python}"
your_ckpt="${CKPT:-playground/Pretrained_models/StarVLA/Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt}"
gpu_id="${GPU_ID:-0}"
port="${PORT:-6694}"
################# star Policy Server ######################

# export DEBUG=true
CUDA_VISIBLE_DEVICES=$gpu_id ${star_vla_python} deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16

# #################################
