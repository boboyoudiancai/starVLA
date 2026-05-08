#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_GIT_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
source "${REPO_GIT_ROOT}/.starvla.env"
cd "${REPO_ROOT}"
python3 starVLA/path_tool.py setup-links >/dev/null

# run_policy_server.sh
#
# Launches the starVLA WebSocket policy server for VLA-Arena evaluation.
# Run this script first, then launch eval_vla_arena.sh in a separate terminal.

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

###########################################################################################
# === Please modify the following paths according to your environment ===
export starVLA_python="${STARVLA_PYTHON:-python}"
your_ckpt="${CKPT:-playground/Checkpoints/qwen2.5-libero/checkpoints/steps_30000_pytorch_model.pt}"
gpu_id="${GPU_ID:-7}"
port="${PORT:-1009${gpu_id}}"
###########################################################################################

CUDA_VISIBLE_DEVICES=${gpu_id} ${starVLA_python} deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16
