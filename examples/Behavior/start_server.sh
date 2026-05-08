#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_GIT_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
source "${REPO_GIT_ROOT}/.starvla.env"
cd "${REPO_ROOT}"
python3 starVLA/path_tool.py setup-links >/dev/null

# Debug: Print the current Python environment
echo "Using Python: $(which python)"

### MANUALLY SET THESE ###

# set necessary environment variables
export star_vla_python="${STARVLA_PYTHON:-python}"
export sim_python="${SIM_PYTHON:-python}"
export BEHAVIOR_ASSET_PATH="${BEHAVIOR_ASSET_PATH:-playground/Datasets/behavior-1k}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"



# set model path and port
MODEL_PATH="${CKPT:-playground/Checkpoints/BEHAVIOR-QwenDual-Pretrained-224/checkpoints/steps_300000_pytorch_model.pt}"
PORT="${PORT:-10197}"
WRAPPERS="DefaultWrapper"
USE_STATE=True  # whether to use state as part of the observation

# Configure task name
TASK_NAME="turning_on_radio"  
### END OF MANUALLY SETUP ###

# Force Vulkan to use only the NVIDIA ICD to avoid duplicate ICDs seen by the loader
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
# Prefer NVIDIA GLX vendor when any GL deps are touched
export __GLX_VENDOR_LIBRARY_NAME=nvidia

# Start server
echo "▶️ Starting server on port ${PORT}..."
CUDA_VISIBLE_DEVICES=5 ${star_vla_python} deployment/model_server/server_policy.py \
    --ckpt_path ${MODEL_PATH} \
    --port ${PORT} \
    --is_debug \
    --use_bf16
