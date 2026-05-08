#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_GIT_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
source "${REPO_GIT_ROOT}/.starvla.env"
cd "${REPO_ROOT}"
python3 starVLA/path_tool.py setup-links >/dev/null

# Debug: print current python environment
echo "Using Python: $(which python)"

### MANUALLY SET THESE ###
# set necessary environment variables
export star_vla_python="${STARVLA_PYTHON:-python}"
export sim_python="${SIM_PYTHON:-python}"
export TASKS_JSONL_PATH="${TASKS_JSONL_PATH:-examples/Behavior/tasks.jsonl}"
export BEHAVIOR_ASSET_PATH="${BEHAVIOR_ASSET_PATH:-playground/Datasets/behavior-1k}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# set model path and port
MODEL_PATH="${CKPT:-playground/Checkpoints/BEHAVIOR-QwenDual-Pretrained-224/checkpoints/steps_300000_pytorch_model.pt}"
PORT="${PORT:-10197}"
WRAPPERS="RGBLowResWrapper" # DefaultWrapper, RGBLowResWrapper or RichObservationWrapper
USE_STATE=True  

# set task name
TASK_NAME="turning_on_radio" 
EVAL_INSTANCE_IDS="0"
### END OF MANUALLY SETUP ###


# Force Vulkan to use only the NVIDIA ICD to avoid duplicate ICDs seen by the loader
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
# Prefer NVIDIA GLX vendor when any GL deps are touched
export __GLX_VENDOR_LIBRARY_NAME=nvidia

# run single task
export DEBUG=true
echo "▶️ Running task '${TASK_NAME}'..."
CUDA_VISIBLE_DEVICES=5 ${sim_python} examples/Behavior/start_behavior_env.py \
    --ckpt-path ${MODEL_PATH} \
    --eval-instance-ids \"${EVAL_INSTANCE_IDS}\"  \
    --eval-on-train-instances True \
    --port ${PORT} \
    --task-name ${TASK_NAME} \
    --behavior-tasks-jsonl-path ${TASKS_JSONL_PATH} \
    --behavior-asset-path ${BEHAVIOR_ASSET_PATH} \
    --wrappers ${WRAPPERS} \
    --use-state ${USE_STATE}
    

# stop server
echo "⏹️ Stopping server (PID: ${SERVER_PID})..."
kill ${SERVER_PID}
wait ${SERVER_PID} 2>/dev/null
echo "✅ Server stopped"
