#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_GIT_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
source "${REPO_GIT_ROOT}/.starvla.env"
cd "${REPO_ROOT}"
python3 starVLA/path_tool.py setup-links >/dev/null

echo "Using Python: $(which python)"
export star_vla_python="${STARVLA_PYTHON:-python}"
export sim_python="${SIM_PYTHON:-python}"
export BEHAVIOR_PATH="${BEHAVIOR_PATH:-playground/Datasets/behavior-1k}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Configure model path and port
MODEL_PATH="${CKPT:-playground/Pretrained_models/Qwen3-VL-GR00T-Behavior-nostate/checkpoints/steps_20000_pytorch_model.pt}"
PORT="${PORT:-10197}"
WRAPPERS="DefaultWrapper"
USE_STATE=False  # Whether to use state as part of the observation

# Configure task name
TASK_NAME="turning_on_radio"  # Choose a simple task
LOG_FILE="$(dirname "${MODEL_PATH}")/client_logs/log_${TASK_NAME}.txt"
SERVER_LOG_FILE="$(dirname "${MODEL_PATH}")/server_logs/log_${TASK_NAME}.txt"
mkdir -p "$(dirname "${LOG_FILE}")" "$(dirname "${SERVER_LOG_FILE}")"

# Start server
echo "▶️ Starting server on port ${PORT}..."
CUDA_VISIBLE_DEVICES=0 ${star_vla_python} deployment/model_server/server_policy.py \
    --ckpt_path ${MODEL_PATH} \
    --port ${PORT} \
    --is_debug \
    --use_bf16
    
    #  > ${SERVER_LOG_FILE} 2>&1 &


# SERVER_PID=$!
# sleep 15  # Wait for server to start

# Check if server started successfully
if ps -p ${SERVER_PID} > /dev/null; then
    echo "✅ Server started successfully (PID: ${SERVER_PID})"
else
    echo "❌ Failed to start server"
    exit 1
fi
