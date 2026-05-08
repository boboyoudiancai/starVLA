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
MODEL_PATH="${CKPT:-playground/Checkpoints/1007_qwenLargefm/checkpoints/steps_20000_pytorch_model.pt}"
PORT="${PORT:-10197}"
WRAPPERS="RGBLowResWrapper"
USE_STATE=False  # Whether to use state as part of the observation

# Configure task name
TASK_NAME="turning_on_radio"  # Choose a simple task
LOG_FILE="$(dirname "${MODEL_PATH}")/log_${TASK_NAME}.txt"

# Start server
echo "▶️ Starting server on port ${PORT}..."
CUDA_VISIBLE_DEVICES=0 ${star_vla_python} deployment/model_server/server_policy.py \
    --ckpt_path ${MODEL_PATH} \
    --port ${PORT} \
    --use_bf16 > server_log.txt 2>&1 &

SERVER_PID=$!
sleep 15  # Wait for server to start

# Check if server started successfully
if ps -p ${SERVER_PID} > /dev/null; then
    echo "✅ Server started successfully (PID: ${SERVER_PID})"
else
    echo "❌ Failed to start server"
    exit 1
fi

# Run a single task
echo "▶️ Running task '${TASK_NAME}'..."
CUDA_VISIBLE_DEVICES=0 ${sim_python} examples/Behavior/start_behavior_env.py \
    --ckpt-path ${MODEL_PATH} \
    --eval-on-train-instances True \
    --port ${PORT} \
    --task-name ${TASK_NAME} \
    --behaviro-data-path ${BEHAVIOR_PATH} \
    --wrappers ${WRAPPERS} \
    --use-state ${USE_STATE} > ${LOG_FILE} 2>&1

# Check if task completed
if [ $? -eq 0 ]; then
    echo "✅ Task '${TASK_NAME}' completed successfully. Log: ${LOG_FILE}"
else
    echo "❌ Task '${TASK_NAME}' failed. Check log: ${LOG_FILE}"
fi

# Stop server
echo "⏹️ Stopping server (PID: ${SERVER_PID})..."
kill ${SERVER_PID}
wait ${SERVER_PID} 2>/dev/null
echo "✅ Server stopped"
