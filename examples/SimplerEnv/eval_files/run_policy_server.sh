#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_GIT_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
source "${REPO_GIT_ROOT}/.starvla.env"
cd "${REPO_ROOT}"
python3 starVLA/path_tool.py setup-links >/dev/null

export star_vla_python="${STARVLA_PYTHON:-python}"
export sim_python="${SIM_PYTHON:-python}"
export SimplerEnv_PATH="${SIMPLERENV_PATH:-${REPO_ROOT}/../SimplerEnv}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
port="${PORT:-6678}"
gpu_id="${GPU_ID:-0}"
your_ckpt="${CKPT:-playground/Checkpoints/0418_oxe_bridge_rt_1_QwenGR00T/checkpoints/steps_10000_pytorch_model.pt}"

ckpt_dir=$(dirname "${your_ckpt}")
ckpt_base=$(basename "${your_ckpt}")
ckpt_name="${ckpt_base%.*}"
output_server_dir="${ckpt_dir}/output_server"
mkdir -p "${output_server_dir}"
log_file="${output_server_dir}/${ckpt_name}_policy_server_${port}.log"


#### run server #####
CUDA_VISIBLE_DEVICES=${gpu_id} ${star_vla_python} deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16 \
    2>&1 | tee "${log_file}"
