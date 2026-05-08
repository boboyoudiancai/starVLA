#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_GIT_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
source "${REPO_GIT_ROOT}/.starvla.env"
cd "${REPO_ROOT}"
python3 starVLA/path_tool.py setup-links >/dev/null

CKPT="${CKPT:-playground/Checkpoints/0405_libero4in1_CosmoPredict2GR00T/checkpoints/steps_50000_pytorch_model.pt}"

###########################################################################################
# === Please modify the following paths according to your environment ===
export LIBERO_HOME="${LIBERO_HOME:-${REPO_ROOT}/../LIBERO}"
export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero"
export LIBERO_Python="${LIBERO_PYTHON:-python}"

export PYTHONPATH="${LIBERO_HOME}:${REPO_ROOT}:${PYTHONPATH:-}"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

host="127.0.0.1"
base_port=6694
unnorm_key="franka"
your_ckpt="${CKPT}"

# export DEBUG=true

folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')
# model_root: playground/Checkpoints/<run_id>
model_root=$(echo "$your_ckpt" | awk -F'/checkpoints/' '{print $1}')
# === End of environment variable configuration ===
###########################################################################################

task_suite_name=libero_goal
num_trials_per_task=50
video_out_path="${model_root}/results/${task_suite_name}/${folder_name}"

${LIBERO_Python} ./examples/LIBERO/eval_files/eval_libero.py \
    --args.pretrained-path ${your_ckpt} \
    --args.host "$host" \
    --args.port $base_port \
    --args.task-suite-name "$task_suite_name" \
    --args.num-trials-per-task "$num_trials_per_task" \
    --args.video-out-path "$video_out_path"
