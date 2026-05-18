#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
# shellcheck disable=SC1091
source "${REPO_ROOT}/.starvla.env"
cd "${REPO_ROOT}"

usage() {
  cat <<USAGE
Usage:
  $(basename "$0") <ckpt_path> [num_trials_per_task]

Example:
  $(basename "$0") playground/Checkpoints/foo/checkpoints/steps_80000_pytorch_model.pt
USAGE
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage
  exit 1
fi

CKPT_INPUT="$1"
NUM_TRIALS_PER_TASK="${2:-50}"

if [[ "${CKPT_INPUT}" = /* ]]; then
  CKPT="${CKPT_INPUT}"
else
  CKPT="${REPO_ROOT}/${CKPT_INPUT}"
fi
CKPT="$(python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "${CKPT}")"

if [[ ! -f "${CKPT}" ]]; then
  echo "checkpoint not found: ${CKPT}" >&2
  exit 1
fi
if [[ "${CKPT}" != "${REPO_ROOT}"/* ]]; then
  echo "checkpoint must live under REPO_ROOT: ${REPO_ROOT}" >&2
  exit 1
fi

SUITES=(libero_spatial libero_object libero_goal libero_10)
START_PORT="${START_PORT:-6694}"
SERVER_ENV_NAME="${SERVER_ENV_NAME:-starvla}"
EVAL_ENV_NAME="${EVAL_ENV_NAME:-libero}"
LIBERO_HOME="${LIBERO_HOME:-${REPO_ROOT}/../LIBERO}"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${HOME}/.libero}"
RESULTS_ROOT="${REPO_ROOT}/results/libero"
LOCAL_LOG_ROOT="${REPO_ROOT}/.eval4_logs/libero"

find_conda_bin() {
  if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
    dirname "${CONDA_EXE}"
    return 0
  fi
  if command -v conda >/dev/null 2>&1; then
    dirname "$(command -v conda)"
    return 0
  fi
  local candidates=(
    "${HOME}/miniconda3/bin/conda"
    "${HOME}/anaconda3/bin/conda"
    "${HOME}/mambaforge/bin/conda"
    "${HOME}/miniforge3/bin/conda"
  )
  local c
  for c in "${candidates[@]}"; do
    if [[ -x "${c}" ]]; then
      dirname "${c}"
      return 0
    fi
  done
  return 1
}

CONDA_BIN_DIR="$(find_conda_bin || true)"
if [[ -z "${CONDA_BIN_DIR}" ]]; then
  echo "cannot find conda" >&2
  exit 1
fi
SERVER_PY="${CONDA_BIN_DIR}/../envs/${SERVER_ENV_NAME}/bin/python"
EVAL_PY="${CONDA_BIN_DIR}/../envs/${EVAL_ENV_NAME}/bin/python"

if [[ ! -x "${SERVER_PY}" ]]; then
  echo "server python not found: ${SERVER_PY}" >&2
  exit 1
fi
if [[ ! -x "${EVAL_PY}" ]]; then
  echo "eval python not found: ${EVAL_PY}" >&2
  exit 1
fi
if [[ ! -d "${LIBERO_HOME}" ]]; then
  echo "LIBERO_HOME not found: ${LIBERO_HOME}" >&2
  exit 1
fi

ensure_libero_config() {
  mkdir -p "${LIBERO_CONFIG_PATH}"
  cat > "${LIBERO_CONFIG_PATH}/config.yaml" <<CFG
benchmark_root: ${LIBERO_HOME}/libero/libero
bddl_files: ${LIBERO_HOME}/libero/libero/bddl_files
init_states: ${LIBERO_HOME}/libero/libero/init_files
datasets: ${DATA_ROOT}/data/libero
assets: ${LIBERO_HOME}/libero/libero/assets
CFG
}

find_free_ports() {
  local need="$1"
  local port="${START_PORT}"
  local found=()
  while [[ "${#found[@]}" -lt "${need}" ]]; do
    if ! ss -ltn | awk '{print $4}' | grep -q ":${port}"; then
      found+=("${port}")
    fi
    port=$((port + 1))
  done
  printf '%s\n' "${found[@]}"
}

find_free_server_gpus() {
  local need="$1"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
    | sort -t',' -k2,2n -k3,3n \
    | head -n "${need}" \
    | awk -F',' '{gsub(/ /, "", $1); print $1}'
}

find_render_gpu() {
  local gpu_count
  gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
  local idx
  for ((idx=0; idx<gpu_count; idx++)); do
    if MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID="${idx}" "${EVAL_PY}" - <<'PY' >/dev/null 2>&1
import mujoco  # noqa: F401
PY
    then
      echo "${idx}"
      return 0
    fi
  done
  return 1
}

ensure_libero_config
mapfile -t PORTS < <(find_free_ports 4)
mapfile -t SERVER_GPUS < <(find_free_server_gpus 4)
if [[ "${#SERVER_GPUS[@]}" -lt 4 ]]; then
  echo "need 4 GPUs, found ${#SERVER_GPUS[@]}" >&2
  exit 1
fi
RENDER_GPU="$(find_render_gpu || true)"
if [[ -z "${RENDER_GPU}" ]]; then
  echo "cannot find a usable EGL render gpu" >&2
  exit 1
fi

FOLDER_NAME="$(python3 - <<'PY' "${CKPT}"
import os, sys
ckpt = os.path.realpath(sys.argv[1])
parts = ckpt.strip('/').split('/')
print(f"{parts[-3]}_{parts[-2]}_{parts[-1]}")
PY
)"
RUN_ROOT="${RESULTS_ROOT}/${FOLDER_NAME}"
LOCAL_RUN_ROOT="${LOCAL_LOG_ROOT}/${FOLDER_NAME}"
mkdir -p "${RUN_ROOT}" "${LOCAL_RUN_ROOT}"

echo "ckpt=${CKPT}"
echo "results_root=${RUN_ROOT}"
echo "local_log_root=${LOCAL_RUN_ROOT}"
echo "render_gpu=${RENDER_GPU}"

launch_suite() {
  local suite="$1"
  local server_gpu="$2"
  local port="$3"
  local out_dir="${RUN_ROOT}/${suite}"
  local local_out_dir="${LOCAL_RUN_ROOT}/${suite}"
  mkdir -p "${out_dir}" "${local_out_dir}"

  cat > "${out_dir}/log_paths.txt" <<TXT
launcher.log -> ${local_out_dir}/launcher.log
server.log -> ${local_out_dir}/server.log
eval.log -> ${local_out_dir}/eval.log
TXT

  nohup env \
    REPO_ROOT="${REPO_ROOT}" \
    LIBERO_HOME="${LIBERO_HOME}" \
    LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH}" \
    SERVER_PY="${SERVER_PY}" \
    EVAL_PY="${EVAL_PY}" \
    CKPT="${CKPT}" \
    SUITE="${suite}" \
    SERVER_GPU="${server_gpu}" \
    RENDER_GPU="${RENDER_GPU}" \
    PORT="${port}" \
    OUT_DIR="${out_dir}" \
    LOCAL_OUT_DIR="${local_out_dir}" \
    NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK}" \
    bash -lc '
set -euo pipefail
cd "${REPO_ROOT}"
export PYTHONPATH="${LIBERO_HOME}:${REPO_ROOT}:${PYTHONPATH:-}"
export LIBERO_HOME
export LIBERO_CONFIG_PATH
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

copy_log_if_present() {
  local src="$1"
  local dst="$2"
  if [[ -f "$src" ]]; then
    cp -f "$src" "$dst.tmp" && mv -f "$dst.tmp" "$dst"
  fi
}

copy_logs_to_results() {
  copy_log_if_present "${LOCAL_OUT_DIR}/launcher.log" "${OUT_DIR}/launcher.log"
  copy_log_if_present "${LOCAL_OUT_DIR}/server.log" "${OUT_DIR}/server.log"
  copy_log_if_present "${LOCAL_OUT_DIR}/eval.log" "${OUT_DIR}/eval.log"
}

exec >"${LOCAL_OUT_DIR}/launcher.log" 2>&1

echo "$(date -Iseconds) launcher start suite=${SUITE} server_gpu=${SERVER_GPU} render_gpu=${RENDER_GPU} port=${PORT} ckpt=${CKPT} out=${OUT_DIR}"
CUDA_VISIBLE_DEVICES="${SERVER_GPU}" "${SERVER_PY}" deployment/model_server/server_policy.py \
  --ckpt_path "${CKPT}" \
  --port "${PORT}" \
  --use_bf16 \
  >"${LOCAL_OUT_DIR}/server.log" 2>&1 &
SERVER_PID=$!
echo "${SERVER_PID}" > "${OUT_DIR}/server.pid"
cleanup() {
  kill "${SERVER_PID}" 2>/dev/null || true
  copy_logs_to_results || true
}
trap cleanup EXIT

READY=0
for _ in $(seq 1 180); do
  if PORT="${PORT}" python3 - <<"PY"
import os, socket, sys
port = int(os.environ["PORT"])
s = socket.socket()
s.settimeout(1)
try:
    s.connect(("127.0.0.1", port))
except Exception:
    sys.exit(1)
else:
    s.close()
    sys.exit(0)
PY
  then
    READY=1
    break
  fi
  sleep 1
done
if [[ "${READY}" -ne 1 ]]; then
  echo "server not ready on port ${PORT}" >&2
  exit 1
fi

echo "$(date -Iseconds) server ready suite=${SUITE} port=${PORT}"
MUJOCO_EGL_DEVICE_ID="${RENDER_GPU}" "${EVAL_PY}" ./examples/LIBERO/eval_files/eval_libero.py \
  --args.pretrained-path "${CKPT}" \
  --args.host 127.0.0.1 \
  --args.port "${PORT}" \
  --args.task-suite-name "${SUITE}" \
  --args.num-trials-per-task "${NUM_TRIALS_PER_TASK}" \
  --args.video-out-path "${OUT_DIR}" \
  >"${LOCAL_OUT_DIR}/eval.log" 2>&1
STATUS=$?
echo "${STATUS}" > "${OUT_DIR}/eval.exitcode"
echo "$(date -Iseconds) eval exit suite=${SUITE} status=${STATUS}"
copy_logs_to_results
exit "${STATUS}"
' > /dev/null 2>&1 < /dev/null &

  local launch_pid=$!
  echo "${launch_pid}" > "${out_dir}/launcher.pid"
  echo "${suite} server_gpu=${server_gpu} render_gpu=${RENDER_GPU} port=${port} launch_pid=${launch_pid} out=${out_dir}"
}

for i in "${!SUITES[@]}"; do
  launch_suite "${SUITES[$i]}" "${SERVER_GPUS[$i]}" "${PORTS[$i]}"
done
