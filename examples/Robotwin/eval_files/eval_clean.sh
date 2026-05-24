#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if REPO_ROOT_GUESS="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel 2>/dev/null)"; then
    :
else
    REPO_ROOT_GUESS="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
fi
ENV_FILE="${STARVLA_ENV_FILE:-${REPO_ROOT_GUESS}/.starvla.env}"

if [[ ! -f "${ENV_FILE}" ]]; then
    echo "starVLA env file not found: ${ENV_FILE}" >&2
    exit 1
fi

# shellcheck disable=SC1090
source "${ENV_FILE}"

REPO_ROOT="${REPO_ROOT:-${REPO_ROOT_GUESS}}"
DATA_ROOT="${DATA_ROOT:-}"
cd "${REPO_ROOT}"

usage() {
    cat >&2 <<USAGE
Usage:
  CUDA_DEVICES=<gpu_ids> bash $(basename "$0") <ckpt_path> [policy_name]

Example:
  CUDA_DEVICES=0,1,2,3,4 bash $(basename "$0") playground/Checkpoints/<model>/checkpoints/<step>.pt

Environment:
  CUDA_DEVICES                    Candidate physical GPUs. Required.
  ROBOTWIN_PATH                   RoboTwin checkout path. Default: REPO_ROOT/../RoboTwin.
  ROBOTWIN_FREE_GPU_MAX_MEM_MB    Max used memory for a GPU to be treated as free. Default: 1024.
  ROBOTWIN_BASE_PORT              First candidate policy-server port. Default: 5694.
  ROBOTWIN_SERVER_TIMEOUT         Policy-server readiness timeout. Default: 1800.
USAGE
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
    usage
    exit 1
fi

if [[ -z "${CUDA_DEVICES:-}" ]]; then
    echo "CUDA_DEVICES is required, e.g. CUDA_DEVICES=0,1,2,3,4" >&2
    exit 1
fi

CKPT_INPUT="$1"
POLICY_NAME="${2:-robotwin_clean}"

if [[ "${CKPT_INPUT}" = /* ]]; then
    CKPT_PATH="${CKPT_INPUT}"
else
    CKPT_PATH="${REPO_ROOT}/${CKPT_INPUT}"
fi
CKPT_PATH="$(python3 -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' "${CKPT_PATH}")"

if [[ ! -f "${CKPT_PATH}" ]]; then
    echo "checkpoint not found: ${CKPT_PATH}" >&2
    exit 1
fi

ROBOTWIN_PATH="${ROBOTWIN_PATH:-${REPO_ROOT}/../RoboTwin}"
if [[ ! -d "${ROBOTWIN_PATH}" ]]; then
    echo "ROBOTWIN_PATH not found: ${ROBOTWIN_PATH}" >&2
    exit 1
fi

trim() {
    local value="$1"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf '%s' "${value}"
}

find_conda_python() {
    local env_name="$1"
    local -a search_dirs=()

    if [[ -n "${CONDA_EXE:-}" ]]; then
        search_dirs+=("$(dirname "$(dirname "${CONDA_EXE}")")/envs")
    fi
    if [[ -n "${CONDA_PREFIX:-}" ]]; then
        search_dirs+=("$(dirname "${CONDA_PREFIX}")")
    fi
    search_dirs+=(
        "${HOME}/miniconda3/envs"
        "${HOME}/anaconda3/envs"
        "${HOME}/miniforge3/envs"
        "${HOME}/mambaforge/envs"
        "/opt/conda/envs"
    )

    local base
    for base in "${search_dirs[@]}"; do
        if [[ -x "${base}/${env_name}/bin/python" ]]; then
            printf '%s\n' "${base}/${env_name}/bin/python"
            return 0
        fi
    done

    return 1
}

resolve_python() {
    local explicit_path="$1"
    local env_name="$2"

    if [[ -n "${explicit_path}" ]]; then
        if [[ ! -x "${explicit_path}" ]]; then
            echo "python not executable: ${explicit_path}" >&2
            exit 1
        fi
        printf '%s\n' "${explicit_path}"
        return 0
    fi

    if ! find_conda_python "${env_name}"; then
        echo "cannot find Python for conda env: ${env_name}" >&2
        exit 1
    fi
}

gpu_used_memory_mb() {
    local gpu_id="$1"
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | awk -F',' -v target="${gpu_id}" '
            {
                gsub(/ /, "", $1)
                gsub(/ /, "", $2)
                if ($1 == target) {
                    print $2
                    found = 1
                }
            }
            END {
                if (!found) {
                    exit 1
                }
            }'
}

select_free_gpus() {
    local max_mem="${ROBOTWIN_FREE_GPU_MAX_MEM_MB:-1024}"
    local -a candidates=()
    local -a selected=()
    local raw_gpu=""
    local gpu_id=""
    local used_mem=""

    IFS=',' read -ra candidates <<< "${CUDA_DEVICES}"
    for raw_gpu in "${candidates[@]}"; do
        gpu_id="$(trim "${raw_gpu}")"
        if [[ -z "${gpu_id}" ]]; then
            continue
        fi
        used_mem="$(gpu_used_memory_mb "${gpu_id}" 2>/dev/null || true)"
        if [[ -z "${used_mem}" ]]; then
            echo "skip unknown gpu: ${gpu_id}" >&2
            continue
        fi
        if (( used_mem <= max_mem )); then
            selected+=("${gpu_id}")
        else
            echo "skip busy gpu=${gpu_id} used_mem_mb=${used_mem} threshold_mb=${max_mem}" >&2
        fi
    done

    if (( ${#selected[@]} == 0 )); then
        echo "no free GPU found in CUDA_DEVICES=${CUDA_DEVICES}" >&2
        exit 1
    fi

    local IFS=,
    printf '%s\n' "${selected[*]}"
}

STARVLA_PYTHON="$(resolve_python "${STARVLA_PYTHON:-}" "${ROBOTWIN_STARVLA_ENV:-starvla}")"
ROBOTWIN_PYTHON="$(resolve_python "${ROBOTWIN_PYTHON:-}" "${ROBOTWIN_ENV:-robotwin}")"
export STARVLA_PYTHON ROBOTWIN_PYTHON ROBOTWIN_PATH
export PATH="$(dirname "${ROBOTWIN_PYTHON}"):$(dirname "${STARVLA_PYTHON}"):${PATH}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

STARVLA_SITE="$("${STARVLA_PYTHON}" - <<'PY'
import site
paths = site.getsitepackages()
print(paths[0] if paths else "")
PY
)"
if [[ -n "${STARVLA_SITE}" && -d "${STARVLA_SITE}" ]]; then
    export PYTHONPATH="${STARVLA_SITE}:${PYTHONPATH:-}"
fi

FREE_GPUS="$(select_free_gpus)"
export CUDA_VISIBLE_DEVICES="${FREE_GPUS}"

RUN_FOLDER="$(python3 - "${CKPT_PATH}" <<'PY'
import os, sys
ckpt = os.path.realpath(sys.argv[1])
parts = ckpt.strip('/').split('/')
print(f"{parts[-3]}_{parts[-2]}_{parts[-1]}")
PY
)"
RESULTS_ROOT="${ROBOTWIN_RESULTS_ROOT:-${REPO_ROOT}/results/robotwin/${RUN_FOLDER}}"
LOG_ROOT="${ROBOTWIN_LOG_ROOT:-${REPO_ROOT}/.eval4_logs/robotwin/${RUN_FOLDER}}"
mkdir -p "${RESULTS_ROOT}" "${LOG_ROOT}"
export ROBOTWIN_RESULTS_ROOT="${RESULTS_ROOT}"
export ROBOTWIN_LOG_ROOT="${LOG_ROOT}"

LAUNCHER_LOG="${LOG_ROOT}/launcher.log"
PID_FILE="${LOG_ROOT}/launcher.pid"

echo "repo_root=${REPO_ROOT}"
echo "data_root=${DATA_ROOT}"
echo "robotwin_path=${ROBOTWIN_PATH}"
echo "ckpt=${CKPT_PATH}"
echo "candidate_gpus=${CUDA_DEVICES}"
echo "free_gpus=${FREE_GPUS}"
echo "log_root=${LOG_ROOT}"

nohup bash "${SCRIPT_DIR}/start_eval.sh" \
    --mode demo_clean \
    --name "${POLICY_NAME}" \
    --ckpt "${CKPT_PATH}" \
    --jobs-per-gpu 1 \
    --base-port "${ROBOTWIN_BASE_PORT:-5694}" \
    --server-timeout "${ROBOTWIN_SERVER_TIMEOUT:-1800}" \
    all \
    > "${LAUNCHER_LOG}" 2>&1 &

echo "$!" > "${PID_FILE}"
sleep 5

if ! kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
    echo "RoboTwin launcher exited early. See ${LAUNCHER_LOG}" >&2
    tail -n 80 "${LAUNCHER_LOG}" >&2 || true
    exit 1
fi

echo "launcher_pid=$(cat "${PID_FILE}")"
echo "launcher_log=${LAUNCHER_LOG}"
