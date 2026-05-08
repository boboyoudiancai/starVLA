#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_GIT_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
source "${REPO_GIT_ROOT}/.starvla.env"
cd "${REPO_ROOT}"
python3 starVLA/path_tool.py setup-links >/dev/null

# Usage:
#   bash examples/LIBERO/data_preparation.sh
# or
#   export DEST=/path/to/dir && bash examples/LIBERO/data_preparation.sh
# or
#   bash examples/LIBERO/data_preparation.sh /path/to/dir

DEST="${DEST:-${1:-playground/Datasets}}"

mkdir -p "$DEST"

TOKEN="${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}"
DEFAULT_HF_MAX_WORKERS=1
DEFAULT_HF_RATE_LIMIT_SLEEP=310
DEFAULT_HF_MAX_RETRIES=20
DEFAULT_HF_SERVER_ERROR_SLEEP=120
HF_MAX_WORKERS="${HF_MAX_WORKERS:-${DEFAULT_HF_MAX_WORKERS}}"
HF_RATE_LIMIT_SLEEP="${HF_RATE_LIMIT_SLEEP:-${DEFAULT_HF_RATE_LIMIT_SLEEP}}"
HF_MAX_RETRIES="${HF_MAX_RETRIES:-${DEFAULT_HF_MAX_RETRIES}}"
HF_SERVER_ERROR_SLEEP="${HF_SERVER_ERROR_SLEEP:-${DEFAULT_HF_SERVER_ERROR_SLEEP}}"

is_uint() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

sanitize_uint_env() {
  local name="$1"
  local default_value="$2"
  local value="${!name}"
  if ! is_uint "${value}"; then
    echo "WARNING: ${name} must be an integer, got: ${value}. Falling back to ${default_value}."
    printf -v "${name}" '%s' "${default_value}"
  fi
}

sanitize_uint_env HF_MAX_WORKERS "${DEFAULT_HF_MAX_WORKERS}"
sanitize_uint_env HF_RATE_LIMIT_SLEEP "${DEFAULT_HF_RATE_LIMIT_SLEEP}"
sanitize_uint_env HF_MAX_RETRIES "${DEFAULT_HF_MAX_RETRIES}"
sanitize_uint_env HF_SERVER_ERROR_SLEEP "${DEFAULT_HF_SERVER_ERROR_SLEEP}"

export HF_HUB_DISABLE_TELEMETRY=1

python -m pip install -U "huggingface_hub==0.36.2" "hf-xet>=1.1.3" "socksio"

if [[ -z "${TOKEN}" ]]; then
  echo "WARNING: HF_TOKEN/HUGGINGFACE_HUB_TOKEN is not set."
  echo "The Hub applies stricter per-IP limits without an authenticated token."
fi

hf_download_with_retry() {
  local repo_id="$1"
  local repo_type="$2"
  local local_dir="$3"
  local attempt=1
  local log_file
  log_file="$(mktemp)"

  while (( attempt <= HF_MAX_RETRIES )); do
    echo "[HF] Downloading ${repo_id} -> ${local_dir} (attempt ${attempt}/${HF_MAX_RETRIES})"

    set +e
    if [[ -n "${TOKEN}" ]]; then
      hf download "${repo_id}" \
        --repo-type "${repo_type}" \
        --local-dir "${local_dir}" \
        --token "${TOKEN}" \
        --max-workers "${HF_MAX_WORKERS}" \
        2>&1 | tee "${log_file}"
    else
      hf download "${repo_id}" \
        --repo-type "${repo_type}" \
        --local-dir "${local_dir}" \
        --max-workers "${HF_MAX_WORKERS}" \
        2>&1 | tee "${log_file}"
    fi
    status=${PIPESTATUS[0]}
    set -e

    if [[ ${status} -eq 0 ]]; then
      rm -f "${log_file}"
      return 0
    fi

    if grep -Eqi '429|rate limit|quota of [0-9]+ api requests per 5 minutes' "${log_file}"; then
      echo "[HF] Rate limited. Sleeping ${HF_RATE_LIMIT_SLEEP}s before retry. HF_MAX_RETRIES=${HF_MAX_RETRIES}"
      sleep "${HF_RATE_LIMIT_SLEEP}"
      ((attempt++))
      continue
    fi

    if grep -Eqi '500 Server Error|502 Server Error|503 Server Error|504 Server Error|Internal Error' "${log_file}"; then
      echo "[HF] HF server error. Sleeping ${HF_SERVER_ERROR_SLEEP}s before retry. HF_MAX_RETRIES=${HF_MAX_RETRIES}"
      sleep "${HF_SERVER_ERROR_SLEEP}"
      ((attempt++))
      continue
    fi

    echo "[HF] Download failed with a non-rate-limit error."
    cat "${log_file}"
    rm -f "${log_file}"
    return "${status}"
  done

  echo "[HF] Exhausted retries for ${repo_id}."
  cat "${log_file}"
  rm -f "${log_file}"
  return 1
}

for repo in \
  IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot \
  IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot \
  IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot \
  IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot
do
  hf_download_with_retry "$repo" dataset "$DEST/libero/${repo##*/}"
done

hf_download_with_retry "StarVLA/LLaVA-OneVision-COCO" dataset "$DEST/LLaVA-OneVision-COCO"
unzip -o -- "$DEST/LLaVA-OneVision-COCO/sharegpt4v_coco.zip" -d "$DEST/LLaVA-OneVision-COCO/"

## move modality
mkdir -p "$DEST/libero/libero_10_no_noops_1.0.0_lerobot/meta"
mkdir -p "$DEST/libero/libero_goal_no_noops_1.0.0_lerobot/meta"
mkdir -p "$DEST/libero/libero_object_no_noops_1.0.0_lerobot/meta"
mkdir -p "$DEST/libero/libero_spatial_no_noops_1.0.0_lerobot/meta"
cp "examples/LIBERO/train_files/modality.json" "$DEST/libero/libero_10_no_noops_1.0.0_lerobot/meta"
cp "examples/LIBERO/train_files/modality.json" "$DEST/libero/libero_goal_no_noops_1.0.0_lerobot/meta"
cp "examples/LIBERO/train_files/modality.json" "$DEST/libero/libero_object_no_noops_1.0.0_lerobot/meta"
cp "examples/LIBERO/train_files/modality.json" "$DEST/libero/libero_spatial_no_noops_1.0.0_lerobot/meta"

echo ""
echo "Done. Dataset layout:"
echo "  playground/Datasets -> ${DATA_ROOT}/data"
echo "  playground/Datasets/libero -> ${DEST}/libero"
echo "  playground/Datasets/LLaVA-OneVision-COCO -> ${DEST}/LLaVA-OneVision-COCO"
