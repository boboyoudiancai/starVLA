#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_GIT_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
source "${REPO_GIT_ROOT}/.starvla.env"
cd "${REPO_ROOT}"
python3 starVLA/path_tool.py setup-links >/dev/null

# Usage:
#   bash examples/VLA-Arena/data_preparation.sh
# or
#   export DEST=/path/to/dir && bash examples/VLA-Arena/data_preparation.sh
# or
#   bash examples/VLA-Arena/data_preparation.sh /path/to/dir
#
# Downloads the three VLA-Arena L0 datasets (Small / Medium / Large) from
# HuggingFace in LeRobot (openpi) format and wires them up for StarVLA training.
#
# HuggingFace repos:
#   VLA-Arena/VLA_Arena_L0_S_lerobot_openpi
#   VLA-Arena/VLA_Arena_L0_M_lerobot_openpi
#   VLA-Arena/VLA_Arena_L0_L_lerobot_openpi
#
# After this script:
#   playground/Datasets/vla_arena/ -> $DEST/vla_arena/
#
# NOTE: The modality.json maps dataset keys to StarVLA keys.
#   If the primary camera key in your dataset differs from
#   "observation.images.agentview_rgb", update train_files/modality.json
#   (video.primary_image.original_key) before training.

DEST="${DEST:-${1:-playground/Datasets}}"

mkdir -p "$DEST/vla_arena"

python -m pip install -U "huggingface-hub==0.35.3"

for repo in \
  VLA-Arena/VLA_Arena_L0_L_lerobot_openpi \
  # VLA-Arena/VLA_Arena_L0_M_lerobot_openpi \
  # VLA-Arena/VLA_Arena_L0_S_lerobot_openpi
do
  hf download "$repo" --repo-type dataset --local-dir "$DEST/vla_arena/${repo##*/}"
done

## copy modality.json into each dataset's meta/ directory
for dataset in \
  VLA_Arena_L0_L_lerobot_openpi \
  # VLA_Arena_L0_M_lerobot_openpi \
  # VLA_Arena_L0_S_lerobot_openpi
do
  cp "examples/VLA-Arena/train_files/modality.json" \
     "$DEST/vla_arena/${dataset}/meta/modality.json"
done

echo ""
echo "Done. Dataset layout:"
echo "  playground/Datasets -> ${DATA_ROOT}/data"
echo "  playground/Datasets/vla_arena -> $DEST/vla_arena"
echo ""
echo "Available data_mix values for training:"
echo "  vla_arena_L0_L   - large split"
