#!/bin/bash
# Launch the UniversalVTG temporal grounding server for VSeek.
#
# Prerequisites:
#   1. Pre-extract PE features:
#      CUDA_VISIBLE_DEVICES=1 python scripts/data_ops/extract_pe_features_lvb.py
#
#   2. Ensure the UniversalVTG checkpoint exists:
#      <universalvtg_path>/experiments/universalvtg/models/best.pth
#
# Usage:
#   bash scripts/run_vtg_server.sh [port] [gpu]
#
# Example:
#   bash scripts/run_vtg_server.sh 9002 0

PORT=${1:-9002}
GPU=${2:-0}
shift 2 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"
export HF_HUB_ENABLE_HF_TRANSFER=0

python "${PROJECT_ROOT}/src/vseek/tools/server_vtg.py" \
    "retriever.port=$PORT" \
    "retriever.gpu_number=0" \
    "$@"
