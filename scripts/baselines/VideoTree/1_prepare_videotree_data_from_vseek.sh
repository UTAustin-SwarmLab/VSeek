#!/bin/bash
set -euo pipefail

# Output root for prepared JSONs
OUTPUT_ROOT="./prepared"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${OUTPUT_ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${TIMESTAMP}_1_prepare_videotree_data_from_vseek.log"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "Logging to ${LOG_FILE}"

# Paths aligned with src/vseek/config/retriever/config.yaml

LVB_DATASET_PATH="/nas/mars/dataset/longvideobench/LongVideoBench/"
LVB_BURNED_PATH="/nas/mars/dataset/longvideobench/"

LVBENCH_DATASET_PATH="/nas/mars/dataset/LVBench"
LVBENCH_BURNED_PATH="/nas/mars/dataset/LVBench/videos"

VIDEOMME_DATASET_PATH="/nas/mars/dataset/Video-MME"
VIDEOMME_BURNED_PATH="/nas/mars/dataset/Video-MME/burn-subtitles"

MLVU_DATASET_PATH="/nas/mars/dataset/MLVU/MLVU"
MLVU_BURNED_PATH="/nas/mars/dataset/MLVU/MLVU"

# subtitle (recommended) or question_only
NARRATION_SOURCE="subtitle"

# -1 means all examples
MAX_EXAMPLES="-1"

prepare_one() {
  local dataset_name="$1"
  local dataset_path="$2"
  local burned_path="$3"
  local out_dir="${OUTPUT_ROOT}/${dataset_name}"

  echo "Preparing ${dataset_name} -> ${out_dir}"
  python prepare_videotree_data_from_vseek.py \
    --dataset_name "${dataset_name}" \
    --dataset_path "${dataset_path}" \
    --burned_path "${burned_path}" \
    --output_dir "${out_dir}" \
    --narration_source "${NARRATION_SOURCE}" \
    --max_examples "${MAX_EXAMPLES}"
}

prepare_one "lvb" "${LVB_DATASET_PATH}" "${LVB_BURNED_PATH}"
prepare_one "lvbench" "${LVBENCH_DATASET_PATH}" "${LVBENCH_BURNED_PATH}"
prepare_one "videomme" "${VIDEOMME_DATASET_PATH}" "${VIDEOMME_BURNED_PATH}"
prepare_one "mlvu" "${MLVU_DATASET_PATH}" "${MLVU_BURNED_PATH}"

echo ""
echo "Done. Prepared files are under: ${OUTPUT_ROOT}/<dataset>/"
echo "Each dataset folder contains: data.json, anno.json, duration.json"
