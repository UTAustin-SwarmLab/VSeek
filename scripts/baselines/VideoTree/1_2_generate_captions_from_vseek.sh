#!/bin/bash
set -euo pipefail

# Uses prepared anno.json files from:
#   1_prepare_videotree_data_from_vseek.sh

OUTPUT_ROOT="./prepared"
CAPTION_ROOT="${OUTPUT_ROOT}/captions"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${OUTPUT_ROOT}/logs"
mkdir -p "${LOG_DIR}" "${CAPTION_ROOT}"
LOG_FILE="${LOG_DIR}/${TIMESTAMP}_1_2_generate_captions_from_vseek.log"
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

# vLLM / OpenAI-compatible API config
MODEL_NAME="Qwen/Qwen3-VL-4B-Instruct"
BASE_URL="http://127.0.0.1:8000/v1"
BASE_URLS="http://127.0.0.1:8000/v1,http://127.0.0.1:8001/v1,http://127.0.0.1:8002/v1,http://127.0.0.1:8005/v1,http://127.0.0.1:8006/v1,http://127.0.0.1:8007/v1"
API_KEY="EMPTY"
REQUEST_TIMEOUT="5"
MAX_RETRIES="2"
MAX_TOKENS="256"

# Caption sampling config
SAMPLE_FPS="1.0"
MAX_FRAMES_PER_VIDEO="-1"
MAX_EXAMPLES="-1"  # -1 means all
OVERWRITE="false"  # true to regenerate existing video_id captions
VIDEO_WORKERS="4"  # 0 auto-selects workers from number of endpoints
DATASET_MAX_PARALLEL="4"

caption_one() {
  local dataset_name="$1"
  local dataset_path="$2"
  local burned_path="$3"
  local prepared_dir="${OUTPUT_ROOT}/${dataset_name}"
  local out_dir="${CAPTION_ROOT}/${dataset_name}"
  local out_json="${out_dir}/captions_vllm.json"
  local dataset_log="${LOG_DIR}/${TIMESTAMP}_1_2_generate_captions_${dataset_name}.log"

  mkdir -p "${out_dir}"
  echo "Generating captions for ${dataset_name} -> ${out_json}"
  echo "Per-dataset log: ${dataset_log}"

  if [[ "${OVERWRITE}" == "true" ]]; then
    python data_extraction/extract_captions.py \
      --dataset_name "${dataset_name}" \
      --dataset_path "${dataset_path}" \
      --burned_path "${burned_path}" \
      --prepared_dir "${prepared_dir}" \
      --output_json "${out_json}" \
      --model "${MODEL_NAME}" \
      --base_url "${BASE_URL}" \
      --base_urls "${BASE_URLS}" \
      --api_key "${API_KEY}" \
      --request_timeout "${REQUEST_TIMEOUT}" \
      --max_retries "${MAX_RETRIES}" \
      --max_tokens "${MAX_TOKENS}" \
      --num_workers "${VIDEO_WORKERS}" \
      --sample_fps "${SAMPLE_FPS}" \
      --max_frames_per_video "${MAX_FRAMES_PER_VIDEO}" \
      --max_examples "${MAX_EXAMPLES}" \
      --overwrite \
      > "${dataset_log}" 2>&1
  else
    python data_extraction/extract_captions.py \
      --dataset_name "${dataset_name}" \
      --dataset_path "${dataset_path}" \
      --burned_path "${burned_path}" \
      --prepared_dir "${prepared_dir}" \
      --output_json "${out_json}" \
      --model "${MODEL_NAME}" \
      --base_url "${BASE_URL}" \
      --base_urls "${BASE_URLS}" \
      --api_key "${API_KEY}" \
      --request_timeout "${REQUEST_TIMEOUT}" \
      --max_retries "${MAX_RETRIES}" \
      --max_tokens "${MAX_TOKENS}" \
      --num_workers "${VIDEO_WORKERS}" \
      --sample_fps "${SAMPLE_FPS}" \
      --max_frames_per_video "${MAX_FRAMES_PER_VIDEO}" \
      --max_examples "${MAX_EXAMPLES}" \
      > "${dataset_log}" 2>&1
  fi
}

declare -a DATASET_NAMES=("lvb" "lvbench" "videomme" "mlvu")
declare -a DATASET_PATHS=("${LVB_DATASET_PATH}" "${LVBENCH_DATASET_PATH}" "${VIDEOMME_DATASET_PATH}" "${MLVU_DATASET_PATH}")
declare -a BURNED_PATHS=("${LVB_BURNED_PATH}" "${LVBENCH_BURNED_PATH}" "${VIDEOMME_BURNED_PATH}" "${MLVU_BURNED_PATH}")

any_failed=0
batch_size=0
declare -a JOB_PIDS=()
declare -a JOB_NAMES=()

for i in "${!DATASET_NAMES[@]}"; do
  name="${DATASET_NAMES[$i]}"
  dpath="${DATASET_PATHS[$i]}"
  bpath="${BURNED_PATHS[$i]}"

  caption_one "${name}" "${dpath}" "${bpath}" &
  JOB_PIDS+=("$!")
  JOB_NAMES+=("${name}")
  batch_size=$((batch_size + 1))

  is_last=0
  if (( i == ${#DATASET_NAMES[@]} - 1 )); then
    is_last=1
  fi

  if (( batch_size >= DATASET_MAX_PARALLEL || is_last == 1 )); then
    for j in "${!JOB_PIDS[@]}"; do
      pid="${JOB_PIDS[$j]}"
      jname="${JOB_NAMES[$j]}"
      if ! wait "${pid}"; then
        echo "[ERROR] Caption generation failed for dataset: ${jname}"
        any_failed=1
      fi
    done
    JOB_PIDS=()
    JOB_NAMES=()
    batch_size=0
  fi
done

if [[ "${any_failed}" == "1" ]]; then
  echo "One or more datasets failed. Check per-dataset logs in ${LOG_DIR}."
  exit 1
fi

echo ""
echo "Done. Captions are under: ${CAPTION_ROOT}/<dataset>/captions_vllm.json"
