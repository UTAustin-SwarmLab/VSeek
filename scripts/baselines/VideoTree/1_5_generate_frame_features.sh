#!/bin/bash
set -euo pipefail

SCRIPT_LABEL="VideoTree frame feature generation"

notify_both() {
  local message="$1"
  notify.py "${message}" || true
  slack_notify.py "${message}" || true
}

on_exit() {
  local exit_code=$?
  if (( exit_code == 0 )); then
    notify_both "${SCRIPT_LABEL} completed"
  else
    notify_both "${SCRIPT_LABEL} failed (exit=${exit_code})"
  fi
}

trap on_exit EXIT

# Uses JSONs prepared by:
#   1_prepare_videotree_data_from_vseek.sh

OUTPUT_ROOT="./prepared"
FEATURE_ROOT="${OUTPUT_ROOT}/frame_features"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${OUTPUT_ROOT}/logs"
mkdir -p "${LOG_DIR}" "${FEATURE_ROOT}"
LOG_FILE="${LOG_DIR}/${TIMESTAMP}_1_5_generate_frame_features.log"
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

# Feature extraction config
MODEL_NAME_OR_PATH="openai/clip-vit-base-patch32"
DEVICE="cuda"
SAMPLE_FPS="1.0"
BATCH_SIZE="512"
MAX_EXAMPLES="-1"  # -1 means all
OVERWRITE="false"  # true to recompute existing .pt files

# Parallel execution config
# - MAX_PARALLEL_JOBS controls how many datasets run concurrently
# - GPU_IDS are assigned round-robin to launched jobs using CUDA_VISIBLE_DEVICES
MAX_PARALLEL_JOBS=4
GPU_IDS=("5")

extract_one() {
  local dataset_name="$1"
  local dataset_path="$2"
  local burned_path="$3"
  local gpu_id="$4"
  local prepared_dir="${OUTPUT_ROOT}/${dataset_name}"
  local out_dir="${FEATURE_ROOT}/${dataset_name}"
  local dataset_log="${LOG_DIR}/${TIMESTAMP}_1_5_generate_frame_features_${dataset_name}.log"

  mkdir -p "${out_dir}"
  echo "Generating frame features for ${dataset_name} -> ${out_dir} (GPU ${gpu_id})"
  echo "Per-dataset log: ${dataset_log}"

  if [[ "${OVERWRITE}" == "true" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu_id}" python generate_frame_features_from_vseek.py \
      --dataset_name "${dataset_name}" \
      --dataset_path "${dataset_path}" \
      --burned_path "${burned_path}" \
      --prepared_dir "${prepared_dir}" \
      --output_dir "${out_dir}" \
      --model_name_or_path "${MODEL_NAME_OR_PATH}" \
      --device "${DEVICE}" \
      --sample_fps "${SAMPLE_FPS}" \
      --batch_size "${BATCH_SIZE}" \
      --max_examples "${MAX_EXAMPLES}" \
      --overwrite \
      > "${dataset_log}" 2>&1
  else
    CUDA_VISIBLE_DEVICES="${gpu_id}" python generate_frame_features_from_vseek.py \
      --dataset_name "${dataset_name}" \
      --dataset_path "${dataset_path}" \
      --burned_path "${burned_path}" \
      --prepared_dir "${prepared_dir}" \
      --output_dir "${out_dir}" \
      --model_name_or_path "${MODEL_NAME_OR_PATH}" \
      --device "${DEVICE}" \
      --sample_fps "${SAMPLE_FPS}" \
      --batch_size "${BATCH_SIZE}" \
      --max_examples "${MAX_EXAMPLES}" \
      > "${dataset_log}" 2>&1
  fi
}

launch_dataset() {
  local dataset_name="$1"
  local gpu_id="$2"
  case "${dataset_name}" in
    lvb)
      extract_one "lvb" "${LVB_DATASET_PATH}" "${LVB_BURNED_PATH}" "${gpu_id}" &
      ;;
    lvbench)
      extract_one "lvbench" "${LVBENCH_DATASET_PATH}" "${LVBENCH_BURNED_PATH}" "${gpu_id}" &
      ;;
    videomme)
      extract_one "videomme" "${VIDEOMME_DATASET_PATH}" "${VIDEOMME_BURNED_PATH}" "${gpu_id}" &
      ;;
    mlvu)
      extract_one "mlvu" "${MLVU_DATASET_PATH}" "${MLVU_BURNED_PATH}" "${gpu_id}" &
      ;;
    *)
      echo "Unknown dataset: ${dataset_name}"
      exit 1
      ;;
  esac
}

DATASETS=("lvb" "lvbench" "videomme" "mlvu")
PIDS=()
FAIL=0
launch_idx=0

for dataset_name in "${DATASETS[@]}"; do
  gpu_id="${GPU_IDS[$((launch_idx % ${#GPU_IDS[@]}))]}"
  launch_dataset "${dataset_name}" "${gpu_id}"
  pid=$!
  echo "Launched ${dataset_name} as PID ${pid} on GPU ${gpu_id}"
  PIDS+=("${pid}")
  launch_idx=$((launch_idx + 1))

  if (( ${#PIDS[@]} >= MAX_PARALLEL_JOBS )); then
    wait "${PIDS[0]}" || FAIL=1
    PIDS=("${PIDS[@]:1}")
  fi
done

for pid in "${PIDS[@]}"; do
  wait "${pid}" || FAIL=1
done

if (( FAIL != 0 )); then
  echo "One or more dataset jobs failed. Check per-dataset logs in ${LOG_DIR}."
  exit 1
fi

echo ""
echo "Done. Frame features are under: ${FEATURE_ROOT}/<dataset>/<uid>.pt"
