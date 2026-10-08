#!/bin/bash
# vLLM rollout eval of VSeek-EM and VSeek-PULS on test.parquet only.
# Temperatures 0.0..0.6 (step 0.1), 4 samples per question, one dataset per GPU.
set -u

ROOT="/home/hg22723/projects/VSeek-R1"
source /home/hg22723/anaconda3/etc/profile.d/conda.sh
conda activate vseek-vllm
cd "$ROOT"

export TOKENIZERS_PARALLELISM=true
export VLLM_USE_V1=1
export NCCL_P2P_DISABLE=1
unset PYTORCH_CUDA_ALLOC_CONF

TOOL_CONFIG="$ROOT/src/vseek/config/retriever/toolconfig_temp_sweep.yaml"
OUT_ROOT="$ROOT/results/temp_sweep"
LOG_ROOT="$OUT_ROOT/logs"
mkdir -p "$LOG_ROOT"

EM_MODEL="/home/hg22723/projects/VSeek-R2/checkpoints/vseek-em/huggingface"
PULS_MODEL="/home/hg22723/projects/VSeek-R2/checkpoints/vseek-puls/huggingface"
TEMPS=(0.0 0.1 0.2 0.3 0.4 0.5 0.6)
PASSES=4
BATCH_SIZE=8

# gpu|tag|parquet|max_prompt_length
JOBS=(
  "2|videomme|/nas/mars/dataset/Video-MME/window_8/tagsummary/test.parquet|2048"
  "3|lvb|/nas/mars/dataset/longvideobench/window_8/tagsummary/test.parquet|2536"
  "4|mlvu|/nas/mars/dataset/MLVU/window_8/tagsummary/test.parquet|2048"
  "5|cgbench|/nas/mars/dataset/CGBench/window_8/tagsummary/test.parquet|2048"
  "6|lvbench|/nas/mars/dataset/LVBench/window_8/tagsummary/test.parquet|2536"
)

run_dataset() {
  local gpu="$1"
  local tag="$2"
  local parquet="$3"
  local max_prompt="$4"
  local log="$LOG_ROOT/gpu${gpu}_${tag}.log"

  if [[ "$parquet" != */test.parquet ]]; then
    echo "Refusing non-test parquet: $parquet" | tee -a "$log"
    exit 1
  fi

  {
    echo "GPU $gpu dataset $tag parquet $parquet"
    for model_name in vseek-em vseek-puls; do
      if [[ "$model_name" == "vseek-em" ]]; then
        model_path="$EM_MODEL"
      else
        model_path="$PULS_MODEL"
      fi
      for temp in "${TEMPS[@]}"; do
        temp_tag="${temp/./p}"
        out_dir="$OUT_ROOT/${model_name}/t${temp_tag}"
        echo "===== ${model_name} ${tag} temperature ${temp} ====="
        CUDA_VISIBLE_DEVICES="$gpu" \
        RAY_TMPDIR="/tmp/ray_vseek_temp_sweep_gpu${gpu}" \
          python3 scripts/run_agent_data_vllmrollout.py \
            --parquet "$parquet" \
            --batch_size "$BATCH_SIZE" \
            --output_dir "$out_dir" \
            --output_prefix "$tag" \
            --topk 4 \
            --local_model_path "$model_path" \
            --prompt_type tagsummary \
            --agent_type tagsummary \
            --passes "$PASSES" \
            --temperature "$temp" \
            --max_prompt_length "$max_prompt" \
            --store_preds false \
            --tool_config_path "$TOOL_CONFIG"
        status=$?
        if [[ "$status" -ne 0 ]]; then
          echo "FAILED ${model_name} ${tag} temperature ${temp} exit ${status}"
          exit "$status"
        fi
        echo "===== finished ${model_name} ${tag} temperature ${temp} ====="
      done
    done
    echo "DONE gpu $gpu dataset $tag"
  } > "$log" 2>&1
}

echo "Waiting for retriever on port 9105..."
for _ in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:9105/health" >/dev/null; then
    echo "Retriever is up"
    break
  fi
  sleep 10
done
if ! curl -sf "http://127.0.0.1:9105/health" >/dev/null; then
  echo "Retriever on port 9105 did not become healthy" >&2
  exit 1
fi

for entry in "${JOBS[@]}"; do
  IFS="|" read -r gpu tag parquet max_prompt <<< "$entry"
  run_dataset "$gpu" "$tag" "$parquet" "$max_prompt" &
  echo "Launched $tag on GPU $gpu pid $!"
done

wait
echo "All temperature-sweep jobs finished"
