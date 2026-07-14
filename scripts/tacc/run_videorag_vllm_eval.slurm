#!/bin/bash
#SBATCH -J videorag_vllm_eval
#SBATCH -o videorag_vllm_eval-%j.out
#SBATCH -e videorag_vllm_eval-%j.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -t 24:00:00
#SBATCH --mem=0

set -euo pipefail

echo "Starting job on $(hostname)"
date

source ~/.bashrc
module load gcc/14.2.0 cuda/12.8
export CC="$(which gcc)"
export CXX="$(which g++)"
export CUDA_HOME="$TACC_CUDA_DIR"
export PATH="${CUDA_HOME}/bin:${PATH}"

conda activate vseek-vllm4
echo "Conda environment: $CONDA_DEFAULT_ENV"

export HF_HOME="${SCRATCH}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
mkdir -p "$HF_HOME"
echo "HuggingFace cache: $HF_HOME"

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/home1/11123/harshgoel99/VSeek-R1}"
cd "$PROJECT_ROOT" || exit 1
echo "Working directory: $(pwd)"

# ---------------------------
# Eval + vLLM configuration
# ---------------------------
MODEL="${MODEL:-Qwen/Qwen3-VL-4B-Thinking}"
DEVICE="${DEVICE:-0}"
FRAMES="${FRAMES:-64}"
PROMPT_TYPE="${PROMPT_TYPE:-cot}"
DATASETS="${DATASETS:-lvb mlvu videomme}"
OUTPUT_DIR="${OUTPUT_DIR:-results/videorag_vllm}"
TEMPERATURE="${TEMPERATURE:-0.7}"
VLLM_PORT="${VLLM_PORT:-8001}"

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export NCCL_P2P_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_USE_V1=1
export CUDA_VISIBLE_DEVICES="$DEVICE"

VLLM_PID=""
cleanup() {
    if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "Stopping vLLM server (pid=$VLLM_PID)"
        kill "$VLLM_PID" || true
    fi
}
trap cleanup EXIT

echo "Launching vLLM server on port $VLLM_PORT (GPU $DEVICE)"
vllm serve "$MODEL" \
    --tensor-parallel-size 1 \
    --port "$VLLM_PORT" \
    --trust-remote-code \
    > "vllm_videorag_${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!

echo "Waiting for vLLM to become healthy..."
for i in {1..120}; do
    if python3 - <<PY
import requests
try:
    r = requests.get("http://127.0.0.1:${VLLM_PORT}/v1/models", timeout=2)
    raise SystemExit(0 if r.status_code == 200 else 1)
except Exception:
    raise SystemExit(1)
PY
    then
        echo "vLLM is ready."
        break
    fi
    sleep 2
    if [ "$i" -eq 120 ]; then
        echo "vLLM failed to start in time."
        exit 1
    fi
done

echo "Running videorag eval..."
bash scripts/evals/run_videorag_vllm.sh \
    --device "$DEVICE" \
    --model "$MODEL" \
    --frames "$FRAMES" \
    --output-dir "$OUTPUT_DIR" \
    --prompt-type "$PROMPT_TYPE" \
    --vllm-port "$VLLM_PORT" \
    --temperature "$TEMPERATURE" \
    --datasets $DATASETS

echo "VideoRAG vLLM eval completed."
