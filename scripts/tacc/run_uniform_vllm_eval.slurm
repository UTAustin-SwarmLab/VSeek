#!/bin/bash
#SBATCH -J uniform_vllm_eval
#SBATCH -o uniform_vllm_eval-%j.out
#SBATCH -e uniform_vllm_eval-%j.err
#SBATCH -p gh
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -t 48:00:00
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
DATASETS="${DATASETS:-lvb videomme mlvu}"
OUTPUT_DIR="${OUTPUT_DIR:-results/uniform_vllm}"
TEMPERATURE="${TEMPERATURE:-0.7}"
VLLM_PORT="${VLLM_PORT:-8002}"

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export NCCL_P2P_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_USE_V1=1
export CUDA_VISIBLE_DEVICES="$DEVICE"


echo "Running uniform eval..."
bash scripts/evals/run_uniform_vllm.sh \
    --device "$DEVICE" \
    --model "$MODEL" \
    --frames "$FRAMES" \
    --output-dir "$OUTPUT_DIR" \
    --prompt-type "$PROMPT_TYPE" \
    --vllm-port "$VLLM_PORT" \
    --temperature "$TEMPERATURE" \
    --datasets $DATASETS

echo "Uniform vLLM eval completed."
