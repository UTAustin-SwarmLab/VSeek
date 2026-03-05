#!/bin/bash

# MODEL="OpenGVLab/InternVL2_5-8B"
# MODEL="meta-llama/Llama-3.2-11B-Vision-Instruct"
# MODEL="OpenGVLab/InternVL3_5-4B"
MODEL="Qwen/Qwen3-VL-4B-Instruct"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export NCCL_P2P_DISABLE=1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PORT=${1:-8010}
CUDA_VISIBLE_DEVICES=${2:-4}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
vllm serve $MODEL \
    --tensor-parallel-size 1 \
    --port $PORT \
    --trust-remote-code
    
# Optional parameters (uncomment as needed):
# --limit-mm-per-prompt image=3
# --gpu-memory-utilization 0.85
# --max-model-len 8192
# --disable-log-requests
