#!/bin/bash

# MODEL="OpenGVLab/InternVL2_5-8B"
# MODEL="meta-llama/Llama-3.2-11B-Vision-Instruct"
MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES="6"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PORT=8001
uv run vllm serve $MODEL \
    --tensor-parallel-size 1 \
    --port $PORT \
    --trust-remote-code

# Optional parameters (uncomment as needed):
# --limit-mm-per-prompt image=3
# --gpu-memory-utilization 0.85
# --max-model-len 8192
# --disable-log-requests
