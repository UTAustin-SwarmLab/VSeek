#!/bin/bash
# Wrapper script to setup environment on compute nodes before executing commands
# Usage: ./ray_wrapper.sh [COMMAND] [ARGS...]

# 1. Setup Environment
source ~/.bashrc
# Fix: Ensure standard CUDA compiler is used, not NVHPC
module load gcc/14.2.0 cuda/12.8
export CC=$(which gcc)
export CXX=$(which g++)
export CUDA_HOME="$TACC_CUDA_DIR"
export PATH="${CUDA_HOME}/bin:${PATH}"
conda activate vseek-vllm4

# 2. Source Captured Environment (Critical for TACC SSH)
if [ -f ~/worker_env.sh ]; then
    source ~/worker_env.sh
fi

conda activate vseek-vllm4

# 3. Export Common Variables (Fallback)
export HF_HOME="${WORK}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"

# 4. Disable torch.compile for vLLM compatibility (V1 engine required by verl)
# export TORCH_COMPILE_DISABLE=1
# export TORCH_DYNAMO_DISABLE=1
export VLLM_USE_V1=1

# NCCL network configuration - exclude loopback, let NCCL find best interface
export NCCL_SOCKET_IFNAME=^lo,docker  # Exclude loopback and docker, use everything else
export NCCL_IB_DISABLE=0              # Enable InfiniBand/RDMA
export NCCL_NET_GDR_LEVEL=5           # Enable GPU Direct RDMA
export NCCL_DEBUG=WARN                # Show warnings to help diagnose  

# 4. Debugging: Verify GPU visibility
echo "[Wrapper $(hostname)] GPU Check:"
nvidia-smi 2>/dev/null || echo "nvidia-smi not found or failed"
python3 -c 'import torch; print(f"CUDA Available: {torch.cuda.is_available()}")'

# 5. Execute the passed command
exec "$@"
