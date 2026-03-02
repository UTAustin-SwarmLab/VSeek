#!/bin/bash
# Wrapper script to setup environment on compute nodes before executing commands
# Usage: ./ray_wrapper.sh [COMMAND] [ARGS...]

# 1. Setup Environment
source ~/.bashrc

# Activate conda FIRST so $CONDA_PREFIX is correct
conda activate vseek-vllm2
echo "[Wrapper $(hostname)] Activated conda: $CONDA_DEFAULT_ENV"

# Fix: Ensure standard CUDA compiler is used, not NVHPC
module load gcc/15.1.0 cuda/12.9
export CUDA_HOME="$TACC_CUDA_DIR"
export CC=$(which gcc)
export CXX=$(which g++)

# Set LD_LIBRARY_PATH AFTER conda activation (CRITICAL for NCCL)
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
echo "[Wrapper $(hostname)] LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# 2. Source Captured Environment (Critical for TACC SSH)
if [ -f ~/worker_env.sh ]; then
    echo "[Wrapper $(hostname)] Sourcing worker_env.sh"
    source ~/worker_env.sh
fi

# 3. Export Common Variables (Fallback)
export HF_HOME="${SCRATCH}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"

# 4. Disable torch.compile for vLLM compatibility (V1 engine required by verl)
export VLLM_USE_V1=1

# Fix harmony encoding download issues
export HARMONY_CACHE_DIR="${HF_HOME}/harmony_cache"
export VLLM_DISABLE_HARMONY=1

# 5. CRITICAL: Verify PyTorch imports correctly before starting Ray
echo "[Wrapper $(hostname)] Testing PyTorch import..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')" || {
    echo "[Wrapper $(hostname)] ERROR: PyTorch import failed!"
    echo "[Wrapper $(hostname)] Debugging NCCL..."
    ldd $CONDA_PREFIX/lib/python3.11/site-packages/torch/lib/libtorch_cuda.so 2>&1 | grep nccl || echo "Could not check NCCL"
    exit 1
}

# 6. GPU Check
echo "[Wrapper $(hostname)] GPU Check:"
nvidia-smi 2>/dev/null || echo "nvidia-smi not found or failed"

# 7. Execute the passed command
echo "[Wrapper $(hostname)] Executing: $@"
exec "$@"
