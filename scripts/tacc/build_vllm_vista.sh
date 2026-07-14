#!/bin/bash
# Build script for vllm on TACC Vista (ARM64/Grace Hopper)
# This script handles the complex build process for ARM64 architecture

set -e  # Exit on error

echo "=========================================="
echo "Building vllm on TACC Vista (ARM64)"
echo "=========================================="

# Configuration
CONDA_ENV="vseek-vllm2"
VLLM_DIR="/work/11123/harshgoel99/vista/vllm"
SKIP_FLASH_ATTN=true  # Set to false to attempt building flash-attn

# Activate conda environment
echo "Activating conda environment: $CONDA_ENV"
source ~/.bashrc
conda activate $CONDA_ENV

# Set up module environment
echo "Loading modules..."
module purge
module load gcc/14.2.0
module load cuda/12.8

# Set compiler environment
echo "Setting up compiler environment..."
export CUDA_HOME="$TACC_CUDA_DIR"
export CC=/opt/apps/gcc/14.2.0/bin/gcc
export CXX=/opt/apps/gcc/14.2.0/bin/g++
export CUDAHOSTCXX=/opt/apps/gcc/14.2.0/bin/g++

# Critical: Only target Grace Hopper (sm_90)
export TORCH_CUDA_ARCH_LIST="9.0"

# Verify environment
echo "=========================================="
echo "Environment Check:"
echo "GCC Version: $(gcc --version | head -1)"
echo "CUDA Version: $(nvcc --version | grep release)"
echo "Python: $(python --version)"
echo "Conda Env: $CONDA_PREFIX"
echo "=========================================="

# Install dependencies (skip flash-attn if needed)
echo "Installing Python dependencies..."
if [ "$SKIP_FLASH_ATTN" = true ]; then
    echo "Skipping flash-attn (will build vllm without it)"
    grep -v "flash-attn" /home1/11123/harshgoel99/VSeek-R1/requirements_vllm_slurm2.txt > /tmp/requirements_no_flash.txt
    pip install -r /tmp/requirements_no_flash.txt
else
    echo "Attempting to install flash-attn..."
    MAX_JOBS=2 TORCH_CUDA_ARCH_LIST="9.0" pip install flash-attn==2.8.3 --no-build-isolation --no-cache-dir || {
        echo "WARNING: flash-attn build failed, continuing without it"
        SKIP_FLASH_ATTN=true
    }
fi

# Navigate to vllm directory
echo "Navigating to vllm directory: $VLLM_DIR"
cd $VLLM_DIR

# Clean previous builds
echo "Cleaning previous build artifacts..."
rm -rf build build_debug .deps CMakeCache.txt *.egg-info

# Set vllm build flags
export VLLM_NO_MARLIN_MOE=1  # Disable problematic marlin MOE kernels
export VLLM_NO_USAGE_STATS=1

if [ "$SKIP_FLASH_ATTN" = true ]; then
    export VLLM_NO_FLASH_ATTN=1
    echo "Building vllm WITHOUT flash-attn support"
else
    echo "Building vllm WITH flash-attn support"
fi

# Build vllm
echo "=========================================="
echo "Building vllm..."
echo "=========================================="
MAX_JOBS=4 pip install -e . --no-build-isolation --no-deps -v

# Verify installation
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="
python -c "import vllm; print(f'✓ vllm version: {vllm.__version__}')" && \
python -c "import torch; print(f'✓ PyTorch version: {torch.__version__}')" && \
python -c "import torch; print(f'✓ CUDA available: {torch.cuda.is_available()}')" && \
echo "=========================================="
echo "✓ Build completed successfully!"
echo "=========================================="
