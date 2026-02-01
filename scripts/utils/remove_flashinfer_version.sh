#!/bin/bash
# Script to remove version constraint from flashinfer-python in vllm requirements

VLLM_DIR="${1:-/work/11123/harshgoel99/vista/vllm}"
CUDA_REQS="$VLLM_DIR/requirements/cuda.txt"

echo "Removing flashinfer-python version constraint..."
echo "File: $CUDA_REQS"

# Check if file exists
if [ ! -f "$CUDA_REQS" ]; then
    echo "Error: File not found: $CUDA_REQS"
    exit 1
fi

# Show before
echo -e "\nBefore:"
grep flashinfer-python "$CUDA_REQS" || echo "flashinfer-python not found"

# Replace flashinfer-python==X.X.X with flashinfer-python
sed -i 's/^flashinfer-python==.*/flashinfer-python/' "$CUDA_REQS"

# Also handle if it has >= or other operators
sed -i 's/^flashinfer-python[><=!].*/flashinfer-python/' "$CUDA_REQS"

# Show after
echo -e "\nAfter:"
grep flashinfer-python "$CUDA_REQS" || echo "flashinfer-python not found"

echo -e "\n✓ Done! flashinfer-python version constraint removed."
