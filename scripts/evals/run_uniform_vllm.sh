#!/bin/bash

# Default values
DEVICE=0
MODEL="Qwen/Qwen3-VL-4B-Thinking"
FRAMES=64
OUTPUT_DIR="results/uniform_vllm"
DATASETS=("lvb" "lvbench" "videomme")
PROMPT_TYPE="cot"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--device)
            DEVICE="$2"
            shift 2
            ;;
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -f|--frames)
            FRAMES="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --datasets)
            # Read remaining arguments as datasets until a flag is found
            shift
            DATASETS=()
            while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do
                DATASETS+=("$1")
                shift
            done
            ;;
        --prompt-type)
            PROMPT_TYPE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  -d, --device ID       GPU device ID (default: 0)"
            echo "  -m, --model PATH      Model path or HF ID (default: Qwen/Qwen3-VL-4B-Thinking)"
            echo "  -f, --frames N        Max frames per turn (default: 64)"
            echo "  -o, --output-dir DIR  Output directory (default: results/uniform_vllm)"
            echo "  --datasets NAMES      Space-separated list of datasets (default: lvb lvbench videomme)"
            echo "  --prompt-type TYPE    Agent prompt type: base or cot (default: cot)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Export VLLM environment variable
export VLLM_USE_V1=1

echo "========================================================"
echo "Running Uniform Agent Evaluation with VLLM"
echo "Device: $DEVICE"
echo "Model: $MODEL"
echo "Frames per turn: $FRAMES"
echo "Output Dir: $OUTPUT_DIR"
echo "Datasets: ${DATASETS[*]}"
echo "Prompt Type: $PROMPT_TYPE"
echo "========================================================"

for DATASET in "${DATASETS[@]}"; do
    echo "Processing dataset: $DATASET"
    
    # Construct specific output dir for this run configuration
    # You can adjust this naming scheme as preferred
    RUN_OUTPUT_DIR="${OUTPUT_DIR}/${DATASET}_f${FRAMES}"
    
    echo "Running command..."
    CUDA_VISIBLE_DEVICES=$DEVICE python3 scripts/run_uniform_agent_data.py \
        inference.max_images_per_turn=$FRAMES \
        +inference.agent_prompt_type=$PROMPT_TYPE \
        llm.model="$MODEL" \
        inference.output_dir="$OUTPUT_DIR" \
        inference.max_output_tokens=4096 \
        inference.gpu_number=$DEVICE \
        dataset.name=$DATASET
        
    echo "Finished $DATASET"
    echo "--------------------------------------------------------"
done

echo "All evaluations completed."

