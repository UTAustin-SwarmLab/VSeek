#!/bin/bash

# Default values
DEVICE=0
# MODEL="Qwen/Qwen3-VL-4B-Thinking"
MODEL="OpenGVLab/InternVL3_5-4B-HF"
FRAMES=64
OUTPUT_DIR="results/uniform_vllm"
DATASETS="lvb"
PROMPT_TYPE="cot"
SERVER_PORT=8002
TEMPERATURE=0.7
# Parse command line arguments
while [ $# -gt 0 ]; do
    case "$1" in
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
            shift
            DATASETS=""
            while [ $# -gt 0 ]; do
                case "$1" in
                    -*) break ;;
                    *) DATASETS="$DATASETS $1"; shift ;;
                esac
            done
            ;;
        --prompt-type)
            PROMPT_TYPE="$2"
            shift 2
            ;;
        --vllm-port)
            SERVER_PORT="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  -d, --device ID       GPU device ID (default: 0)"
            echo "  -m, --model PATH      Model path or HF ID (default: Qwen/Qwen3-VL-4B-Thinking)"
            echo "  -f, --frames N        Max frames per turn (default: 64)"
            echo "  -o, --output-dir DIR  Output directory (default: results/uniform_vllm)"
            echo "  --vllm-port PORT      Port for vLLM server (default: 8002)"
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

# Start vLLM server in background
echo "Starting vLLM server on port $SERVER_PORT using device $DEVICE..."

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export NCCL_P2P_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_USE_V1=1

# Global export to ensure vLLM and children see only this GPU as GPU 0
export CUDA_VISIBLE_DEVICES=$DEVICE
MODEL_NAME=${MODEL##*/}
RUN_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# notify.py "Starting experiment $MODEL on $DEVICE with datasets $DATASETS"

for DATASET in $DATASETS; do
    echo "Processing dataset: $DATASET"
    
    # Construct specific output dir for this run configuration
    RUN_OUTPUT_DIR="${OUTPUT_DIR}/${DATASET}/${PROMPT_TYPE}/f${FRAMES}/${MODEL_NAME}"
    mkdir -p "$RUN_OUTPUT_DIR"
    RUN_LOG_FILE="${RUN_OUTPUT_DIR}/run_${RUN_TIMESTAMP}.log"
    
    echo "Running command..."
    echo "Writing output to $RUN_LOG_FILE"
    # CUDA_VISIBLE_DEVICES is already exported globally
    python3 scripts/run_uniform_agent_data.py \
        +agent_type=uniform \
        inference.max_images_per_turn="$FRAMES" \
        +inference.agent_prompt_type="$PROMPT_TYPE" \
        llm.model="$MODEL" \
        llm.server_url="http://localhost:$SERVER_PORT/v1" \
        inference.output_dir="$RUN_OUTPUT_DIR" \
        inference.max_output_tokens=8192 \
        inference.gpu_number="$DEVICE" \
        dataset.name="$DATASET" \
        +inference.passes=16 \
        inference.temperature="$TEMPERATURE" \
        +inference.batch_size=16 \
        2>&1 | tee "$RUN_LOG_FILE"
        
    echo "Finished $DATASET"
    echo "--------------------------------------------------------"
done

echo "All evaluations completed."

notify.py "Experiment $MODEL on $DEVICE completed with datasets $DATASETS"