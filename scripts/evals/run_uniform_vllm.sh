#!/bin/bash

# Default values
DEVICE=0
MODEL="Qwen/Qwen3-VL-4B-Thinking"
FRAMES=64
OUTPUT_DIR="results/uniform_vllm"
DATASETS="lvb videomme mlvu lvbench"
PROMPT_TYPE="cot"
SERVER_PORT=8002
TEMPERATURE=0.7
BATCH_SIZE=16
MAX_MM_CACHE_RESTARTS=3
RESTART_SLEEP_SECONDS=5
MM_CACHE_ERROR_PATTERN="AssertionError: Expected a cached item for mm_hash="
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
        --batch-size)
            BATCH_SIZE="$2"
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

for DATASET in $DATASETS; do
    echo "Processing dataset: $DATASET"
    
    # Construct specific output dir for this run configuration
    RUN_OUTPUT_DIR="${OUTPUT_DIR}/${DATASET}/${PROMPT_TYPE}/f${FRAMES}/${MODEL_NAME}"
    
    echo "Running command..."
    mm_cache_restarts=0
    while true; do
        run_log="$(mktemp)"
        mm_error_detected=0

        # Run in background so we can detect the mm cache error and kill stalled runs.
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
            +inference.passes=4 \
            inference.temperature="$TEMPERATURE" \
            +inference.batch_size="$BATCH_SIZE" \
            > >(tee "$run_log") 2>&1 &
        cmd_pid=$!

        while kill -0 "$cmd_pid" >/dev/null 2>&1; do
            if grep -Fq "$MM_CACHE_ERROR_PATTERN" "$run_log"; then
                mm_error_detected=1
                mm_cache_restarts=$((mm_cache_restarts + 1))
                echo "Detected mm cache error for $DATASET (attempt ${mm_cache_restarts}/${MAX_MM_CACHE_RESTARTS})."

                echo "Stopping stalled run and cleaning stale vLLM/Ray workers..."
                kill -TERM "$cmd_pid" >/dev/null 2>&1 || true
                sleep 2
                kill -KILL "$cmd_pid" >/dev/null 2>&1 || true
                wait "$cmd_pid" >/dev/null 2>&1 || true
                pkill -f "vLLMHttpServer|run_uniform_agent_data.py|ray::|raylet" >/dev/null 2>&1 || true
                break
            fi
            sleep 2
        done

        if [ "$mm_error_detected" -eq 1 ]; then
            rm -f "$run_log"
            if [ "$mm_cache_restarts" -gt "$MAX_MM_CACHE_RESTARTS" ]; then
                echo "Exceeded max mm cache restart attempts for $DATASET. Exiting."
                exit 1
            fi
            sleep "$RESTART_SLEEP_SECONDS"
            echo "Retrying dataset $DATASET..."
            continue
        fi

        wait "$cmd_pid"
        cmd_status=$?
        if [ "$cmd_status" -eq 0 ]; then
            rm -f "$run_log"
            break
        fi

        echo "Command failed for $DATASET (non-mm_cache error). Exiting."
        grep -F "$MM_CACHE_ERROR_PATTERN" "$run_log" >/dev/null 2>&1 || cat "$run_log"
        rm -f "$run_log"
        exit "$cmd_status"
    done
        
    echo "Finished $DATASET"
    echo "--------------------------------------------------------"
done

echo "All evaluations completed."
