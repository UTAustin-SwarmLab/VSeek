#!/bin/bash

# ==========================================
# Default Values (used if flags are not provided)
# ==========================================
BATCH_SIZE=32
TOPK=4
# MODEL_PATH="Qwen/Qwen3-VL-4B-Thinking"
# MODEL_PATH="checkpoints/vseek/qwen3-4bt_vl_all-pulsreward-vllm-wtool-tagsummary/global_step_760/actor/huggingface"
# MODEL_PATH="checkpoints/vseek/qwen3-4bt_vl_all-emreward-vllm-wtool-tagsummary/global_step_680/actor/huggingface"
# MODEL_PATH="checkpoints/vseek/qwen3-4bt_vl_all-emreward-vllm-wtool-tagsummary/global_step_680/actor/huggingface"
MODEL_PATH="checkpoints/vseek/qwen3-4bt_vl_lvbvmmemlvu-emreward-vllm-wtool-tagsummary-gspo-klentr/actor/huggingface"

# MODEL_PATH="Qwen/Qwen3-VL-4B-Thinking"
STORE_PREDS=false
# OUTPUT_BASE="./results/vseek/Qwen3-VL-4B-Thinking"
# OUTPUT_BASE="./results/vseek/VSeek-Puls"
OUTPUT_BASE="./results/vseek/VSeek-GSPO"
# OUTPUT_BASE="./results/vseek/Qwen3-VL-4B-Thinking-Fanout"

PROMPT_TYPE="tagsummary"      # tag, tagsummary
AGENT_TYPE="tagsummary"  # tag, tagsummary, fanout
PASSES=1              # Default
DEVICES="1"
LATENCY_ANALYSIS=false
# ==========================================
# Argument Parsing Logic
# ==========================================
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --prompt-type) PROMPT_TYPE="$2"; shift ;;
        --agent-type)  AGENT_TYPE="$2"; shift ;;
        --passes)      PASSES="$2"; shift ;;
        --passes-per-question) PASSES="$2"; shift ;;
        --batch-size)  BATCH_SIZE="$2"; shift ;;
        --model-path)  MODEL_PATH="$2"; shift ;;
        --devices)     DEVICES="$2"; shift ;;
        --output-base) OUTPUT_BASE="$2"; shift ;;
        --datasets) DATASET_TAGS="$2"; shift ;;
        --store-preds)
            if [[ -n "${2:-}" && ! "$2" =~ ^-- ]]; then
                STORE_PREDS="$2"
                shift
            else
                STORE_PREDS=true
            fi
            ;;
        --latency-analysis)
            LATENCY_ANALYSIS=true
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --prompt-type <str>   (default: tagsummary)"
            echo "  --agent-type <str>    (default: tagsummary)"
            echo "  --passes <int>        (default: 16)"
            echo "  --passes-per-question <int> alias of --passes"
            echo "  --batch-size <int>    (default: 32)"
            echo "  --datasets <tags>     comma-separated tags: videomme,lvb,mlvu,cgbench,lvbench (default: all)"
            echo "  --store-preds <bool>  (default: false)"
            echo "  --latency-analysis    force LVB-only run and keep preds/timings"
            exit 0
            ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ "$LATENCY_ANALYSIS" = true ]; then
    DATASET_TAGS="lvb"
    STORE_PREDS=true
fi

echo "==================================="
echo "Starting Run with Configuration:"
echo "Prompt Type : $PROMPT_TYPE"
echo "Agent Type  : $AGENT_TYPE"
echo "Passes      : $PASSES"
echo "Model Path  : $MODEL_PATH"
echo "Output Base : $OUTPUT_BASE"
echo "Datasets    : ${DATASET_TAGS:-all}"
echo "Latency Run : $LATENCY_ANALYSIS"
echo "==================================="

# ==========================================
# Main Execution Loop
# ==========================================

# Datasets to process
# Format: "DatasetName|ParquetPath|OutputPrefix"
# DATASETS=(
#     "Video-MME|/nas/mars/dataset/Video-MME/window_8/|videomme"
#     "LVB|/nas/mars/dataset/longvideobench/window_8/|lvb"
# )
# DATASETS=(
#     "LVB|/nas/mars/dataset/longvideobench/window_8/|lvb"
# )
ALL_DATASETS=(
    "Video-MME|/nas/mars/dataset/Video-MME/window_8/|videomme"
    "LVB|/nas/mars/dataset/longvideobench/window_8/|lvb"
    "MLVU|/nas/mars/dataset/MLVU/window_8/|mlvu"
    "CGBench|/nas/mars/dataset/CGBench/window_8/|cgbench"
    "LVBench|/nas/mars/dataset/LVBench/window_8/|lvbench"
    "Mined|/home/hg22723/projects/VSeek-R1/mined_payloads/puls_refined_selective/|mined"
)

# Filter datasets by tag if --datasets specified (comma-separated: videomme,lvb,mlvu,cgbench,lvbench)
if [ -n "$DATASET_TAGS" ]; then
    DATASETS=()
    IFS=',' read -ra TAGS <<< "$DATASET_TAGS"
    for entry in "${ALL_DATASETS[@]}"; do
        tag="${entry##*|}"
        for t in "${TAGS[@]}"; do
            if [ "$tag" == "$t" ]; then
                DATASETS+=("$entry")
                break
            fi
        done
    done
    echo "Running datasets: $DATASET_TAGS"
else
    DATASETS=("${ALL_DATASETS[@]}")
fi

for entry in "${DATASETS[@]}"; do
    IFS="|" read -r name parquet_path prefix <<< "$entry"
    
    echo "Running evaluation for $name..."
    export CUDA_VISIBLE_DEVICES="$DEVICES"
    if [ "$name" == "LVB" ] || [ "$name" == "LVBench" ]; then
        MAX_PROMPT_LENGTH=2536
    else
        MAX_PROMPT_LENGTH=2048
    fi
    # Note: Ensure the python flags (--prompt_type, etc.) match exactly what your python script expects
    python3 scripts/run_agent_data_vllmrollout.py \
        --parquet "$parquet_path" \
        --batch_size "$BATCH_SIZE" \
        --output_dir "$OUTPUT_BASE" \
        --topk "$TOPK" \
        --local_model_path="$MODEL_PATH" \
        --output_prefix "$prefix" \
        --prompt_type "$PROMPT_TYPE" \
        --agent_type "$AGENT_TYPE" \
        --passes "$PASSES" \
        --max_prompt_length "$MAX_PROMPT_LENGTH" \
        --store_preds "$STORE_PREDS"
        
    echo "Finished $name"
    echo "-----------------------------------"
done