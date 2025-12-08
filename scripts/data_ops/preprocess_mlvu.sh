#!/bin/bash

# MLVU Dataset Preprocessing Script

# Default values
LOCAL_DATASET_PATH="${LOCAL_DATASET_PATH:-/nas/mars/dataset/mlvu}"
BURNED_PATH="${BURNED_PATH:-/nas/mars/dataset/mlvu/burned}"
TRAIN_RATIO="${TRAIN_RATIO:-0.9}"
LOCAL_SAVE_DIR="${LOCAL_SAVE_DIR:-~/data/mlvu}"
INDEX_PATH="${INDEX_PATH:-/nas/mars/dataset/mlvu/index}"
WINDOW_SIZE="${WINDOW_SIZE:-8}"
PROMPT_TYPE="${PROMPT_TYPE:-tag}"
MAX_FRAMES_PER_TURN="${MAX_FRAMES_PER_TURN:-16}"
GPU_NUMBER="${GPU_NUMBER:-0}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --local_dataset_path)
            LOCAL_DATASET_PATH="$2"
            shift 2
            ;;
        --burned_path)
            BURNED_PATH="$2"
            shift 2
            ;;
        --local_save_dir)
            LOCAL_SAVE_DIR="$2"
            shift 2
            ;;
        --index_path)
            INDEX_PATH="$2"
            shift 2
            ;;
        --window_size)
            WINDOW_SIZE="$2"
            shift 2
            ;;
        --prompt_type)
            PROMPT_TYPE="$2"
            shift 2
            ;;
        --max_frames_per_turn)
            MAX_FRAMES_PER_TURN="$2"
            shift 2
            ;;
        --gpu_number)
            GPU_NUMBER="$2"
            shift 2
            ;;
        --embed_frames)
            EMBED_FRAMES="--embed_frames"
            shift 1
            ;;
        --train_ratio)
            TRAIN_RATIO="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dataset_path PATH      Path to MLVU dataset (default: \$MLVU_DATASET_PATH or /nas/mars/dataset/mlvu)"
            echo "  --burned_path PATH       Path to burned MLVU dataset"
            echo "  --local_save_dir PATH    Output directory for parquet files"
            echo "  --index_path PATH        Path to save/load precomputed video index"
            echo "  --mode [index|parquet|both] Mode of operation (default: both)"
            echo "  -h, --help               Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if dataset path exists
if [ ! -d "$LOCAL_DATASET_PATH" ]; then
    echo "Error: Dataset path does not exist: $DATASET_PATH"
    echo "Please set MLVU_DATASET_PATH environment variable or use --dataset_path option"
    exit 1
fi

echo "====================================="
echo "MLVU Dataset Preprocessing"
echo "====================================="
echo "Dataset path: $DATASET_PATH"
echo "Mode: $MODE"
echo ""

# Change to project root (assuming script is in scripts/data_ops/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../" && pwd)"
cd "$PROJECT_ROOT" || exit 1

echo "Project root: $PROJECT_ROOT"

# Run the preprocessing script
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Build arguments array
ARGS=(
    --local_dataset_path "$LOCAL_DATASET_PATH"
    --burned_path "$BURNED_PATH"
    --train_ratio "$TRAIN_RATIO"
    --local_save_dir "$LOCAL_SAVE_DIR"
    --index_path "$INDEX_PATH"
    --window_size "$WINDOW_SIZE"
    --prompt_type "$PROMPT_TYPE"
    --max_frames_per_turn "$MAX_FRAMES_PER_TURN"
    --gpu_number "$GPU_NUMBER"
)

# Add embed_frames flag if index_path is set
if [ -n "$INDEX_PATH" ]; then
    ARGS+=(--embed_frames)
fi

# Execute python script
python3 src/data/mvlu_preprocessor.py "${ARGS[@]}"

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "====================================="
    echo "Preprocessing completed successfully!"
    echo "====================================="
else
    echo ""
    echo "====================================="
    echo "Preprocessing failed with exit code: $exit_code"
    echo "====================================="
fi

exit $exit_code

