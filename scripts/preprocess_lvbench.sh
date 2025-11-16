#!/bin/bash
# Script to preprocess LVBench dataset
# This script demonstrates how to use the lvbench_preprocessor.py

# Default values
LOCAL_DATASET_PATH="/nas/mars/dataset/LVBench"
INDEX_PATH="/home/hg22723/vseek/dataset"
LOCAL_SAVE_DIR="/nas/mars/dataset/LVBench"
BURNED_PATH="/nas/mars/dataset/LVBench/videos"
WINDOW_SIZE=8
GPU_NUMBER=0
PROMPT_TYPE="tag"  # Options: tag, openai, tagsummary
TRAIN_RATIO=0.8
SEED=42
THUMB_MAX_SIDE=224
THUMB_QUALITY=85
VSEEK_WORKERS=4

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --local_dataset_path PATH    Path to LVBench dataset (default: $LOCAL_DATASET_PATH)"
    echo "  --burned_path PATH           Path to burned LVBench data with burned videos and subtitles (default: $BURNED_PATH)"
    echo "  --index_path PATH            Path to store indexed videos (default: $INDEX_PATH)"
    echo "  --local_save_dir PATH        Output directory for parquet files (default: $LOCAL_SAVE_DIR)"
    echo "  --window_size SIZE           Window size for VideoFrames index (default: $WINDOW_SIZE)"
    echo "  --gpu_number NUM             GPU number to use (default: $GPU_NUMBER)"
    echo "  --prompt_type TYPE           Prompt type: tag, openai, tagsummary (default: $PROMPT_TYPE)"
    echo "  --train_ratio RATIO          Train split ratio 0-1 (default: $TRAIN_RATIO)"
    echo "  --seed SEED                  Random seed for splitting (default: $SEED)"
    echo "  --thumb_max_side SIZE        Max side for thumbnail resize (default: $THUMB_MAX_SIDE)"
    echo "  --thumb_quality QUALITY      JPEG quality 1-100 (default: $THUMB_QUALITY)"
    echo "  --workers NUM                Number of workers (default: $VSEEK_WORKERS)"
    echo "  -h, --help                   Display this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --window_size 16 --train_ratio 0.9"
    exit 1
}

# Parse command-line arguments
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
        --index_path)
            INDEX_PATH="$2"
            shift 2
            ;;
        --local_save_dir)
            LOCAL_SAVE_DIR="$2"
            shift 2
            ;;
        --window_size)
            WINDOW_SIZE="$2"
            shift 2
            ;;
        --gpu_number)
            GPU_NUMBER="$2"
            shift 2
            ;;
        --prompt_type)
            PROMPT_TYPE="$2"
            shift 2
            ;;
        --train_ratio)
            TRAIN_RATIO="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --thumb_max_side)
            THUMB_MAX_SIDE="$2"
            shift 2
            ;;
        --thumb_quality)
            THUMB_QUALITY="$2"
            shift 2
            ;;
        --retrieval_model_path)
            RETRIEVAL_MODEL_PATH="$2"
            shift 2
            ;;
        --workers)
            VSEEK_WORKERS="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Error: Unknown option $1"
            usage
            ;;
    esac
done

# Set number of workers
export VSEEK_WORKERS

echo "============================================"
echo "LVBench Dataset Preprocessing"
echo "============================================"
echo ""
echo "Local Dataset Path: $LOCAL_DATASET_PATH"
echo "Burned Path: $BURNED_PATH"
echo "Index Path: $INDEX_PATH"
echo "Local Save Directory: $LOCAL_SAVE_DIR"
echo "Window Size: $WINDOW_SIZE"
echo "Prompt Type: $PROMPT_TYPE"
echo ""

# Check if running in the project root
if [ ! -f "src/data/lvbench_preprocessor.py" ]; then
    echo "Error: Please run this script from the project root directory"
    exit 1
fi

# Create necessary directories
mkdir -p "$LOCAL_DATASET_PATH"
mkdir -p "$BURNED_PATH"
mkdir -p "$INDEX_PATH"
mkdir -p "$LOCAL_SAVE_DIR"

# Build the command based on mode
echo "Running LVBench preprocessor (mode: $MODE)..."

CMD="python src/data/lvbench_preprocessor.py \
    --local_dataset_path \"$LOCAL_DATASET_PATH\" \
    --window_size $WINDOW_SIZE \
    --burned_path "$BURNED_PATH" \
    --train_ratio $TRAIN_RATIO \
    --seed $SEED \
    --index_path \"$INDEX_PATH\" \
    --local_save_dir \"$LOCAL_SAVE_DIR\" \
    --embed_frames \
    --thumb_max_side $THUMB_MAX_SIDE \
    --thumb_quality $THUMB_QUALITY \
    --prompt_type \"$PROMPT_TYPE\""

eval $CMD

# Check if the command succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "============================================"
    echo "Processing Complete!"
    echo "============================================"
    echo ""
    echo "Output locations:"
    if [ "$MODE" = "both" ] || [ "$MODE" = "index" ]; then
        echo "  Indexed videos: $INDEX_PATH/lvbench_window_$WINDOW_SIZE/"
    fi
    if [ "$MODE" = "both" ] || [ "$MODE" = "parquet" ]; then
        echo "  Parquet files: $SAVE_DIR/window_$WINDOW_SIZE/$PROMPT_TYPE/"
    fi
    echo ""
    echo "Next steps:"
    echo "  1. Verify the output files"
    if [ "$MODE" = "both" ] || [ "$MODE" = "parquet" ]; then
        echo "  2. Update your training config to use these parquet files"
        echo "  3. Run training with: python scripts/train/your_training_script.py"
    fi
else
    echo ""
    echo "============================================"
    echo "Processing Failed!"
    echo "============================================"
    exit 1
fi

