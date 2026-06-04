#!/bin/bash
# Script to preprocess LongVideoBench (LVB) dataset
# This script processes the LVB dataset into Parquet format compatible with RLHFDataset

# Default values
LOCAL_DATASET_PATH="/nas/mars/dataset/longvideobench/LongVideoBench/"
BURNED_PATH="/nas/mars/dataset/longvideobench"
LOCAL_SAVE_DIR="/nas/mars/dataset/longvideobench"
INDEX_PATH="/home/hg22723/vseek/dataset"
WINDOW_SIZE=8
TRAIN_RATIO=0.8
SEED=42
PROMPT_TYPE="tag"  # Options: tag, openai, tagsummary
PULS_JSON="puls_refined.json"
THUMB_MAX_SIDE=224
THUMB_QUALITY=85
VSEEK_WORKERS=4

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --local_dataset_path PATH    Path to LVB raw data directory (default: $LOCAL_DATASET_PATH)"
    echo "  --burned_path PATH           Path to LVB burned data directory (default: $BURNED_PATH)"
    echo "  --local_save_dir PATH        Output directory (default: $LOCAL_SAVE_DIR)"
    echo "  --index_path PATH            Path to precomputed VideoFrames index (default: $INDEX_PATH)"
    echo "  --window_size SIZE           Window size for VideoFrames index (default: $WINDOW_SIZE)"
    echo "  --train_ratio RATIO          Train split ratio 0-1 (default: $TRAIN_RATIO)"
    echo "  --seed SEED                  Random seed for splitting (default: $SEED)"
    echo "  --prompt_type TYPE           Prompt type: tag, openai, tagsummary (default: $PROMPT_TYPE)"
    echo "  --puls_json PATH             PULS JSON filename/path (default: $PULS_JSON)"
    echo "  --thumb_max_side SIZE        Max side for thumbnail resize (default: $THUMB_MAX_SIDE)"
    echo "  --thumb_quality QUALITY      JPEG quality 1-100 (default: $THUMB_QUALITY)"
    echo "  --workers NUM                Number of workers (default: $VSEEK_WORKERS)"
    echo "  -h, --help                   Display this help message"
    echo ""
    echo "Example:"
    echo "  $0 --window_size 16 --train_ratio 0.9 --prompt_type openai"
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
        --train_ratio)
            TRAIN_RATIO="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --prompt_type)
            PROMPT_TYPE="$2"
            shift 2
            ;;
        --puls_json)
            PULS_JSON="$2"
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
echo "LVB Dataset Preprocessing"
echo "============================================"
echo ""
echo "Dataset Path: $LOCAL_DATASET_PATH"
echo "Burned Path: $BURNED_PATH"
echo "Save Directory: $LOCAL_SAVE_DIR"
echo "Index Path: $INDEX_PATH"
echo "Window Size: $WINDOW_SIZE"
echo "Train Ratio: $TRAIN_RATIO"
echo "Prompt Type: $PROMPT_TYPE"
echo "PULS JSON: $PULS_JSON"
echo ""

# Check if running in the project root
if [ ! -f "src/data/lvb_preprocessor.py" ]; then
    echo "Error: Please run this script from the project root directory"
    exit 1
fi

# Create necessary directories
mkdir -p "$LOCAL_DATASET_PATH"
mkdir -p "$BURNED_PATH"
mkdir -p "$LOCAL_SAVE_DIR"
mkdir -p "$INDEX_PATH"

# Run the preprocessor
echo "Running LVB preprocessor..."
python3 src/data/lvb_preprocessor.py \
    --local_dataset_path "$LOCAL_DATASET_PATH" \
    --burned_path "$BURNED_PATH" \
    --train_ratio $TRAIN_RATIO \
    --local_save_dir "$LOCAL_SAVE_DIR" \
    --index_path "$INDEX_PATH" \
    --window_size $WINDOW_SIZE \
    --embed_frames \
    --prompt_type "$PROMPT_TYPE" \
    --puls_json "$PULS_JSON" \
    --thumb_max_side $THUMB_MAX_SIDE \
    --thumb_quality $THUMB_QUALITY \
    --seed $SEED

# Check if the command succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "============================================"
    echo "Processing Complete!"
    echo "============================================"
    echo ""
    echo "Output locations:"
    echo "  Parquet files: $LOCAL_SAVE_DIR/window_$WINDOW_SIZE/$PROMPT_TYPE/"
    echo ""
else
    echo ""
    echo "============================================"
    echo "Processing Failed!"
    echo "============================================"
    exit 1
fi