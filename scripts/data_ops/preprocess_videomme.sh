#!/bin/bash
# Process Video-MME dataset
# This script demonstrates how to process the Video-MME dataset and create parquet files

# Default values
DATASET_PATH="/nas/mars/dataset/Video-MME"
BURNED_PATH="/nas/mars/dataset/Video-MME/burn-subtitles"
SAVE_DIR="/nas/mars/dataset/Video-MME"
INDEX_PATH="/home/hg22723/vseek/dataset"
RETRIEVAL_MODEL_PATH=""
WINDOW_SIZE=8
GPU_NUMBER=0
PROMPT_TYPE="tag"  # Options: tag, openai, tagsummary
PULS_JSON="puls_refined.json"
TRAIN_RATIO=0.8
SEED=42
MODE="both"  # Options: both, index, parquet
THUMB_MAX_SIDE=224
THUMB_QUALITY=85
VSEEK_WORKERS=4

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --local_dataset_path PATH          Path to Video-MME dataset (default: $DATASET_PATH)"
    echo "  --burned_path PATH           Path to burned videos with subtitles (default: $BURNED_PATH)"
    echo "  --local_save_dir PATH              Output directory for parquet files (default: $SAVE_DIR)"
    echo "  --index_path PATH            Path to store indexed videos (default: $INDEX_PATH)"
    echo "  --retrieval_model_path PATH  Path to ViClip retrieval model (optional)"
    echo "  --window_size SIZE           Window size for VideoFrames index (default: $WINDOW_SIZE)"
    echo "  --gpu_number NUM             GPU number to use (default: $GPU_NUMBER)"
    echo "  --prompt_type TYPE           Prompt type: tag, openai, tagsummary (default: $PROMPT_TYPE)"
    echo "  --puls_json PATH             PULS JSON filename/path (default: $PULS_JSON)"
    echo "  --train_ratio RATIO          Train split ratio 0-1 (default: $TRAIN_RATIO)"
    echo "  --seed SEED                  Random seed for splitting (default: $SEED)"
    echo "  --mode MODE                  Processing mode: both, index, parquet (default: $MODE)"
    echo "  --thumb_max_side SIZE        Max side for thumbnail resize (default: $THUMB_MAX_SIDE)"
    echo "  --thumb_quality QUALITY      JPEG quality 1-100 (default: $THUMB_QUALITY)"
    echo "  --workers NUM                Number of workers (default: $VSEEK_WORKERS)"
    echo "  -h, --help                   Display this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --window_size 16 --train_ratio 0.9"
    echo "  $0 --mode index --gpu_number 1"
    echo "  $0 --mode parquet --prompt_type openai"
    exit 1
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --local_dataset_path)
            DATASET_PATH="$2"
            shift 2
            ;;
        --burned_path)
            BURNED_PATH="$2"
            shift 2
            ;;
        --local_save_dir)
            SAVE_DIR="$2"
            shift 2
            ;;
        --index_path)
            INDEX_PATH="$2"
            shift 2
            ;;
        --retrieval_model_path)
            RETRIEVAL_MODEL_PATH="$2"
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
        --puls_json)
            PULS_JSON="$2"
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
        --mode)
            MODE="$2"
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
echo "Video-MME Dataset Preprocessing"
echo "============================================"
echo ""
echo "Dataset Path: $DATASET_PATH"
echo "Burned Path: $BURNED_PATH"
echo "Save Directory: $SAVE_DIR"
echo "Index Path: $INDEX_PATH"
echo "Window Size: $WINDOW_SIZE"
echo "Train Ratio: $TRAIN_RATIO"
echo "Prompt Type: $PROMPT_TYPE"
echo "PULS JSON: $PULS_JSON"
echo "Mode: $MODE"
echo ""

# Check if running in the project root
if [ ! -f "src/data/videomme_preprocessor.py" ]; then
    echo "Error: Please run this script from the project root directory"
    exit 1
fi

# Create necessary directories
mkdir -p "$DATASET_PATH"
mkdir -p "$BURNED_PATH"
mkdir -p "$SAVE_DIR"
mkdir -p "$INDEX_PATH"

# Build the command based on mode
echo "Running Video-MME preprocessor (mode: $MODE)..."

CMD="python3 src/data/videomme_preprocessor.py \
    --local_dataset_path \"$DATASET_PATH\" \
    --burned_path \"$BURNED_PATH\" \
    --puls_json \"$PULS_JSON\" \
    --window_size $WINDOW_SIZE \
    --gpu_number $GPU_NUMBER \
    --mode $MODE \
    --train_ratio $TRAIN_RATIO \
    --seed $SEED"

# Add index_path for all modes
CMD="$CMD --index_path \"$INDEX_PATH\""

# Add parquet-specific options for both and parquet modes
if [ "$MODE" = "both" ] || [ "$MODE" = "parquet" ]; then
    CMD="$CMD --local_save_dir \"$SAVE_DIR\" \
    --embed_frames \
    --thumb_max_side $THUMB_MAX_SIDE \
    --thumb_quality $THUMB_QUALITY \
    --prompt_type \"$PROMPT_TYPE\""
fi

# Add retrieval model path if provided
if [ -n "$RETRIEVAL_MODEL_PATH" ]; then
    CMD="$CMD --retrieval_model_path \"$RETRIEVAL_MODEL_PATH\""
fi

# Execute the command
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
        echo "  Indexed videos: $INDEX_PATH/videomme_window_$WINDOW_SIZE/"
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

