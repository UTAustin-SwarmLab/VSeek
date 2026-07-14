#!/bin/bash
# Process CGBench dataset into parquet files.

# Default values
DATASET_PATH="/nas/mars/dataset/CGBench"
BURNED_PATH="/nas/mars/dataset/CGBench/burn-subtitles"
SAVE_DIR="/nas/mars/dataset/CGBench"
INDEX_PATH="/home/hg22723/vseek/dataset"
WINDOW_SIZE=8
PROMPT_TYPE="tagsummary"  # Options: tag, openai, tagsummary, fanout
TRAIN_RATIO=0.8
SEED=42
THUMB_MAX_SIDE=224
THUMB_QUALITY=85
MAX_FRAMES_PER_TURN=16
VSEEK_WORKERS=4

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --local_dataset_path PATH    Path to CGBench dataset (default: $DATASET_PATH)"
    echo "  --burned_path PATH           Path to burned videos with subtitles (default: $BURNED_PATH)"
    echo "  --local_save_dir PATH        Output directory for parquet files (default: $SAVE_DIR)"
    echo "  --index_path PATH            Path to precomputed indexed videos (default: $INDEX_PATH)"
    echo "  --window_size SIZE           Window size for VideoFrames index (default: $WINDOW_SIZE)"
    echo "  --prompt_type TYPE           Prompt type: tag, openai, tagsummary, fanout (default: $PROMPT_TYPE)"
    echo "  --train_ratio RATIO          Train split ratio 0-1 (default: $TRAIN_RATIO)"
    echo "  --seed SEED                  Random seed for splitting (default: $SEED)"
    echo "  --max_frames_per_turn NUM    Maximum number of summary frames (default: $MAX_FRAMES_PER_TURN)"
    echo "  --thumb_max_side SIZE        Max side for thumbnail resize (default: $THUMB_MAX_SIDE)"
    echo "  --thumb_quality QUALITY      JPEG quality 1-100 (default: $THUMB_QUALITY)"
    echo "  --workers NUM                Number of workers (default: $VSEEK_WORKERS)"
    echo "  -h, --help                   Display this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --window_size 16 --train_ratio 0.9"
    echo "  $0 --prompt_type openai --local_dataset_path /nas/mars/dataset/CGBench"
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
        --max_frames_per_turn)
            MAX_FRAMES_PER_TURN="$2"
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
echo "CGBench Dataset Preprocessing"
echo "============================================"
echo ""
echo "Dataset Path: $DATASET_PATH"
echo "Burned Path: $BURNED_PATH"
echo "Save Directory: $SAVE_DIR"
echo "Index Path: $INDEX_PATH"
echo "Window Size: $WINDOW_SIZE"
echo "Train Ratio: $TRAIN_RATIO"
echo "Prompt Type: $PROMPT_TYPE"
echo ""

# Check if running in the project root
if [ ! -f "src/data/cgbench_preprocessor.py" ]; then
    echo "Error: Please run this script from the project root directory"
    exit 1
fi

# Create necessary directories
mkdir -p "$DATASET_PATH"
mkdir -p "$BURNED_PATH"
mkdir -p "$SAVE_DIR"
mkdir -p "$INDEX_PATH"

# Run the preprocessor
echo "Running CGBench preprocessor..."
python3 src/data/cgbench_preprocessor.py \
    --local_dataset_path "$DATASET_PATH" \
    --burned_path "$BURNED_PATH" \
    --train_ratio "$TRAIN_RATIO" \
    --local_save_dir "$SAVE_DIR" \
    --index_path "$INDEX_PATH" \
    --window_size "$WINDOW_SIZE" \
    --max_frames_per_turn "$MAX_FRAMES_PER_TURN" \
    --embed_frames \
    --prompt_type "$PROMPT_TYPE" \
    --thumb_max_side "$THUMB_MAX_SIDE" \
    --thumb_quality "$THUMB_QUALITY" \
    --seed "$SEED"

# Check if the command succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "============================================"
    echo "Processing Complete!"
    echo "============================================"
    echo ""
    echo "Output locations:"
    echo "  Parquet files: $SAVE_DIR/window_$WINDOW_SIZE/$PROMPT_TYPE/"
    echo ""
    echo "Next steps:"
    echo "  1. Verify the output files"
    echo "  2. Update your training config to use these parquet files"
    echo "  3. Run training with: python scripts/train/your_training_script.py"
else
    echo ""
    echo "============================================"
    echo "Processing Failed!"
    echo "============================================"
    exit 1
fi

