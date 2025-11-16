#!/bin/bash

# MLVU Dataset Preprocessing Script
# This script merges MLVU category JSON files into a single mlvu_val.json file

# Default dataset path (update this to your MLVU dataset location)
DATASET_PATH="${MLVU_DATASET_PATH:-/nas/mars/dataset/mlvu}"
OUTPUT_FILENAME="mlvu_val.json"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset_path)
            DATASET_PATH="$2"
            shift 2
            ;;
        --output_filename)
            OUTPUT_FILENAME="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dataset_path PATH      Path to MLVU dataset (default: \$MLVU_DATASET_PATH or /nas/mars/dataset/mlvu)"
            echo "  --output_filename NAME   Output filename (default: mlvu_val.json)"
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
if [ ! -d "$DATASET_PATH" ]; then
    echo "Error: Dataset path does not exist: $DATASET_PATH"
    echo "Please set MLVU_DATASET_PATH environment variable or use --dataset_path option"
    exit 1
fi

echo "====================================="
echo "MLVU Dataset Preprocessing"
echo "====================================="
echo "Dataset path: $DATASET_PATH"
echo "Output filename: $OUTPUT_FILENAME"
echo ""

# Change to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

# Run the preprocessing script
python scripts/merge_mlvu_categories.py \
    --dataset_path "$DATASET_PATH" \
    --output_filename "$OUTPUT_FILENAME"

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "====================================="
    echo "Preprocessing completed successfully!"
    echo "====================================="
    echo ""
    echo "Next steps:"
    echo "1. Index the videos with:"
    echo "   python scripts/data_ops/run_data_pipeline.py dataset.name=mlvu dataset.dataset_path=$DATASET_PATH"
    echo ""
else
    echo ""
    echo "====================================="
    echo "Preprocessing failed with exit code: $exit_code"
    echo "====================================="
    echo ""
fi

exit $exit_code

