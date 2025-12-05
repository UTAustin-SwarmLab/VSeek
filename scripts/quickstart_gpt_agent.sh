#!/bin/bash
# Quick start script for GPT agent training data generation

set -e

echo "============================================================"
echo "GPT Agent Training Data Generator - Quick Start"
echo "============================================================"

# Check for OpenAI API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ ERROR: OPENAI_API_KEY environment variable not set"
    echo ""
    echo "Please set your OpenAI API key:"
    echo "  export OPENAI_API_KEY='your-api-key-here'"
    echo ""
    exit 1
fi

echo "✅ OpenAI API key found"

# Default parameters
PARQUET_PATH="${PARQUET_PATH:-~/data/lvb/test.parquet}"
MODEL="${MODEL:-gpt-4-turbo-preview}"
N_SAMPLES="${N_SAMPLES:-4}"
COUNT="${COUNT:-10}"
BATCH_SIZE="${BATCH_SIZE:-4}"
OUTPUT_DIR="${OUTPUT_DIR:-~/results/gpt_agent_training_data}"
SELECTION_STRATEGY="${SELECTION_STRATEGY:-all_correct}"

echo ""
echo "📋 Configuration:"
echo "  Parquet path:        $PARQUET_PATH"
echo "  Model:               $MODEL"
echo "  Samples per question: $N_SAMPLES"
echo "  Number of questions: $COUNT"
echo "  Batch size:          $BATCH_SIZE"
echo "  Selection strategy:  $SELECTION_STRATEGY"
echo "  Output directory:    $OUTPUT_DIR"
echo ""

# Confirm with user (can be skipped with -y flag)
if [ "$1" != "-y" ]; then
    read -p "Continue? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

echo ""
echo "🚀 Starting generation..."
echo "============================================================"

# Run the script
python scripts/run_gpt_agent_data.py \
    --parquet "$PARQUET_PATH" \
    --model "$MODEL" \
    --n_samples "$N_SAMPLES" \
    --count "$COUNT" \
    --batch_size "$BATCH_SIZE" \
    --output_dir "$OUTPUT_DIR" \
    --selection_strategy "$SELECTION_STRATEGY" \
    --output_prefix "gpt_agent_$(date +%Y%m%d_%H%M%S)"

echo ""
echo "============================================================"
echo "✅ Generation complete!"
echo ""
echo "📁 Output files:"
echo "  - Training data: $OUTPUT_DIR/*_best_responses.jsonl"
echo "  - Statistics:    $OUTPUT_DIR/*_stats.json"
echo ""

# Ask if user wants to analyze the results
read -p "Would you like to analyze the results? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Find the most recent output file
    LATEST_JSONL=$(ls -t "$OUTPUT_DIR"/*_best_responses.jsonl 2>/dev/null | head -1)
    
    if [ -n "$LATEST_JSONL" ]; then
        echo ""
        echo "📊 Analyzing: $LATEST_JSONL"
        echo "============================================================"
        python scripts/analyze_gpt_training_data.py "$LATEST_JSONL"
    else
        echo "❌ No output files found to analyze"
    fi
fi

echo ""
echo "Done! 🎉"



