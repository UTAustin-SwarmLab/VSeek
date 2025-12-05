#!/bin/bash

# Common parameters
BATCH_SIZE=32
TOPK=4
MODEL_PATH="checkpoints/vseek/qwen3-4bt_vl_lvb-emreward-vllm-wtool-tag-v2/global_step_400/actor/huggingface/"
OUTPUT_BASE="./results/window_8_topk_4_vllm_q3_4bt/"

# Datasets to process
# Format: "DatasetName|ParquetPath|OutputPrefix"
DATASETS=(
    "LVBench|/nas/mars/dataset/LVBench/window_8/tag|lvbench"
    "Video-MME|/nas/mars/dataset/Video-MME/window_8/tag|videomme"
    "LVB|/nas/mars/dataset/longvideobench/window_8/tag|lvb"
)

for entry in "${DATASETS[@]}"; do
    IFS="|" read -r name parquet_path prefix <<< "$entry"
    
    echo "Running evaluation for $name..."
    python3 scripts/run_agent_data_vllmrollout.py \
        --parquet "$parquet_path" \
        --batch_size "$BATCH_SIZE" \
        --output_dir "$OUTPUT_BASE" \
        --topk "$TOPK" \
        --local_model_path="$MODEL_PATH" \
        --output_prefix "$prefix"
        
    echo "Finished $name"
    echo "-----------------------------------"
done
