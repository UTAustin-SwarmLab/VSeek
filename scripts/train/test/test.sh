CUDA_VISIBLE_DEVICES=0 python3 scripts/run_agent_data_vllmrollout.py \
    --parquet /nas/mars/dataset/Video-MME/window_8/tagsummary \
    --output_prefix vllm_videomme_puls_600 \
    --prompt_type tagsummary \
    --output_dir results/vseek_tagsummary \
    --batch_size 32 \
    --local_model_path checkpoints/vseek/qwen3-4bt_vl_all-pulsreward-vllm-wtool-tagsummary/global_step_600/actor/huggingface \
    > results/vseek_tagsummary/vllm_videomme_puls_600.log 2>&1 &

CUDA_VISIBLE_DEVICES=6 python3 scripts/run_agent_data_vllmrollout.py \
    --parquet /nas/mars/dataset/longvideobench/window_8/tagsummary \
    --output_prefix vllm_lvb_puls_600 \
    --prompt_type tagsummary \
    --max_prompt_length 2536 \
    --output_dir results/vseek_tagsummary \
    --batch_size 32 \
    --local_model_path checkpoints/vseek/qwen3-4bt_vl_all-pulsreward-vllm-wtool-tagsummary/global_step_600/actor/huggingface \
    > results/vseek_tagsummary/vllm_lvb_puls_600.log 2>&1 &

CUDA_VISIBLE_DEVICES=7 python3 scripts/run_agent_data_vllmrollout.py \
    --parquet /nas/mars/dataset/MLVU/window_8/tagsummary \
    --output_prefix vllm_mlvu_puls_600 \
    --prompt_type tagsummary \
    --output_dir results/vseek_tagsummary \
    --batch_size 32 \
    --local_model_path checkpoints/vseek/qwen3-4bt_vl_all-pulsreward-vllm-wtool-tagsummary/global_step_600/actor/huggingface \
    > results/vseek_tagsummary/vllm_mlvu_puls_600.log 2>&1 &

wait