python3 src/vseek/puls/propose_refine_puls_with_gpt.py \
  --input_json  "/nas/mars/dataset/longvideobench/LongVideoBench/puls_new.json" \
  --output_json  "/nas/mars/dataset/longvideobench/LongVideoBench/puls_refined.json" \
  --output_jsonl  "/home/hg22723/projects/VSeek-R1/tmp/lvb_puls_eval_improve_debug_gpt52.jsonl" \
  --summary_json "/home/hg22723/projects/VSeek-R1/tmp/lvb_puls_eval_smoke_summary_improve_gpt52.json" \
  --proposer_model gpt-5.2 \
  --evaluator_model gpt-5.2 \
  --num_workers 32

python3 src/vseek/puls/propose_refine_puls_with_gpt.py \
  --input_json  "/nas/mars/dataset/MLVU/MLVU/puls_new.json" \
  --output_json  "/nas/mars/dataset/MLVU/MLVU/puls_refined.json" \
  --output_jsonl  "/home/hg22723/projects/VSeek-R1/tmp/mlvu_puls_eval_improve_debug_gpt52.jsonl" \
  --summary_json "/home/hg22723/projects/VSeek-R1/tmp/mlvu_puls_eval_smoke_summary_improve_gpt52.json" \
  --proposer_model gpt-5.2 \
  --evaluator_model gpt-5.2 \
  --num_workers 32

python3 src/vseek/puls/propose_refine_puls_with_gpt.py \
  --input_json  "/nas/mars/dataset/Video-MME/puls_new.json" \
  --output_json  "/nas/mars/dataset/Video-MME/puls_refined.json" \
  --output_jsonl  "/home/hg22723/projects/VSeek-R1/tmp/videomme_puls_eval_improve_debug_gpt52.jsonl" \
  --summary_json "/home/hg22723/projects/VSeek-R1/tmp/videomme_puls_eval_smoke_summary_improve_gpt52.json" \
  --proposer_model gpt-5.2 \
  --evaluator_model gpt-5.2 \
  --num_workers 32

python3 src/vseek/puls/propose_refine_puls_with_gpt.py \
  --input_json  "/nas/mars/dataset/CGBench/puls_new.json" \
  --output_json  "/nas/mars/dataset/CGBench/puls_refined.json" \
  --output_jsonl  "/home/hg22723/projects/VSeek-R1/tmp/cgbench_puls_eval_improve_debug_gpt52.jsonl" \
  --summary_json "/home/hg22723/projects/VSeek-R1/tmp/cgbench_puls_eval_smoke_summary_improve_gpt52.json" \
  --proposer_model gpt-5.2 \
  --evaluator_model gpt-5.2 \
  --num_workers 32

  python3 src/vseek/puls/propose_refine_puls_with_gpt.py \
  --input_json  "/nas/mars/dataset/LVBench/puls_new.json" \
  --output_json  "/nas/mars/dataset/LVBench/puls_refined.json" \
  --output_jsonl  "/home/hg22723/projects/VSeek-R1/tmp/lvbench_puls_eval_improve_debug_gpt52.jsonl" \
  --summary_json "/home/hg22723/projects/VSeek-R1/tmp/lvbench_puls_eval_smoke_summary_improve_gpt52.json" \
  --proposer_model gpt-5.2 \
  --evaluator_model gpt-5.2 \
  --num_workers 32