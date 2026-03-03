python "/home/hg22723/projects/VSeek-R1/src/vseek/puls/test/evaluate_puls_with_gpt.py" \
  --input_json "/nas/mars/dataset/longvideobench/LongVideoBench/puls_refined.json" \
  --output_jsonl "/home/hg22723/projects/VSeek-R1/tmp/lvb_puls_eval_smoke_gpt52.jsonl" \
  --summary_json "/home/hg22723/projects/VSeek-R1/tmp/lvb_puls_eval_smoke_summary_gpt52.json" \
  --model "gpt-5.2" \
  --seed 42 \
  --num_workers 32

python "/home/hg22723/projects/VSeek-R1/src/vseek/puls/test/evaluate_puls_with_gpt.py" \
  --input_json "/nas/mars/dataset/MLVU/MLVU/puls_refined.json" \
  --output_jsonl "/home/hg22723/projects/VSeek-R1/tmp/lvb_puls_eval_smoke_gpt52.jsonl" \
  --summary_json "/home/hg22723/projects/VSeek-R1/tmp/lvb_puls_eval_smoke_summary_gpt52.json" \
  --model "gpt-5.2" \
  --seed 42 \
  --num_workers 32

python "/home/hg22723/projects/VSeek-R1/src/vseek/puls/test/evaluate_puls_with_gpt.py" \
  --input_json "/nas/mars/dataset/Video-MME/puls_refined.json" \
  --output_jsonl "/home/hg22723/projects/VSeek-R1/tmp/lvb_puls_eval_smoke_gpt52.jsonl" \
  --summary_json "/home/hg22723/projects/VSeek-R1/tmp/lvb_puls_eval_smoke_summary_gpt52.json" \
  --model "gpt-5.2" \
  --seed 42 \
  --num_workers 32