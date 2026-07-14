python3 src/vseek/puls/direct_primitives.py \
  --input_json "/nas/mars/dataset/longvideobench/LongVideoBench/puls_new.json" \
  --output_json "/nas/mars/dataset/longvideobench/LongVideoBench/puls_direct.json" \
  --model gpt-5.2 \
  --num_workers 32

python3 src/vseek/puls/direct_primitives.py \
  --input_json "/nas/mars/dataset/MLVU/MLVU/puls_new.json" \
  --output_json "/nas/mars/dataset/MLVU/MLVU/puls_direct.json" \
  --model gpt-5.2 \
  --num_workers 32

python3 src/vseek/puls/direct_primitives.py \
  --input_json "/nas/mars/dataset/Video-MME/puls_new.json" \
  --output_json "/nas/mars/dataset/Video-MME/puls_direct.json" \
  --model gpt-5.2 \
  --num_workers 32