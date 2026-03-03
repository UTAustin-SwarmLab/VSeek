#!/bin/bash
# VideoTree QA - different ways to run. Uncomment the one you need.
# Requires: OPENAI_API_KEY when using gpt5

# --- 1) Single video + question (GPT-5) ---
# python run_realtime_videotree.py \
#   --video /home/ss99569/code/video-agent/VSeek-R1/datasets/Video-MME/videos/data/_CqKv0Y1FB0.mp4 \
#   --question "What is the main activity in this video?"

# --- 2) MCQ with options and ground truth ---
# python run_realtime_videotree.py \
#   --video /path/to/video.mp4 \
#   --question "What happens at the end?" \
#   --choices "Person leaves" "Person sits down" "Person waves" "Person runs" "Nothing" \
#   --answer "A" \
#   --output result.json

# --- 3) Batch from JSON manifest ---
# Create entries.json: [{"video": "...", "question": "...", "choices": [...], "answer": "A"}]
# python run_realtime_videotree.py \
#   --manifest entries.json \
#   --output results.json

# --- 4) Local Qwen3-VL (OpenAI-compatible server at localhost:8000) ---
# python run_realtime_videotree.py \
#   --video /path/to/video.mp4 \
#   --question "What happens?" \
#   --model-variant qwen4b_instruct

# --- 5) With extra options (more passes, more frames) ---
# python run_realtime_videotree.py \
#   --video /path/to/video.mp4 \
#   --question "What happens?" \
#   --n-passes 8 \
#   --max-frames 1024 \
#   --output result.json

# --- 6) puls_refined.json red socks question (Video-MME fFjv93ACGo8) ---
python run_realtime_videotree.py \
  --video /home/ss99569/code/video-agent/VSeek-R1/datasets/Video-MME/videos/data/fFjv93ACGo8.mp4 \
  --question "How many red socks are above the fireplace at the end of this video?" \
  --choices "A. 1." "B. 4." "C. 2." "D. 3." \
  --answer "D" \
  --output result.json
