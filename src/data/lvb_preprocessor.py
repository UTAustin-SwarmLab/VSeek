"""
Preprocess the LongVideoBench (LVB) dataset to a Parquet format compatible with RLHFDataset.
This is a tool-based setup: prompts include resource paths inside the message content;
there are no multimodal attachments or <video> tokens.
"""

import argparse
import os
import random
import json
from tqdm import tqdm
import datasets
from omegaconf import OmegaConf, DictConfig

from verl.utils.hdfs_io import copy, makedirs
from data.lvb import LongVideoBench
from vseek.data.frame import VideoFrames
import cv2
import base64


def build_prompt(entry: dict) -> list[dict]:
    question_text: str = entry["question"].strip()
    candidates: list[str] = entry.get("candidates", [])
    paths: dict = entry.get("paths", {})

    # Compose user message with embedded tool resources; do not mention videos
    options_block = "".join([f"\n{idx}) {opt}" for idx, opt in enumerate(candidates)]) if candidates else ""
    resources = {
        "raw_video_path": paths.get("raw_video_path"),
        "subtitle_path": paths.get("subtitle_path"),
        "resource_path": paths.get("video_path"),
    }
    question_text = f"Question: {question_text} \n"
    user_content = (
        question_text +
        "\nOptions: \n" +
        options_block
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a video analysis assistant that would aim to answer the user's question over multiple turns. You will not have access to the complete video. You will have access to a tool-based retrieval system to retrieve the relevant frames of interest from a video. \n"
                "Follow these instructions precisely on every turn.\n\n"
                "Turn Checklist:\n"
                "1) Reason: Write your step-by-step reasoning inside <think>...</think>.\n"
                "2) Decide: Based on your reasoning, decide if you have enough information in the frames obtained so far to answer.\n"
                "3) Act (Choose ONE):\n"
                "   - If the answer is NO, output <tool_call>...</tool_call>. You can call the tool upto 3 times every turn. However, you will obtain a fixed number of frames per turn.\n"
                "   - If the answer is YES, output exactly one <answer>...</answer>.\n"
                "4) Final Check: Your output must contain the <think> block and EXACTLY ONE action block (<tool_call> OR <answer>).\n\n"
                "5) Once you answer the question, the trajectory ends.\n\n"
                "Tool Call Specification:\n"
                "- Use JSON strictly inside <tool_call>...</tool_call> with this shape:\n"
                "  {\"name\": \"video_search\", \"arguments\": {\"query\": <string>, \"mode\": \"base\"|\"subtitle\"}}\n"
                "- Choose mode=\"base\" for language based search or mode=\"subtitle\" for subtitle based match to retrieve the frames.\n"
                "- Emit EXACTLY ONE <tool_call> per turn when you need more information; no extra text outside the tags.\n\n"
                "Answer Specification:\n"
                "- When you have enough information, output ONLY the option number inside <answer>...</answer>.\n"
                "- Options are numbered 0..N-1 as mentioned in the question. Do not include any words, just the number.\n\n"
                "Behavioral Rules:\n"
                "- Always include a non-empty <think> block.\n"
                "- Output exactly one of <tool_call> or <answer> on each turn. Never both.\n"
                "- If you lack frames/evidence, use <tool_call> to retrieve them before answering.\n\n"
                "Examples:\n"
                "Turn 1:\n"
                "<think>I should first locate where the chef uses a mixing bowl.</think>\n"
                "<tool_call>\n{\"name\": \"video_search\", \"arguments\": {\"query\": \"a chef with a large mixing bowl\", \"mode\": \"base\"}}\n</tool_call>\n"
                "Turn 2:\n (After the first search, the agent receives frames of the chef placing an empty bowl on the counter.)\n"
                "<think>I saw the bowl but no ingredient yet. I should fetch the next action of adding an ingredient.</think>\n"
                "<tool_call>\n{\"name\": \"video_search\", \"arguments\": {\"query\": \"chef adding an ingredient to the bowl\", \"mode\": \"base\"}}\n</tool_call>\n"
                "Turn 3:\n (After the second search, the agent receives frames of the chef pouring flour into the bowl.)\n"
                "<think>These frames show flour being added. I can answer now.</think>\n"
                "<answer>3</answer>\n"
                "\nEXAMPLE 2 (Temporal reasoning with language search):\n"
                "Turn 1:\n"
                "<think>I need to find when the person picks up the red ball.</think>\n"
                "<tool_call>\n{\"name\": \"video_search\", \"arguments\": {\"query\": \"a person picking up a red ball\", \"mode\": \"base\"}}\n</tool_call>\n"
                "Turn 2:\n (After the first search, the agent receives frames of a person bending over and grabbing a red ball.)\n"
                "<think>The immediate next action is throwing the ball to a dog. I can answer.</think>\n"
                "<answer>1</answer>\n"
                "\nEXAMPLE 3 (Subtitle-guided search):\n"
                "Turn 1:\n"
                "<think>I should locate the moment the subtitle 'you're interested in.' appears.</think>\n"
                "<tool_call>\n{\"name\": \"video_search\", \"arguments\": {\"query\": \"you're interested in.\", \"mode\": \"subtitle\"}}\n</tool_call>\n"
                "Turn 2:\n (After the first search, the agent receives frames of a dark haired woman wearing a hat.)\n"
                "<think>The frames around the subtitle disambiguate nearby objects. I can answer now.</think>\n"
                "<answer>0</answer>\n"
            ),
        },
        {"role": "user", "content": user_content},
    ]
    return messages


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dataset_path", required=True, help="Root path to LVB raw data directory.")
    parser.add_argument("--local_save_dir", default=os.path.expanduser("~/data/lvb"), help="Output directory.")
    parser.add_argument("--hdfs_dir", default=None, help="Optional HDFS directory to mirror outputs.")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="Train split ratio (0-1).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting.")
    parser.add_argument("--burned_path", required=True, help="Root path to LVB burned data directory.")
    parser.add_argument("--index_path", default=None, help="Root path to precomputed VideoFrames index.")
    parser.add_argument("--window_size", type=int, default=8, help="Window size used in VideoFrames index.")
    parser.add_argument("--embed_frames", action="store_true", help="Embed base64 frames per window into parquet rows.")
    parser.add_argument("--thumb_max_side", type=int, default=256, help="Max side for thumbnail resize.")
    parser.add_argument("--thumb_quality", type=int, default=85, help="JPEG quality for thumbnails (1-100).")
    args = parser.parse_args()

    local_dataset_path = args.local_dataset_path
    local_save_dir = os.path.expanduser(args.local_save_dir)
    burned_path = args.burned_path
    os.makedirs(local_save_dir, exist_ok=True)

    # Minimal config for LongVideoBench loader
    cfg: DictConfig = OmegaConf.create(
        {
            "dataset": {
                "dataset_path": local_dataset_path,
                "burned_path": burned_path,
            }
        }
    )

    lvb = LongVideoBench(cfg)
    raw_entries: list[dict] = lvb.load_data()

    processed_rows: list[dict] = []
    data_source = "lvb"
    
    def _encode_frame(frame, max_side: int, quality: int) -> str:
        height, width = frame.shape[:2]
        scale = max_side / max(height, width) if max(height, width) > max_side else 1.0
        if scale < 1.0:
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
        ret, buffer = cv2.imencode(".jpg", frame, encode_params)
        if not ret:
            raise ValueError("Could not encode frame")
        return base64.b64encode(buffer).decode("utf-8")

    data_root = None
    if args.embed_frames and args.index_path is not None:
        dir_name = f"lvb_window_{args.window_size}"
        data_root = os.path.join(args.index_path, dir_name)
    for idx, entry in tqdm(enumerate(raw_entries), desc="Processing LVB entries"):
        messages = build_prompt(entry)
        correct_choice = entry.get("correct_choice", None)
        row = {
            "data_source": data_source,
            "prompt": messages,
            "ability": "video_reasoning",
            # Simple rule-based reward: exact match on option index as string
            "reward_model": {"style": "exact_match", "ground_truth": str(correct_choice) if correct_choice is not None else None},

            "extra_info": {
                "index": idx,
                "question": entry.get("question", ""),
                "candidates": entry.get("candidates", []),
                "correct_choice": correct_choice,
                "metadata": entry.get("metadata", {}),
                "paths": {
                    "raw_video_path": entry.get("paths", {}).get("raw_video_path"),
                    "subtitle_path": entry.get("paths", {}).get("subtitle_path"),
                    "video_path": entry.get("paths", {}).get("video_path"),
                },
                "tools_kwargs": {
                    "video_search": {
                        "execute_kwargs": {
                            "topk": 4,
                            "video_id": entry.get("metadata", {}).get("video_id"),
                        }
                    },
                },
            },
        }
        # Optionally embed precomputed frames per window for this video
        print(f"data_root: {data_root}")
        if data_root is not None:
            try:
                video_id = entry.get("metadata", {}).get("video_id")
                vf_path = os.path.join(data_root, str(video_id))
                video_frames = VideoFrames.load(vf_path)
                frames_by_window = []
                for window_idx in video_frames.frames_by_window.keys():
                    chunk = video_frames.get_frame_chunk(window_idx)
                    encoded = [_encode_frame(f, args.thumb_max_side, args.thumb_quality) for f in chunk]
                    frames_by_window.append({"window_idx": window_idx, "encoded_frames": encoded})

                row["extra_info"].setdefault("precomputed_frames", [])
                row["extra_info"]["precomputed_frames"] = frames_by_window
                
                row["extra_info"]["tools_kwargs"]["video_search"]["execute_kwargs"]["precomputed_frames"] = frames_by_window
                # Ensure Arrow-friendly keys
            except Exception:
                pass

        processed_rows.append(row)

    # Train/test split
    random.seed(args.seed)
    indices = list(range(len(processed_rows)))
    random.shuffle(indices)
    split_point = int(len(indices) * args.train_ratio)
    train_idx = set(indices[:split_point])

    train_rows = [processed_rows[i] for i in range(len(processed_rows)) if i in train_idx]
    test_rows = [processed_rows[i] for i in range(len(processed_rows)) if i not in train_idx]

    # Tag split in extra_info
    for r in train_rows:
        r["extra_info"]["split"] = "train"
    for r in test_rows:
        r["extra_info"]["split"] = "test"

    train_ds = datasets.Dataset.from_list(train_rows)
    test_ds = datasets.Dataset.from_list(test_rows) if test_rows else datasets.Dataset.from_list([])

    train_path = os.path.join(local_save_dir, f"window_{args.window_size}", f"train.parquet")
    test_path = os.path.join(local_save_dir, f"window_{args.window_size}", f"test.parquet")

    train_ds.to_parquet(train_path)
    test_ds.to_parquet(test_path)

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_save_dir, dst=args.hdfs_dir)
