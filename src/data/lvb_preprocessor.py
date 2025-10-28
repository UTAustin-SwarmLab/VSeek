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

from data.prompts.prompts import tagbased, openaitooluse

def build_prompt(args, entry: dict) -> list[dict]:
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
    user_content = "Answer the following multiple choice question: \n" + question_text + "\nOptions: \n" + options_block

    if args.prompt_type == "tag":
        system_prompt = tagbased.system_prompt
    elif args.prompt_type == "openai":
        system_prompt = openaitooluse.system_prompt
    else:
        raise ValueError(f"Invalid prompt type: {args.prompt_type}")
    
    messages = [
        {
            "role": "system",
            "content": system_prompt
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
    parser.add_argument("--thumb_max_side", type=int, default=224, help="Max side for thumbnail resize.")
    parser.add_argument("--thumb_quality", type=int, default=85, help="JPEG quality for thumbnails (1-100).")
    parser.add_argument("--prompt_type", type=str, default="tag", help="Prompt type: tagbased or openai.")
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
    else:
        raise ValueError(f"Invalid index path or embed frames is not enabled: {args.index_path} or {args.embed_frames}")
    for idx, entry in tqdm(enumerate(raw_entries), desc="Processing LVB entries"):
        messages = build_prompt(args, entry)
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
                print(f"Encoded {len(frames_by_window)} frames for video {video_id}")
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

    train_path = os.path.join(local_save_dir, f"window_{args.window_size}", f"train_{args.prompt_type}.parquet")
    test_path = os.path.join(local_save_dir, f"window_{args.window_size}", f"test_{args.prompt_type}.parquet")

    train_ds.to_parquet(train_path)
    test_ds.to_parquet(test_path)

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_save_dir, dst=args.hdfs_dir)
