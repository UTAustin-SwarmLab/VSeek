"""
Preprocess the Video-MME dataset to a Parquet format compatible with RLHFDataset.
This script:
1. Loads the Video-MME dataset from videomme.py
2. Stores videos to /nas/mars/dataset/Video-MME/
3. Indexes videos by window sizes (similar to LVBench)
4. Creates parquet files with formatted questions and answers
"""

import argparse
import os
import random
import json
from tqdm import tqdm
import datasets
from omegaconf import OmegaConf, DictConfig
import shutil
from collections import defaultdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import traceback
import torch
import cv2
import base64

from verl.utils.hdfs_io import copy, makedirs
from data.videomme import VideoMME
from vseek.data.frame import VideoFrames
from data.prompts.prompts import tagbased, openaitooluse, tagbasedsummary



def build_prompt(args, entry: dict) -> list[dict]:
    """Build prompt messages for the entry."""
    question_text: str = entry["question"].strip()
    candidates: list[str] = entry.get("candidates", [])
    paths: dict = entry.get("paths", {})
    
    # Compose user message with embedded tool resources
    options_block = "".join([f"\n{opt}" for idx, opt in enumerate(candidates)]) if candidates else ""
    resources = {
        "raw_video_path": paths.get("raw_video_path"),
        "subtitle_path": paths.get("subtitle_path"),
        "resource_path": paths.get("video_path"),
    }
    question_text = f"Question: {question_text} \n"
    user_content = question_text + "\nOptions: \n" + options_block
    
    # Select system prompt based on prompt type
    if args.prompt_type == "tag":
        system_prompt = tagbased.system_prompt
    elif args.prompt_type == "openai":
        system_prompt = openaitooluse.system_prompt
    elif args.prompt_type == "tagsummary":
        system_prompt = tagbasedsummary.system_prompt
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


def _encode_frame(frame, max_side: int, quality: int) -> str:
    """Encode frame to base64 JPEG string."""
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess Video-MME dataset to create indexed videos and parquet files"
    )
    parser.add_argument(
        "--local_dataset_path",
        required=True,
        help="Root path to store Video-MME data (default: /nas/mars/dataset/Video-MME/)"
    )
    
    parser.add_argument(
        "--burned_path",
        required=True,
        help="Root path to burned Video-MME data with burned videos and subtitles"
    )
    parser.add_argument(
        "--local_save_dir",
        default=os.path.expanduser("~/data/videomme"),
        help="Output directory for parquet files."
    )
    parser.add_argument(
        "--hdfs_dir",
        default=None,
        help="Optional HDFS directory to mirror outputs."
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="Train split ratio (0-1)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting."
    )
    parser.add_argument(
        "--index_path",
        required=True,
        help="Root path to save precomputed VideoFrames index."
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=8,
        help="Window size for VideoFrames index."
    )
    parser.add_argument(
        "--retrieval_model_path",
        type=str,
        default=None,
        help="Path to ViClip retrieval model"
    )
    
    parser.add_argument(
        "--embed_frames",
        action="store_true",
        help="Embed base64 frames per window into parquet rows."
    )
    parser.add_argument(
        "--thumb_max_side",
        type=int,
        default=224,
        help="Max side for thumbnail resize."
    )
    parser.add_argument(
        "--thumb_quality",
        type=int,
        default=85,
        help="JPEG quality for thumbnails (1-100)."
    )
    parser.add_argument(
        "--prompt_type",
        type=str,
        default="tag",
        help="Prompt type: tag or openai or tagsummary"
    )

    parser.add_argument(
        "--gpu_number",
        type=int,
        default=0,
        help="GPU number to use for video indexing"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["index", "parquet", "both"],
        default="both",
        help="Mode: 'index' to only index videos, 'parquet' to only create parquet files, 'both' for both"
    )
    
    args = parser.parse_args()
    
    local_dataset_path = args.local_dataset_path
    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)
    os.makedirs(local_dataset_path, exist_ok=True)
    
    # Create config for VideoMME loader
    cfg: DictConfig = OmegaConf.create({
        "dataset": {
            "videomme": {
                "dataset_path": local_dataset_path,
                "burned_path": args.burned_path,
            },
        },
        "retriever": {
            "window_size": args.window_size,
            "index_path": args.index_path,
            "retrieval_model_path": args.retrieval_model_path,
            "gpu_number": args.gpu_number,
        }
    })
    
    videomme = VideoMME(cfg)
    
    
    # Step 2: Create parquet files (if mode is 'parquet' or 'both')
    print("\n" + "="*80)
    print("STEP 2: Creating parquet files")
    print("="*80)
    
    # Load data
    raw_entries: list[dict] = videomme.load_data()
    print(f"Loaded {len(raw_entries)} entries")
    
    processed_rows: list[dict] = []
    data_source = "videomme"
    
    # Determine if we should embed frames
    data_root = None
    if args.embed_frames and args.index_path is not None:
        dir_name = f"videomme_window_{args.window_size}"
        data_root = os.path.join(args.index_path, dir_name)
        
        if not os.path.exists(data_root):
            print(f"Warning: Index path {data_root} does not exist. Frames will not be embedded.")
            data_root = None
    
    cached_video_frames = {}
    cached_ids = []
    
    # Process each entry
    for idx, entry in tqdm(enumerate(raw_entries), desc="Processing entries for parquet"):
        try:
            messages = build_prompt(args, entry)
            correct_choice = entry.get("correct_choice", None)
            
            row = {
                "data_source": data_source,
                "prompt": messages,
                "ability": "video_reasoning",
                "reward_model": {
                    "style": "exact_match",
                    "ground_truth": str(correct_choice) if correct_choice is not None else None
                },
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
                                "dataset": "videomme",
                            },
                        },
                    },
                },
            }
            
            # Optionally embed precomputed frames per window
            if data_root is not None:
                try:
                    video_id = entry.get("metadata", {}).get("video_id")
                    vf_path = os.path.join(data_root, str(video_id))
                    
                    if os.path.exists(vf_path):
                        # Use caching to avoid reloading the same video frames
                        if vf_path not in cached_video_frames:
                            video_frames = VideoFrames.load(vf_path)
                            cached_ids.append(vf_path)
                        
                        # Keep cache size limited
                        if len(cached_ids) > 10:
                            pop_id = cached_ids.pop(0)
                            del cached_video_frames[pop_id]
                        
                        if vf_path in cached_video_frames:
                            frames_by_window = cached_video_frames[vf_path]['frames_by_window']
                            encoded_video_summary = cached_video_frames[vf_path]['video_summary']
                        else:
                            frames_by_window = []
                            
                            for window_idx in video_frames.frames_by_window.keys():
                                chunk = video_frames.get_frame_chunk(window_idx)
                                encoded = [
                                    _encode_frame(f, args.thumb_max_side, args.thumb_quality)
                                    for f in chunk
                                ]
                                frames_by_window.append({
                                    "window_idx": window_idx,
                                    "encoded_frames": encoded
                                })
                            
                            # Video summary is uniformly sampled frames
                            video_summary = video_frames.uniformly_sample_frames(args.window_size)
                            encoded_video_summary = [
                                _encode_frame(f, args.thumb_max_side, args.thumb_quality)
                                for f in video_summary
                            ]
                            
                            cached_video_frames[vf_path] = {
                                "frames_by_window": frames_by_window,
                                "video_summary": encoded_video_summary,
                            }
                        
                        row["extra_info"].setdefault("precomputed_frames", [])
                        row["extra_info"]["precomputed_frames"] = frames_by_window
                        row["extra_info"]["tools_kwargs"]["video_search"]["execute_kwargs"]["precomputed_frames"] = frames_by_window
                        
                        # Video summary is uniformly sampled frames
                        row["extra_info"]["tools_kwargs"]["video_search"]["create_kwargs"] = {
                            "video_summary": encoded_video_summary,
                        }
                        
                        print(f"Encoded video summary: {len(encoded_video_summary)}")
                        print(f"Encoded {len(frames_by_window)} frames for video {video_id}")
                        print(f"Total video length: {len(video_frames.all_frames)}")
                        print(f"Total video windows: {len(video_frames.frames_by_window)}")
                except Exception as e:
                    print(f"Error processing video frames for {video_id}: {e}")
                    traceback.print_exc()
            
            processed_rows.append(row)
        
        except Exception as e:
            print(f"Error processing entry {idx}: {e}")
            print(traceback.format_exc())
            continue

    print(f"Processed {len(processed_rows)} rows")

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

    print(f"Train split: {len(train_rows)} rows")
    print(f"Test split: {len(test_rows)} rows")

    # Create datasets
    train_ds = datasets.Dataset.from_list(train_rows)
    test_ds = datasets.Dataset.from_list(test_rows) if test_rows else datasets.Dataset.from_list([])

    # Save parquet files
    output_dir = os.path.join(local_save_dir, f"window_{args.window_size}", args.prompt_type)
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train.parquet")
    test_path = os.path.join(output_dir, "test.parquet")

    train_ds.to_parquet(train_path)
    test_ds.to_parquet(test_path)

    print(f"\nSaved train parquet to: {train_path}")
    print(f"Saved test parquet to: {test_path}")

    # Copy to HDFS if specified
    if args.hdfs_dir is not None:
        print(f"\nCopying to HDFS: {args.hdfs_dir}")
        makedirs(args.hdfs_dir)
        copy(src=local_save_dir, dst=args.hdfs_dir)
        print("HDFS copy complete!")

    print("\n" + "="*80)
    print("Video-MME preprocessing complete!")
    print("="*80)

