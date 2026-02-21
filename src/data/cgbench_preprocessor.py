"""
Preprocess CGBench into parquet format compatible with RLHFDataset.

This script keeps legacy naming (`mvbench_preprocessor.py`) to avoid breaking
existing commands.
"""

import argparse
import os
import random
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import cv2
import datasets
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from data.cgbench import CGBench
from data.prompts.prompts import fanout, openaitooluse, tagbased, tagbasedsummary
from verl.utils.hdfs_io import copy, makedirs
from vseek.data.frame import VideoFrames


def build_prompt(args, entry: dict) -> list[dict]:
    question_text: str = entry["question"].strip()
    candidates: list[str] = entry.get("candidates", [])
    options_block = "".join([f"\n{idx}) {opt}" for idx, opt in enumerate(candidates)]) if candidates else ""
    user_content = question_text + "\nOptions: \n" + options_block

    if args.prompt_type == "tag":
        system_prompt = tagbased.system_prompt
    elif args.prompt_type == "openai":
        system_prompt = openaitooluse.system_prompt
    elif args.prompt_type == "tagsummary":
        system_prompt = tagbasedsummary.system_prompt
    elif args.prompt_type == "fanout":
        system_prompt = fanout.system_prompt
    else:
        raise ValueError(f"Invalid prompt type: {args.prompt_type}")

    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]


def _encode_frame(frame, max_side: int, quality: int) -> bytes:
    height, width = frame.shape[:2]
    scale = max_side / max(height, width) if max(height, width) > max_side else 1.0
    if scale < 1.0:
        frame = cv2.resize(
            frame,
            (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ret:
        raise ValueError("Could not encode frame")
    return buffer.tobytes()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess CGBench to parquet")
    parser.add_argument("--local_dataset_path", required=True, help="Root path to CGBench raw data")
    parser.add_argument("--burned_path", required=True, help="Path to burned CGBench videos")
    parser.add_argument("--local_save_dir", default=os.path.expanduser("~/data/cgbench"), help="Output directory")
    parser.add_argument("--hdfs_dir", default=None, help="Optional HDFS directory")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="Train split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--index_path", default=None, help="Root path to precomputed VideoFrames index")
    parser.add_argument("--window_size", type=int, default=8, help="Window size used in VideoFrames index")
    parser.add_argument("--max_frames_per_turn", type=int, default=16, help="Maximum number of frames per turn")
    parser.add_argument("--embed_frames", action="store_true", help="Embed JPEG frames per window")
    parser.add_argument("--thumb_max_side", type=int, default=224, help="Max side for thumbnail resize")
    parser.add_argument("--thumb_quality", type=int, default=85, help="JPEG quality")
    parser.add_argument("--prompt_type", type=str, default="tag", help="Prompt type")
    args = parser.parse_args()

    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    cfg: DictConfig = OmegaConf.create(
        {
            "dataset": {"cgbench": {"dataset_path": args.local_dataset_path, "burned_path": args.burned_path}},
            "retriever": {"window_size": args.window_size, "index_path": args.index_path},
        }
    )
    cgbench = CGBench(cfg)
    raw_entries: list[dict] = cgbench.load_data()
    data_source = "cgbench"

    data_root = None
    if args.embed_frames and args.index_path:
        data_root = os.path.join(args.index_path, f"cgbench_window_{args.window_size}")
        if not os.path.exists(data_root):
            print(f"Warning: index path not found: {data_root}. Skipping frame embedding.")
            data_root = None

    entries_by_video = defaultdict(list)
    for idx, entry in enumerate(raw_entries):
        video_id = entry.get("metadata", {}).get("video_id")
        entries_by_video[video_id].append((idx, entry))

    def _process_video_group(video_group):
        video_id, idx_entries = video_group
        rows = []

        video_data_cache = None
        if data_root is not None:
            vf_path = os.path.join(data_root, f"{video_id}.pkl")
            if os.path.exists(vf_path):
                try:
                    video_frames = VideoFrames.load(vf_path)
                    frames_by_window = []
                    for window_idx in video_frames.frames_by_window.keys():
                        chunk = video_frames.get_frame_chunk(window_idx)
                        encoded = [_encode_frame(f, args.thumb_max_side, args.thumb_quality) for f in chunk]
                        frames_by_window.append({"window_idx": window_idx, "encoded_frames": encoded})
                    video_summary = video_frames.uniformly_sample_frames(args.max_frames_per_turn)
                    encoded_video_summary = [_encode_frame(f, args.thumb_max_side, args.thumb_quality) for f in video_summary]
                    video_data_cache = {"frames_by_window": frames_by_window, "video_summary": encoded_video_summary}
                except Exception as exc:
                    print(f"Error loading precomputed frames for {video_id}: {exc}")
                    traceback.print_exc()

        for idx, entry in idx_entries:
            try:
                messages = build_prompt(args, entry)
                correct_choice = entry.get("correct_choice")
                row = {
                    "data_source": data_source,
                    "prompt": messages,
                    "ability": "video_reasoning",
                    "reward_model": {"style": "exact_match", "ground_truth": str(correct_choice) if correct_choice is not None else None},
                    "extra_info": {
                        "index": idx,
                        "question": entry.get("question", ""),
                        "candidates": entry.get("candidates", []),
                        "correct_choice": str(correct_choice) if correct_choice is not None else None,
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
                                    "video_id": str(entry.get("metadata", {}).get("video_id")),
                                    "dataset": "cgbench",
                                    "puls": entry.get("puls", {}),
                                }
                            }
                        },
                    },
                }
                if video_data_cache:
                    row["extra_info"]["tools_kwargs"]["video_search"]["execute_kwargs"]["precomputed_frames"] = video_data_cache["frames_by_window"]
                    row["extra_info"]["tools_kwargs"]["video_search"]["execute_kwargs"]["video_summary"] = video_data_cache["video_summary"]
                rows.append(row)
            except Exception as exc:
                print(f"Error processing entry {idx}: {exc}")
                print(traceback.format_exc())
        return rows

    max_workers = int(os.getenv("VSEEK_WORKERS", min(16, os.cpu_count() or 16)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        grouped_results = list(
            tqdm(executor.map(_process_video_group, entries_by_video.items()), total=len(entries_by_video), desc="Processing video groups")
        )

    processed_rows = [row for group in grouped_results for row in group]
    processed_rows.sort(key=lambda x: x["extra_info"]["index"])

    random.seed(args.seed)
    indices = list(range(len(processed_rows)))
    random.shuffle(indices)
    split_idx = int(len(indices) * args.train_ratio)
    train_idx = set(indices[:split_idx])
    train_rows = [processed_rows[i] for i in range(len(processed_rows)) if i in train_idx]
    test_rows = [processed_rows[i] for i in range(len(processed_rows)) if i not in train_idx]

    for r in train_rows:
        r["extra_info"]["split"] = "train"
    for r in test_rows:
        r["extra_info"]["split"] = "test"

    output_dir = os.path.join(local_save_dir, f"window_{args.window_size}", args.prompt_type)
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.parquet")
    test_path = os.path.join(output_dir, "test.parquet")

    datasets.Dataset.from_list(train_rows).to_parquet(train_path)
    if test_rows:
        datasets.Dataset.from_list(test_rows).to_parquet(test_path)
    else:
        datasets.Dataset.from_list([]).to_parquet(test_path)

    if args.hdfs_dir:
        makedirs(args.hdfs_dir)
        copy(src=local_save_dir, dst=args.hdfs_dir)
