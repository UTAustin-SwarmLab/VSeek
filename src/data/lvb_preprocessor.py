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
        "\n\nUse tools to search for scenes in the video to answer the multiple choice question and decide the correct option."+
        question_text +
        "\nOptions: \n" +
        options_block

    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a video analysis assistant that would answer the user's question with access to a tool-based retrieval system to retrieve the relevant frames of interest from a video. \n"
                "INSTRUCTIONS:\n"
                "1) Read the user's question carefully and the options provided. You will not be able to answer the question directly. You need to use the tool-based retrieval system to retrieve the relevant frames of interest from the video.\n"
                "2) At each step, based on the frames obtained so far, decide whether you can answer directly. If you don't have any frames, you must use the search tool.\n"
                "3) Think inside <think> and </think>. If more information is needed, call the search tool using <tool_call> and </tool_call>.\n"
                "4) Tool usage (JSON inside <tool_call>): {\"name\": \"video_search\", \"arguments\": { ... }}\n"
                "   - Use exactly one argument per call: either \"query\" (language search) OR \"subtitle\" (subtitle match).\n"
                "   Example (language): <tool_call>\n{\"name\": \"video_search\", \"arguments\": {\"query\": \"a chef with a large mixing bowl\", \"mode\": \"base\"}}\n</tool_call>\n"
                "   Example (subtitle): <tool_call>\n{\"name\": \"video_search\", \"arguments\": {\"query\": \"you're interested in.\", \"mode\": \"subtitle\"}}\n</tool_call>\n"
                "5) If you can answer, provide only the option number inside <answer> and </answer>.\n"
                "6) Output exactly ONE non-empty field from (<answer> or <tool_call>) per turn, plus a non-empty <think>.\n"
                "7) Options are numbered 0..N-1; answer with only a single number.\n"
                "\nEXAMPLES:\n"
                "EXAMPLE 1 (Language search):\n"
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
    for idx, entry in tqdm(enumerate(raw_entries)):
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

    train_path = os.path.join(local_save_dir, "train.parquet")
    test_path = os.path.join(local_save_dir, "test.parquet")

    train_ds.to_parquet(train_path)
    test_ds.to_parquet(test_path)

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_save_dir, dst=args.hdfs_dir)
