"""Run VSeek agent inference with UniversalVTG temporal grounding retriever.

This is the same as run_agent_data_vllm.py but pre-wired to use the VTG
config. Requires the VTG server to be running (see scripts/run_vtg_server.sh).

Usage::

    # 1. Start VTG server (in a separate terminal/tmux):
    bash scripts/run_vtg_server.sh 9002 0

    # 2. Start vLLM server for the VLM:
    bash scripts/vllm/run_vllm.sh

    # 3. Run inference:
    python3 scripts/run_agent_data_vtg.py dataset.name=lvb
"""

from pathlib import Path
from datetime import datetime
import glob
import json
import os
import re

from tqdm import tqdm
from vseek.data.exp_io import DataInput
from vseek.data.frame import VideoFrames
from vseek.agent.vtg_agent import VSeekVTGAgent

from data.lvb import LongVideoBench
from data.lvbench import LVBench
from data.videomme import VideoMME
from data.mlvu import MLVU
import hydra
from omegaconf import DictConfig


def build_options_string(candidates: list[str]) -> str:
    lines = [f"{idx}. {text}" for idx, text in enumerate(candidates)]
    return "\n".join(lines)


def parse_answer(answer: str) -> str:
    if answer is None:
        return ""
    numbers = re.findall(r'\d+', answer)
    letters = re.findall(r'[a-zA-Z]+', answer)
    if numbers:
        return numbers[0]
    if letters:
        return letters[0]
    return ""


def calculate_accuracy(results: list[dict]) -> float:
    if not results:
        return 0.0
    correct = sum(1 for r in results if parse_answer(r["pred"]) == str(r["gt"]))
    return correct / len(results)


def write_results(results: list[dict], file_path: str):
    with open(file_path, "w") as f:
        json.dump(results, f)


@hydra.main(version_base=None, config_path="../src/vseek/config/retriever", config_name="config_vtg")
def main(cfg: DictConfig):
    dataset_name = cfg.dataset.name

    if dataset_name == "lvb":
        dataset = LongVideoBench(cfg)
    elif dataset_name == "lvbench":
        dataset = LVBench(cfg)
    elif dataset_name == "videomme":
        dataset = VideoMME(cfg)
    elif dataset_name == "mlvu":
        dataset = MLVU(cfg)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    entries = dataset.load_data()

    window_size = cfg.retriever.window_size
    dir_name = f"{dataset_name}_window_{window_size}"
    data_root = Path(cfg.retriever.index_path).joinpath(dir_name)

    # Filter to test-only: keep entries whose video has PE features
    features_root = Path(cfg.retriever.features_path)
    entries = [
        e for e in entries
        if (features_root / dataset_name / f"{e['metadata']['video_id']}.pt").exists()
    ]
    print(f"Filtered to {len(entries)} entries with PE features (test set)")

    results = []
    results_path = f"{cfg.inference.output_dir}/{dataset_name}_{window_size}_vtg"
    os.makedirs(results_path, exist_ok=True)
    # Resume into the most recent existing timestamped results file if one
    # exists; otherwise start a fresh timestamped file.
    existing = sorted(glob.glob(f"{results_path}/agent_results_*.json"))
    if existing:
        results_file = existing[-1]
        with open(results_file, "r") as f:
            results = json.load(f)
        print(f"Resuming from {results_file} ({len(results)} entries already done)")
    else:
        run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        results_file = f"{results_path}/agent_results_{run_timestamp}.json"
        print(f"Starting fresh run -> {results_file}")
    completed_entries = set(r["question_id"] for r in results)

    remaining_entries = [
        e for e in entries if e["metadata"]["id"] not in completed_entries
    ]

    for entry in tqdm(remaining_entries, desc=f"Processing {dataset_name} (VTG)"):
        video_id = entry["metadata"]["video_id"]
        pkl_path = data_root.joinpath(f"{video_id}")
        if not pkl_path.exists():
            print(f"[skip] Missing preprocessed frames: {pkl_path}")
            continue

        video_frames = VideoFrames.load(str(pkl_path))

        question = entry["question"]
        candidates = entry["candidates"]
        correct_choice = entry["correct_choice"]
        options_str = build_options_string(candidates)

        data_input = DataInput(
            video=video_frames,
            question=question,
            options=options_str,
            answer=str(correct_choice),
            video_id=video_id,
        )

        vlm_client = VSeekVTGAgent(config=cfg)
        trajectory = vlm_client.run(data_input)
        pred = trajectory.answer

        result = {
            "video_id": video_id,
            "question_id": entry["metadata"]["id"],
            "pred": pred,
            "gt": str(correct_choice),
            "parsed_pred": parse_answer(pred),
        }
        results.append(result)

        write_results(results, results_file)
        print(f"Video {video_id}: Pred='{pred}' -> Parsed='{result['parsed_pred']}', GT='{correct_choice}'")

    accuracy = calculate_accuracy(results)
    print(f"\nFinal accuracy: {accuracy:.4f} ({int(accuracy * len(results))}/{len(results)})")


if __name__ == "__main__":
    main()
