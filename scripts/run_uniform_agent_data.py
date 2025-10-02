from pathlib import Path
import json
import os

from tqdm import tqdm
from vseek.data.exp_io import DataInput
from vseek.data.frame import VideoFrames
from vseek.agent.uniform_agent import UniformSampleAgent

# LongVideoBench dataset helper
from data.lvb import LongVideoBench
import hydra
from omegaconf import DictConfig


def build_options_string(candidates: list[str]) -> str:
    lines = [f"{idx}. {text}" for idx, text in enumerate(candidates)]
    return "\n".join(lines)


def parse_answer(answer: str) -> str:
    import re
    if answer is None:
        return ""
    numbers = re.findall(r"\d+", answer)
    if numbers:
        return numbers[0]
    return ""


def calculate_accuracy(results: list[dict]) -> float:
    if not results:
        return 0.0
    correct = 0
    for result in results:
        pred = parse_answer(result["pred"])
        gt = str(result["gt"])
        if pred == gt:
            correct += 1
    return correct / len(results)


def write_results(results: list[dict], file_path: str):
    with open(file_path, "w") as f:
        json.dump(results, f)


@hydra.main(version_base=None, config_path="../src/vseek/config/retriever", config_name="config")
def main(cfg: DictConfig):
    lvb = LongVideoBench(cfg)
    entries = lvb.load_data()

    window_size = cfg.retriever.window_size
    dataset_name = "lvb"
    dir_name = f"{dataset_name}_window_{window_size}"
    data_root = Path(cfg.retriever.index_path).joinpath(dir_name)

    results = []
    results_path = f"{cfg.inference.output_dir}/{cfg.dataset.name}_{cfg.inference.max_images_per_turn}"
    results_file = f"{results_path}/agent_results.json"
    if not os.path.exists(results_file):
        os.makedirs(results_path, exist_ok=True)
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            results = json.load(f)
    completed_entries = set([result["question_id"] for result in results])

    remaining_entries = []
    for entry in entries:
        if entry["metadata"]["id"] not in completed_entries:
            remaining_entries.append(entry)

    agent = UniformSampleAgent(config=cfg)

    for entry in tqdm(remaining_entries, desc="Processing LVB entries (uniform agent)"):
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

        trajectory = agent.run(data_input)
        pred = trajectory.answer

        result = {
            "video_id": video_id,
            "question_id": entry['metadata']['id'],
            "pred": pred,
            "gt": str(correct_choice),
            "parsed_pred": parse_answer(pred),
        }
        results.append(result)
        write_results(results, results_file)
        print(f"Video {video_id}: Pred='{pred}' -> Parsed='{result['parsed_pred']}', GT='{correct_choice}'")

    accuracy = calculate_accuracy(results)
    print(f"\nFinished. Total evaluated: {len(results)}")
    print(f"Accuracy: {accuracy:.3f} ({sum(1 for r in results if r['parsed_pred'] == r['gt'])}/{len(results)})")


if __name__ == "__main__":
    main()


