from pathlib import Path
import json
import datetime
import os

from tqdm import tqdm
from vseek.data.exp_io import DataInput
from vseek.data.frame import VideoFrames
from vseek.agent.video_agent import VSeekAgent

# LongVideoBench dataset helper
from data.lvb import LongVideoBench
import hydra
from omegaconf import DictConfig



def build_options_string(candidates: list[str]) -> str:
    lines = [f"{idx + 1}. {text}" for idx, text in enumerate(candidates)]
    return "\n".join(lines)


def parse_answer(answer: str) -> str:
    """Parse the agent's answer to extract the choice number."""
    import re
    if answer is None:
        return ""
    # Try to find a number in the answer
    numbers = re.findall(r'\d+', answer)
    if numbers:
        return numbers[0]  # Return the first number found
    return ""


def calculate_accuracy(results: list[dict]) -> float:
    """Calculate accuracy by comparing predicted vs ground truth choices."""
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
    # Setup logging
    # json_log_path, detailed_log_path = setup_logging()
    print(f"Logging results to:")
    # print(f"  JSON: {json_log_path}")
    # print(f"  Detailed: {detailed_log_path}")
    
    lvb = LongVideoBench(cfg)
    entries = lvb.load_data()

    window_size = cfg.retriever.window_size
    dataset_name = "lvb"
    dir_name = f"{dataset_name}_window_{window_size}"
    data_root = Path(cfg.retriever.index_path).joinpath(dir_name)

    results = []
    results_path = f"{cfg.inference.output_dir}/{cfg.dataset.name}_{cfg.retriever.window_size}"
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
    
    for entry in tqdm(remaining_entries, desc="Processing LVB entries"):
        video_id = entry["metadata"]["video_id"]
        pkl_path = data_root.joinpath(f"{video_id}")
        if not pkl_path.exists():
            # Skip if the preprocessed frames are not available
            # You can generate them via LongVideoBench.save_it_as_vseek_data()
            print(f"[skip] Missing preprocessed frames: {pkl_path}")
            continue

        video_frames = VideoFrames.load(str(pkl_path))

        question = entry["question"]
        candidates = entry["candidates"]
        correct_choice = entry["correct_choice"]
        print(f"Question: {question}")
        print(f"Correct choice: {correct_choice}")
        print(f"Candidates: {candidates}")
        options_str = build_options_string(candidates)

        data_input = DataInput(
            video=video_frames,
            question=question,
            options=options_str,
            answer=str(correct_choice),
            video_id=video_id,
        )

        vlm_client = VSeekAgent(
            config=cfg,

        )

        trajectory = vlm_client.run(data_input)
        pred = trajectory.answer

        result = {
            "video_id": video_id,
            "question_id": entry['metadata']['id'],
            "pred": pred,
            "gt": str(correct_choice),
            "parsed_pred": parse_answer(pred),
        }
        results.append(result)
        
        # Log each result immediately
        # log_result(json_log_path, detailed_log_path, result, question, candidates, options_str)
        results_path = f"{cfg.dataset.name}_{cfg.retriever.window_size}"
        write_results(results, f"{cfg.inference.output_dir}/{results_path}/agent_results.json")
        print(f"Video {video_id}: Pred='{pred}' -> Parsed='{result['parsed_pred']}', GT='{correct_choice}'")

    # Calculate and display accuracy
    accuracy = calculate_accuracy(results)
    print(f"\nFinished. Total evaluated: {len(results)}")
    print(f"Accuracy: {accuracy:.3f} ({sum(1 for r in results if r['parsed_pred'] == r['gt'])}/{len(results)})")
    
    # Log final summary
    # log_summary(json_log_path, detailed_log_path, results, accuracy)
    # print(f"\nResults logged to:")
    # print(f"  JSON: {json_log_path}")
    # print(f"  Detailed: {detailed_log_path}")


if __name__ == "__main__":
    main()
