from pathlib import Path
import json
import datetime
import os

from tqdm import tqdm
from vseek.data.exp_io import DataInput
from vseek.data.frame import VideoFrames
from vseek.setting import DataSetting, VLLMSetting
from vseek.agent.video_agent import VSeekAgent

# LongVideoBench dataset helper
from data.lvb import LongVideoBench

VLLM_SETTING = VLLMSetting()
DATA_SETTING = DataSetting()

OUTPUT_DIR = "output"


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





if __name__ == "__main__":
    # Setup logging
    # json_log_path, detailed_log_path = setup_logging()
    print(f"Logging results to:")
    # print(f"  JSON: {json_log_path}")
    # print(f"  Detailed: {detailed_log_path}")
    
    lvb = LongVideoBench()
    entries = lvb.load_data()

    window_size = DATA_SETTING.window_size
    dataset_name = "lvb"
    dir_name = f"{dataset_name}_window_{window_size}"
    data_root = Path(DATA_SETTING.output_dir).joinpath(dir_name)

    results = []

    for entry in tqdm(entries, desc="Processing LVB entries"):
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
        )

        vlm_client = VSeekAgent(
            api_key=VLLM_SETTING.openai_api_key,
            api_base=VLLM_SETTING.api_base,
            model=VLLM_SETTING.model,
            max_image_width=384,
            max_image_height=384,
            image_quality=90,
            temperature=1.0,
        )

        trajectory = vlm_client.run(data_input)
        pred = trajectory.answer

        result = {
            "video_id": video_id,
            "pred": pred,
            "gt": str(correct_choice),
            "parsed_pred": parse_answer(pred),
        }
        results.append(result)
        
        # Log each result immediately
        # log_result(json_log_path, detailed_log_path, result, question, candidates, options_str)
        
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
