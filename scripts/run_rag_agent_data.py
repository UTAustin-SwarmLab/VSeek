from pathlib import Path

from tqdm import tqdm
from vseek.data.exp_io import DataInput
from vseek.data.frame import VideoFrames
from vseek.agent.video_rag_agent import VideoRAGAgent

# LongVideoBench dataset helper
from data.lvb import LongVideoBench
import hydra
from omegaconf import DictConfig

OUTPUT_DIR = "output"


def build_options_string(candidates: list[str], dataset_name: str) -> str:
    if dataset_name == "lvb":
        lines = [f"{idx}. {text}" for idx, text in enumerate(candidates)]
    elif dataset_name == "lvbench":
        lines = [f"{text}" for idx, text in enumerate(candidates)]
    elif dataset_name == "videomme":
        lines = [f"{text}" for idx, text in enumerate(candidates)]
    elif dataset_name == "mlvu":
        lines = [f"{text}" for idx, text in enumerate(candidates)]
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return "\n".join(lines)


def parse_answer_single(answer: str) -> str:
    import re
    if answer is None:
        return ""
    numbers = re.findall(r"\d+", answer)
    letters = re.findall(r"[a-zA-Z]+", answer)
    if numbers:
        return numbers[0]
    if letters:
        return letters[0]
    return ""

def parse_answer_base(answer: str) -> str:
    import re
    if answer is None:
        return ""
    numbers = re.findall(r"(\d+)", answer)
    letters = re.findall(r"[a-zA-Z]+", answer)
    if letters:
        return letters[0]
    if numbers:
        return numbers[0]
    return ""

def parse_answer_cot(answer: str) -> str:
    import re
    if answer is None:
        return ""
    # numbers = re.findall(r"<\s*answer\s*>([\s\S]*?)<\s*/\s*answer\s*>", answer, flags=re.IGNORECASE)
    numbers = re.findall(r"### (\d+)", answer)
    letters = re.findall(r"### ([a-zA-Z]+)", answer)
    if letters:
        return letters[-1].strip()
    return numbers[-1].strip() if numbers else ""

def calculate_accuracy(results: list[dict], agent_prompt_type: str="base") -> float:
    if not results:
        return 0.0
    correct = 0
    for result in results:
        if agent_prompt_type == "base":
            pred = parse_answer_base(result["pred"])
        elif agent_prompt_type == "cot":
            pred = parse_answer_cot(result["pred"])
        else:
            raise ValueError(f"Invalid agent prompt type: {agent_prompt_type}")
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
            video_id=video_id,
        )
        vlm_client = VideoRAGAgent(
            config=cfg,
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
        print(f"Video {video_id}: Pred='{pred}' -> Parsed='{result['parsed_pred']}', GT='{correct_choice}'")

    # Calculate and display accuracy
    accuracy = calculate_accuracy(results)
    print(f"\nFinished. Total evaluated: {len(results)}")
    print(f"Accuracy: {accuracy:.3f} ({sum(1 for r in results if r['parsed_pred'] == r['gt'])}/{len(results)})")


if __name__ == "__main__":
    main()
