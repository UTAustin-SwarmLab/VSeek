from pathlib import Path
import json
import os
import multiprocessing
import asyncio
# Set start method to spawn before any CUDA init
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

from tqdm import tqdm
from vseek.data.exp_io import DataInput
from vseek.data.frame import VideoFrames

#from vseek.agent.experimental.uniform_model_mm_tool import ToolUniformSampleAgent as UniformSampleAgent
from vseek.agent.uniform_agent import UniformSampleAgent
# LongVideoBench dataset helper
from data.lvb import LongVideoBench
from data.lvbench import LVBench
from data.videomme import VideoMME
from data.mlvu import MLVU
import hydra
from omegaconf import DictConfig


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


async def process_entry(entry, agent, data_root, cfg, results, results_file, semaphore):
    async with semaphore:
        video_id = entry["metadata"]["video_id"]
        pkl_path = data_root.joinpath(f"{video_id}")
        if not pkl_path.exists():
            print(f"[skip] Missing preprocessed frames: {pkl_path}")
            return None

        try:
            # Offload blocking IO to thread pool
            video_frames = await asyncio.to_thread(VideoFrames.load, str(pkl_path))

            question = entry["question"]
            candidates = entry["candidates"]
            correct_choice = entry["correct_choice"]
            correct_choice = str(correct_choice)
            options_str = build_options_string(candidates, cfg.dataset.name)
            
            data_input = DataInput(
                video=video_frames,
                question=question,
                options=options_str,
                answer=str(correct_choice),
                video_id=video_id,
            )

            # Retry logic for vLLM errors
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    trajectory = await agent.run(data_input)
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    print(f"Error processing {video_id} (attempt {attempt+1}/{max_retries}): {e}. Retrying...")
                    await asyncio.sleep(1 * (attempt + 1)) # Backoff
            
            pred = trajectory.answer
            agent_prompt_type = cfg.inference.get("agent_prompt_type", "base")

            result = {
                "video_id": video_id,
                "question_id": entry['metadata']['id'],
                "pred": pred,
                "gt": str(correct_choice),
                "parsed_pred": parse_answer_base(pred) if agent_prompt_type == "base" else parse_answer_cot(pred),
            }
            
            print(f"Video {video_id}: Pred='{pred}' -> Parsed='{result['parsed_pred']}', GT='{correct_choice}'")
            return result
        except Exception as e:
            print(f"Failed to process video {video_id}: {e}")
            return None

async def run_async(cfg: DictConfig):
    if cfg.dataset.name == "lvb":
        dataset = LongVideoBench(cfg)
    elif cfg.dataset.name == "lvbench":
        dataset = LVBench(cfg)
    elif cfg.dataset.name == "videomme":
        dataset = VideoMME(cfg)
    elif cfg.dataset.name == "mlvu":
        dataset = MLVU(cfg)
    else:
        raise ValueError(f"Unsupported dataset: {cfg.dataset.name}")
    
    # Load data (blocking, but happens once)
    entries = await asyncio.to_thread(dataset.load_data)

    window_size = cfg.retriever.window_size
    dir_name = f"{cfg.dataset.name}_window_{window_size}"
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
    
    # Limit concurrency
    batch_size = getattr(cfg.inference, "batch_size", 16)
    semaphore = asyncio.Semaphore(batch_size)
    
    tasks = []
    for entry in remaining_entries:
        tasks.append(process_entry(entry, agent, data_root, cfg, results, results_file, semaphore))
    
    # Process tasks with progress bar
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing LVB entries (async)"):
        result = await coro
        if result:
            results.append(result)
            # Write results periodically or after each
            # Note: writing whole list repeatedly is inefficient for large lists but keeps it simple
            write_results(results, results_file)

    accuracy = calculate_accuracy(results, cfg.inference.get("agent_prompt_type", "base"))
    print(f"\nFinished. Total evaluated: {len(results)}")
    print(f"Accuracy: {accuracy:.3f} ({sum(1 for r in results if r['parsed_pred'] == r['gt'])}/{len(results)})")

@hydra.main(version_base=None, config_path="../src/vseek/config/retriever", config_name="config")
def main(cfg: DictConfig):
    asyncio.run(run_async(cfg))

if __name__ == "__main__":
    main()


