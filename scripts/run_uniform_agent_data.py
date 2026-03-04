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
from vseek.agent.video_rag_agent import VideoRAGAgent
# LongVideoBench dataset helper
from data.lvb import LongVideoBench
from data.lvbench import LVBench
from data.videomme import VideoMME
from data.mlvu import MLVU
from data.cgbench import CGBench
import hydra
from omegaconf import DictConfig, open_dict


def build_options_string(candidates: list[str], dataset_name: str) -> str:
    if dataset_name == "lvb":
        lines = [f"{idx}. {text}" for idx, text in enumerate(candidates)]
    elif dataset_name == "lvbench":
        lines = [f"{text}" for idx, text in enumerate(candidates)]
    elif dataset_name == "videomme":
        lines = [f"{text}" for idx, text in enumerate(candidates)]
    elif dataset_name == "mlvu":
        lines = [f"{idx}. {text}" for idx, text in enumerate(candidates)]
    elif dataset_name == "cgbench":
        lines = [f"{idx}. {text}" for idx, text in enumerate(candidates)]
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


async def process_entry(entry, agent, data_root, cfg, results, results_file, semaphore, passes=1):
    async with semaphore:
        video_id = entry["metadata"]["video_id"]
        pkl_path = data_root.joinpath(f"{video_id}")
        if not pkl_path.exists():
            print(f"[skip] Missing preprocessed frames: {pkl_path}")
            return None
        all_preds = []
        all_parsed_preds = []
        video_frames = await asyncio.to_thread(VideoFrames.load, str(pkl_path))
        for pass_idx in range(passes):
            try:
                # Offload blocking IO to thread pool
                question = entry["question"]
                candidates = entry["candidates"]
                correct_choice = entry["correct_choice"]
                correct_choice = str(correct_choice)
                options_str = build_options_string(candidates, cfg.dataset.name)
                print(options_str)
                data_input = DataInput(
                    video=video_frames,
                    question=question,
                    options=options_str,
                    answer=correct_choice,
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
                
                # For VideoRAGAgent, the return type might need checking, but assuming it returns similar structure
                if trajectory is None:
                    print(f"Warning: agent returned None for {video_id}")
                    return None

                pred = trajectory.answer
                agent_prompt_type = cfg.inference.get("agent_prompt_type", "base")
                all_preds.append(pred)
                all_parsed_preds.append(parse_answer_base(pred) if agent_prompt_type == "base" else parse_answer_cot(pred))
            except Exception as e:
                print(f"Failed to process video {video_id}: {e}")
                import traceback
                traceback.print_exc()
                return None

        # Simple majority vote
        from collections import Counter
        counts = Counter([p for p in all_parsed_preds if p]) # Filter empty? Or count empty as wrong?
        # If all are empty, majority is empty
        if not counts:
            majority_vote = ""
        else:
            majority_vote = counts.most_common(1)[0][0]

        result = {
            "video_id": video_id,
            "question_id": entry['metadata']['id'],
            "all_preds": all_preds,
            "gt": correct_choice,
            "all_parsed_preds": all_parsed_preds,
            "majority_vote": majority_vote,
            "is_majority_correct": (majority_vote == correct_choice),
            "correct_count": sum(1 for p in all_parsed_preds if p == correct_choice),
            "total_passes": passes
        }
        
        print(f"Video {video_id}: Majority='{majority_vote}', GT='{correct_choice}', Correct={result['correct_count']}/{passes}")
        return result


async def run_async(cfg: DictConfig):
    if cfg.dataset.name == "lvb":
        dataset = LongVideoBench(cfg)
    elif cfg.dataset.name == "lvbench":
        dataset = LVBench(cfg)
    elif cfg.dataset.name == "videomme":
        dataset = VideoMME(cfg)
    elif cfg.dataset.name == "mlvu":
        dataset = MLVU(cfg)
    elif cfg.dataset.name == "cgbench":
        dataset = CGBench(cfg)
    else:
        raise ValueError(f"Unsupported dataset: {cfg.dataset.name}")
    
    # Load data (blocking, but happens once)
    entries = await asyncio.to_thread(dataset.load_data)

    window_size = cfg.retriever.window_size
    dir_name = f"{cfg.dataset.name}_window_{window_size}"
    data_root = Path(cfg.retriever.index_path).joinpath(dir_name)

    results = []
    results_path = f"{cfg.inference.output_dir}"
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

    # Prioritize root-level agent_type (e.g. from +agent_type=videorag)
    # Then check inference.agent_type (e.g. from config file or +inference.agent_type=videorag)
    if "agent_type" in cfg:
        agent_type = cfg.agent_type
    else:
        agent_type = cfg.inference.get("agent_type", "uniform")
    
    print(f"Initializing agent type: {agent_type}")
    
    if agent_type == "uniform":
        agent = UniformSampleAgent(config=cfg)
    elif agent_type == "videorag":
        agent = VideoRAGAgent(config=cfg)
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")
    
    # Limit concurrency
    batch_size = getattr(cfg.inference, "batch_size", 16)
    semaphore = asyncio.Semaphore(batch_size)
    
    tasks = []
    for entry in remaining_entries:
        tasks.append(process_entry(entry, agent, data_root, cfg, results, results_file, semaphore, passes=cfg.inference.passes))
    
    # Process tasks with progress bar
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing LVB entries (async)"):
        result = await coro
        if result:
            results.append(result)
            # Write results periodically or after each
            # Note: writing whole list repeatedly is inefficient for large lists but keeps it simple
            write_results(results, results_file)

    import math
    def comb(n, k):
        return math.comb(n, k)
    
    def pass_at_k_estimator(n, c, k):
        if n < k: return 0.0 # Should not happen if we filter ks
        if n - c < k: return 1.0
        return 1.0 - (comb(n-c, k) / comb(n, k))
    
    print(f"Results for {agent_type} on {cfg.dataset.name} with {cfg.inference.passes} passes and {cfg.inference.max_images_per_turn} frames per turn and {cfg.inference.agent_prompt_type} prompt type:")
    total = len(results)
    if total > 0:
        majority_correct = sum(1 for r in results if r.get("is_majority_correct", False))
        majority_acc = majority_correct / total
        
        print(f"\nFinished {agent_type}. Total evaluated: {total} for {cfg.dataset.name}")
        print(f"Majority Vote Accuracy: {majority_acc:.3f} ({majority_correct}/{total})")

        n_passes = getattr(cfg.inference, "passes", 1)
        ks = [1, 2, 4, 5, 8, 10, 16]
        ks = [k for k in ks if k <= n_passes]
        # Include n_passes if not present and > 1
        if n_passes > 1 and n_passes not in ks:
            ks.append(n_passes)
        ks = sorted(list(set(ks)))
        
        for k in ks:
            sum_pass_at_k = 0.0
            for r in results:
                c = r.get("correct_count", 0)
                sum_pass_at_k += pass_at_k_estimator(n_passes, c, k)
            avg_pass_at_k = sum_pass_at_k / total
            print(f"Pass@{k} Accuracy: {avg_pass_at_k:.3f}")
    else:
        print("No results found.")

def _normalize_llm_engine_config(cfg: DictConfig) -> None:
    """
    Normalize engine selection so scripts can switch between:
    - local vLLM engine (default)
    - external OpenAI-compatible vLLM server
    """
    llm_cfg = cfg.llm
    current_engine = str(getattr(llm_cfg, "engine", "vllm")).lower()
    use_external = bool(getattr(llm_cfg, "use_external_vllm", False))

    # Optional env-based override for shell wrappers.
    env_external = os.getenv("USE_EXTERNAL_VLLM")
    if env_external is not None:
        use_external = env_external.strip().lower() in {"1", "true", "yes", "y", "on"}

    env_host = os.getenv("EXTERNAL_VLLM_HOST")
    env_port = os.getenv("EXTERNAL_VLLM_PORT")

    with open_dict(cfg):
        if use_external or current_engine in {"external_vllm", "vllm_server", "server"}:
            cfg.llm.use_external_vllm = True
            cfg.llm.engine = "external_vllm"

            host = env_host or "localhost"
            if env_port:
                cfg.llm.server_url = f"http://{host}:{env_port}/v1"

            server_url = str(getattr(cfg.llm, "server_url", "")).strip()
            if not server_url:
                raise ValueError(
                    "External vLLM mode requires llm.server_url "
                    "(or EXTERNAL_VLLM_PORT/EXTERNAL_VLLM_HOST)."
                )
        else:
            cfg.llm.use_external_vllm = False
            cfg.llm.engine = "vllm"


@hydra.main(version_base=None, config_path="../src/vseek/config/retriever", config_name="config")
def main(cfg: DictConfig):
    _normalize_llm_engine_config(cfg)
    print(
        f"LLM engine mode: {cfg.llm.engine} "
        f"(external={cfg.llm.use_external_vllm}, "
        f"server_url={getattr(cfg.llm, 'server_url', 'N/A')})"
    )
    asyncio.run(run_async(cfg))

if __name__ == "__main__":
    main()


