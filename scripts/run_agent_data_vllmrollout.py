import argparse
import sys
import os
import json
import re
import math
from pathlib import Path
import glob
from tqdm import tqdm
from omegaconf import OmegaConf
import ray
import asyncio
# Ensure vendored 'verl' package (at vendor/verl/verl) is importable as top-level 'verl'
VENDORED_VERL_ROOT = "/home/hg22723/projects/VSeek-R1/vendor/verl"
if VENDORED_VERL_ROOT not in sys.path:
    sys.path.insert(0, VENDORED_VERL_ROOT)
import numpy as np
import datasets
from huggingface_hub import snapshot_download

# Ensure project root is importable so `src.*` modules can be resolved
PROJECT_ROOT = "/home/hg22723/projects/VSeek-R1"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from src.vseek.agent.vllm_rollout import *
from tests.workers.rollout.utils_sglang import (
    load_tokenizer_and_model,
    prepare_inputs,
    initialize_global_process_group,
    clean_torchelastic_env,
)
import torch
import torch.distributed as dist
from tensordict import TensorDict
from verl.experimental.agent_loop import AgentLoopManager

from verl import DataProto
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import HFModelConfig, RolloutConfig


from transformers import AutoTokenizer, AutoModelForCausalLM
def _get_rank_and_world():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    # Fallback for single-process runs or early calls
    return int(os.getenv("RANK", 0)), int(os.getenv("WORLD_SIZE", 1))

def _resolve_parquet_files(parquet_path: str, tag: str=None) -> list[str]:
    path = os.path.expanduser(parquet_path)
    if os.path.isdir(path):
        files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".parquet")]
        files.sort()
        if not files:
            raise RuntimeError(f"No parquet files found in directory: {path}")
        return files
    if os.path.isfile(path):
        parent = os.path.dirname(path) or "."
        # Prefer explicit train+val in same directory if present
        if tag is None:
            train_p = os.path.join(parent, "train.parquet")
            val_p = os.path.join(parent, "test.parquet")
        else:
            train_p = os.path.join(parent, f"train_{tag}.parquet")
            val_p = os.path.join(parent, f"test_{tag}.parquet")
        both: list[str] = []
        if os.path.isfile(train_p):
            both.append(train_p)
        if os.path.isfile(val_p):
            both.append(val_p)
        if len(both) >= 2:
            return sorted(both)
        return [path]
    # Treat as glob pattern
    files = glob.glob(path)
    files = [f for f in files if f.endswith('.parquet')]
    files.sort()
    if not files:
        raise RuntimeError(f"No parquet files matched pattern: {parquet_path}")
    return files


def _load_lvb_examples(
    parquet_path: str,
    count: int | None,
    global_count: bool,
    shuffle_seed: int,
    tag: str=None,
):
    parquet_path = os.path.expanduser(parquet_path)
    data_files = _resolve_parquet_files(parquet_path, tag=tag )
    ds = datasets.load_dataset("parquet", data_files=data_files)["train"]
    if len(ds) == 0:
        raise RuntimeError(f"Empty parquet dataset: {parquet_path}")
    print("Total examples: ", len(ds))
    # Same shuffle across ranks for randomized, non-overlapping shards
    ds = ds.shuffle(seed=shuffle_seed)

    rank, world_size = _get_rank_and_world()
    

    # Deterministic sharding across ALL ranks in the torchrun job
    ds = ds.shard(num_shards=world_size, index=rank, contiguous=True)

    # Determine per-rank slice size
    if count is not None and count > 0:
        per_rank = math.ceil(count / world_size) if global_count else count
        n = min(per_rank, len(ds))
        ds = ds.select(range(n))

    examples: list[dict] = []
    for i in range(len(ds)):
        row = ds[i]
        msgs = row.get("prompt") or []
        extra_info = (row.get("extra_info") or {})
        tk = (extra_info.get("tools_kwargs") or {})
        gt = extra_info.get("correct_choice")
        metadata = extra_info.get("metadata") or {}
        examples.append(
            {
                "messages": list(msgs),
                "tools_kwargs": tk,
                "gt": None if gt is None else str(gt),
                "video_id": metadata.get("video_id"),
            }
        )
        
    print("Rank ", rank, " has ", len(examples), " examples")
    return examples

def parse_answer(answer: str) -> str:
    if not answer:
        return ""
    # Prefer content within <answer>...</answer>
    tag_matches = re.findall(r"<\s*answer\s*>([\s\S]*?)<\s*/\s*answer\s*>", answer, flags=re.IGNORECASE)
    target_text = tag_matches[-1].strip() if tag_matches else answer
    # find numbers or letters
    numbers = re.findall(r"\d+", target_text)
    letters = re.findall(r"[a-zA-Z]+", target_text)
    
    return numbers[0] if numbers else letters[0] if letters else ""

def calculate_accuracy(results: list[dict]) -> float:
    if not results:
        return 0.0
    correct = 0
    for r in results:
        pred = parse_answer(r.get("pred"))
        gt = str(r.get("gt")) if r.get("gt") is not None else ""
        if pred == gt and gt != "":
            correct += 1
    return correct / len(results)


def rollout_agent_data(
    parquet_path: str,
    local_model_path: str,
    max_prompt_length: int,
    max_response_length: int,
    batch_size: int,
    flush_every: int,
    output_dir: str,
    output_prefix: str,
    count: int | None,
    global_count: bool,
    shuffle_seed: int,
    topk: int,
    prompt_type: str,
    hf_local_model_path: str,
):
    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "INFO",
                # Use V0 for stable multimodal support; V1 has cache issues with vision models
                "VLLM_USE_V1": "1",
            }
        },
        ignore_reinit_error=True,
    )
    

    assert torch.cuda.device_count() >= 1
    # initialize_global_process_group()
    # clean_torchelastic_env()

    dtype = "bfloat16"
    tensor_parallel_size = 1
    local_model_path = os.path.expanduser(local_model_path)
    # get tokenizer for model
    tokenizer = AutoTokenizer.from_pretrained(hf_local_model_path)

    # Rank-aware loading and optional limiting
    examples = _load_lvb_examples(
        parquet_path=parquet_path,
        count=count,
        global_count=global_count,
        shuffle_seed=shuffle_seed,
        tag=prompt_type,
    )
    rank, world_size = _get_rank_and_world()
    
      # create standalone rollout server

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir.joinpath(f"{output_prefix}_rank{rank}.jsonl")
    pending_results: list[dict] = []
    all_results: list[dict] = []

    # print("Generating HF output")
    # hf_response_tokens = generate_hf_output(actor_model, input_ids, attention_mask, tokenizer, max_response_length)
    # print("HF output generated")
    # decode the hf_response_tokens

    # Basic checks on Hugging Face decoded responses
    # if not isinstance(hf_response_tokens, (list, tuple)) or len(hf_response_tokens) == 0:
    #     raise AssertionError("HF decoding returned no responses")
    # if len(hf_response_tokens) != input_ids.shape[0]:
    #     raise AssertionError(
    #         f"HF response count {len(hf_response_tokens)} != batch size {input_ids.shape[0]}"
    #     )
    # num_empty = sum(
    #     1
    #     for s in hf_response_tokens
    #     if (s is None) or (isinstance(s, str) and s.strip() == "")
    # )
    # if num_empty > 0:
    #     raise AssertionError(
    #         f"HF decoding returned {num_empty} empty responses out of {len(hf_response_tokens)}"
    #     )
    
    # rollout_config = get_rollout_config(
    #     max_response_length,
    #     max_prompt_length,
    #     dtype,
    #     tensor_parallel_size,
    #     "./scripts/tests/agent/test_config.yaml",

    # )
    # rollout_config.gpu_memory_utilization = 0.8
    # rollout_config.multi_turn.max_assistant_turns = 5
    # rollout_config.multi_turn.enable = True
    # rollout_config.multi_turn.max_parallel_calls = 1
    # rollout_config.max_num_batched_tokens = 65536
    # rollout_config.calculate_log_probs = False
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=os.path.abspath("src/vseek/config")):
        rollout_config = compose(
            config_name="lvb_grpo",
        )   
    
    if 'tagsummary' in args.parquet:
        print("Using tagsummary agent")
        rollout_config.actor_rollout_ref.rollout.agent.default_agent_loop = "vseek_tag_summary_agent"
    else:
        print("Using tag agent")
        rollout_config.actor_rollout_ref.rollout.agent.default_agent_loop = "vseek_tag_agent"
        
    rollout_config.actor_rollout_ref.rollout.name = "vllm"
    rollout_config.actor_rollout_ref.rollout.mode = "async"
    rollout_config.actor_rollout_ref.rollout.tensor_model_parallel_size = 1
    rollout_config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns = 3
    rollout_config.actor_rollout_ref.rollout.multi_turn.enable = True
    rollout_config.actor_rollout_ref.rollout.multi_turn.tool_config_path = "./scripts/tests/agent/test_config.yaml"
    rollout_config.actor_rollout_ref.rollout.multi_turn.max_tool_response_length = 1024
    rollout_config.actor_rollout_ref.rollout.max_num_batched_tokens = 65536
    rollout_config.actor_rollout_ref.rollout.temperature = 0.0
    rollout_config.actor_rollout_ref.rollout.n = 1

    rollout_config.actor_rollout_ref.rollout.agent.num_workers = 1
    rollout_config.actor_rollout_ref.rollout.skip_tokenizer_init = True
    rollout_config.actor_rollout_ref.rollout.over_sample_rate = 0.0
    rollout_config.actor_rollout_ref.rollout.prompt_length = max_prompt_length
    rollout_config.actor_rollout_ref.rollout.response_length = max_response_length
    rollout_config.trainer.n_gpus_per_node = 1
    rollout_config.trainer.nnodes = 1
    rollout_config.actor_rollout_ref.model.path = local_model_path
    rollout_config.actor_rollout_ref.rollout.multi_turn.format = "hermes"
    rollout_config.actor_rollout_ref.rollout.gpu_memory_utilization = 0.9
    # model_config = HFModelConfig(path=hf_local_model_path)
    
    
    rollout_config.actor_rollout_ref.rollout.agent.agent_loop_config_path = "./scripts/tests/agent/test_agent.yaml"

    agent_loop_manager = AgentLoopManager(rollout_config)
    # Minibatch inference with periodic writes
    total = len(examples)
    print(f"Rank {rank}/{world_size} processing {total} examples")
    for start in tqdm(range(0, total, batch_size), desc=f"Rank {rank}/{world_size} processing examples"):
        end = min(start + batch_size, total)
        batch = examples[start:end]
        for ex in batch:
            ex["tools_kwargs"]["video_search"]["execute_kwargs"]["topk"] = topk
            
        preencode_prompts = [ex["messages"] for ex in batch]
        tools_kwargs_list = [ex["tools_kwargs"] for ex in batch]
        prompts = [
            tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in preencode_prompts
        ]
        # Sanitize any unintended multimodal placeholders (e.g., <|video_pad|>, <|image_pad|>)
        # that may be injected by some chat templates. Our run is text-only.
        # prompts = [
        #     re.sub(r"<\|vision_start\|>.*?<\|vision_end\|>", "", p, flags=re.DOTALL)
        #     for p in prompts
        # ]
        # prompts = [p.replace("<|video_pad|>", "").replace("<|image_pad|>", "") for p in prompts]
        input_ids, attention_mask, position_ids = prepare_inputs(
            tokenizer, prompts, max_prompt_length
        )

        prompt_dict = TensorDict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=input_ids.shape[0],
        )

        messages_np = np.asarray(preencode_prompts, dtype=object)
        if 'tagsummary' in args.parquet:
            agent_name = "vseek_tag_summary_agent"
        else:
            agent_name = "vseek_tag_agent"
        dprompts = DataProto(
            batch=prompt_dict,
            non_tensor_batch={
                "raw_prompt": messages_np,
                "tools_kwargs": np.array(tools_kwargs_list, dtype=object),
                "agent_name": np.array([agent_name] * len(messages_np)),
                "data_source": np.array(["lvb"] * len(messages_np)),
                "reward_model": np.array([{"style": "rule", "ground_truth": "1.0"}] * len(messages_np)),
            },
        )

        dprompts.meta_info.update(
            {
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.pad_token_id,
            }
        )

        result = agent_loop_manager.generate_sequences(prompts=dprompts)
        responses = result.batch["responses"]
        num_turns = result.non_tensor_batch["__num_turns__"]

        # Build per-example results and enqueue for write
        for i, ex in enumerate(batch):
            pred_text = tokenizer.decode(responses[i], skip_special_tokens=False)
            parsed_pred = parse_answer(pred_text)
            turns = num_turns[i]
            print(f"Parsed pred: {parsed_pred} over {turns} turns")
            result = {
                "video_id": ex.get("video_id"),
                "pred": pred_text,
                "gt": ex.get("gt"),
                "parsed_pred": parsed_pred,
                "question": ex.get("messages"),
                "turns": str(turns),
            }
            pending_results.append(result)
            all_results.append(result)

        # Periodic flush to JSONL
        if len(pending_results) >= flush_every or end == total:
            with open(out_path, "a", encoding="utf-8") as f:
                for r in pending_results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            pending_results.clear()
            acc = calculate_accuracy(all_results)
            num_correct = sum(1 for r in all_results if parse_answer(r.get("pred")) == (r.get("gt") or ""))
            print(
                f"[rank {rank}] Progress {end}/{total} | Accuracy: {acc:.3f} ({num_correct}/{len(all_results)})"
            )

    return None
    # torch.distributed.barrier()
    # torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", default=os.path.expanduser("~/data/lvb/test.parquet"), help="Path to LVB parquet")
    parser.add_argument("--count", type=int, default=0, help="Total or per-rank examples to run (0=all)")
    parser.add_argument("--global_count", action="store_true", help="Interpret --count as global across all ranks")
    parser.add_argument("--shuffle_seed", type=int, default=42, help="Shuffle seed before sharding")
    parser.add_argument("--local_model_path", default="Qwen/Qwen2.5-VL-7B-Instruct", help="Path to local model")
    parser.add_argument("--hf_local_model_path", default=None, help="Path to local model")
    parser.add_argument("--max_prompt_length", type=int, default=2048, help="Max prompt length")
    parser.add_argument("--max_response_length", type=int, default=10240, help="Max response length")
    parser.add_argument("--batch_size", type=int, default=4, help="Minibatch size for decoding")
    parser.add_argument("--flush_every", type=int, default=32, help="Flush results to disk every N examples")
    parser.add_argument("--output_dir", default=os.path.expanduser("~/results/sglang_runs"), help="Directory to store JSONL outputs")
    parser.add_argument("--output_prefix", default="sglang", help="Filename prefix for JSONL outputs")
    parser.add_argument("--topk", type=int, default=4, help="Top-k sampling parameter")
    parser.add_argument("--prompt_type", type=str, default="tag", help="Prompt type: tag or openai or tagsummary")
    args = parser.parse_args()

    if args.hf_local_model_path is None:
        args.hf_local_model_path = args.local_model_path
        
    rollout_agent_data(
        parquet_path=args.parquet,
        local_model_path=args.local_model_path,
        max_prompt_length=args.max_prompt_length,
        max_response_length=args.max_response_length,
        batch_size=args.batch_size,
        flush_every=args.flush_every,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        count=(args.count if args.count and args.count > 0 else None),
        global_count=args.global_count,
        shuffle_seed=args.shuffle_seed,
        topk=args.topk,
        prompt_type=args.prompt_type,
        hf_local_model_path=args.hf_local_model_path,
    )