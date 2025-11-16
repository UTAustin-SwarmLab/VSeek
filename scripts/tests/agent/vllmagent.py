
import sys
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
    
from verl.experimental.agent_loop import AgentLoopManager


from verl.utils import hf_tokenizer
import argparse
from omegaconf import DictConfig
import os
import ray
import numpy as np
from tests.workers.rollout.utils_sglang import (
    prepare_inputs,
)
from tensordict import TensorDict
from verl.protocol import DataProto
def _load_lvb_examples(parquet_path: str, count: int):
    parquet_path = os.path.expanduser(parquet_path)
    ds = datasets.load_dataset("parquet", data_files=parquet_path)["train"]
    if len(ds) == 0:
        raise RuntimeError(f"Empty parquet dataset: {parquet_path}")
    n = min(count, len(ds))
    preencode_prompts = []
    tools_kwargs_list = []
    for i in range(n):
        row = ds[i]
        msgs = row.get("prompt") or []
        preencode_prompts.append(list(msgs))
        tk = ((row.get("extra_info") or {}).get("tools_kwargs") or {})
        tools_kwargs_list.append(tk)
    return preencode_prompts, tools_kwargs_list


def init_config(args) -> DictConfig:
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=os.path.abspath("vendor/verl/verl/trainer/config")):
        config = compose(
            config_name="ppo_trainer",
            overrides=[
                "actor_rollout_ref.actor.use_dynamic_bsz=true",
                # keep rollout small and async
                "reward_model.reward_manager=naive",
            ],
        )

    # Required settings for agent loop rollout
    model_path = os.path.expanduser(args.model)
    config.actor_rollout_ref.model.path = model_path
    config.actor_rollout_ref.rollout.name = "vllm"
    config.actor_rollout_ref.rollout.mode = "async"
    config.actor_rollout_ref.rollout.tensor_model_parallel_size = 1
    config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns = 3
    config.actor_rollout_ref.rollout.multi_turn.enable = True
    config.actor_rollout_ref.rollout.multi_turn.tool_config_path = "./scripts/tests/agent/test_config.yaml"
    config.actor_rollout_ref.rollout.multi_turn.max_tool_response_length = 1024
    config.actor_rollout_ref.rollout.max_num_batched_tokens = 32768
    config.actor_rollout_ref.rollout.max_model_len = 8192
    config.actor_rollout_ref.rollout.max_num_seqs = 4096

    
    config.actor_rollout_ref.rollout.enforce_eager = True
    
    config.actor_rollout_ref.rollout.prompt_length = 1280
    config.actor_rollout_ref.rollout.response_length = 4096
    config.actor_rollout_ref.rollout.n = 2
    config.actor_rollout_ref.rollout.agent.num_workers = 1
    config.actor_rollout_ref.rollout.skip_tokenizer_init = True
    config.actor_rollout_ref.rollout.gpu_memory_utilization = 0.5
    config.trainer.n_gpus_per_node = 2
    config.reward_model.reward_manager= "naive"
    config.reward_model.enable = False   

    config.custom_reward_function.path = "src/vseek/trainer/reward/vseekreward.py"
    config.custom_reward_function.name = "compute_score"
    # Required for agent loops to work with datasets
    config.data.return_raw_chat = True

    return config

def test_vllm_agent(init_config, parquet_path: str, count: int):
    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "INFO",
                "VLLM_USE_V1": "1",
            }
        },
        ignore_reinit_error=True,
    )

    # Register our custom agent loop via config file
    agent_loop_config = [
        {
            "_target_": "vseek.agent.vllm_agent_loops.VSeekTagAgentLoop",
            "name": "vseek_tag_agent",
        },
    ]
    agent_loop_config_path = "./scripts/tests/agent/test_agent.yaml"
    # with open(agent_loop_config_path, "w") as f:
    #     json.dump(agent_loop_config, f)

    init_config.actor_rollout_ref.rollout.agent.agent_loop_config_path = agent_loop_config_path

    print("Initializing agent loop manager")
    agent_loop_manager = AgentLoopManager(init_config)

    # Build input batch
    preencode_prompts, tools_kwargs_list = _load_lvb_examples(parquet_path, count)
    # For vLLM agent loop we only need a tokenizer, do not instantiate HF model
    tokenizer = hf_tokenizer(os.path.expanduser(args.model), trust_remote_code=True)
    max_prompt_length = init_config.actor_rollout_ref.rollout.prompt_length
    prompts = [
        tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        for message in preencode_prompts
    ]
    input_ids, attention_mask, position_ids = prepare_inputs(tokenizer, prompts, max_prompt_length)

    raw_prompts = preencode_prompts

    
    print(tools_kwargs_list[0].keys())
    batch = DataProto(
        batch=TensorDict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=input_ids.shape[0],
        ),
        non_tensor_batch={
            "raw_prompt": np.array([np.array(p) for p in raw_prompts], dtype=object),
            "agent_name": np.array(["vseek_tag_agent"] * len(raw_prompts)),
            "data_source": np.array(["lvb"] * len(raw_prompts)),
            "reward_model": np.array([{"style": "rule", "ground_truth": "1.0"}] * len(raw_prompts)),
            "tools_kwargs": np.array(tools_kwargs_list, dtype=object),
        },
    )

    n = init_config.actor_rollout_ref.rollout.n
    batch = batch.repeat(n)

    result = agent_loop_manager.generate_sequences(prompts=batch)
    responses = result.batch["responses"]
    
    
    for i,response in enumerate(responses):
        response_text = tokenizer.decode(response, skip_special_tokens=True)
        input_text = batch.non_tensor_batch["raw_prompt"][i]
        print("=========================")
        print(input_text)
        print("---")
        print(response_text)
        
    # print the response
    
    
    # Basic structural checks
    assert len(result) == len(raw_prompts) * n
    seq_len = result.batch["prompts"].size(1) + result.batch["responses"].size(1)
    # assert result.batch["input_ids"].size(1) == seq_len
    # assert result.batch["attention_mask"].size(1) == seq_len
    # assert result.batch["position_ids"].size(1) == seq_len


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", default=os.path.expanduser("~/data/lvb/test.parquet"), help="Path to LVB parquet")
    parser.add_argument("--count", type=int, default=3, help="Number of examples to run")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct", help="Model path")
    args = parser.parse_args()
    init_config = init_config(args)
    test_vllm_agent(init_config, args.parquet, args.count)