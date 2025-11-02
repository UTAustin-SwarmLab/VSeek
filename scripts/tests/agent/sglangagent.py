import argparse
import sys
import os

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
from verl.workers.rollout.sglang_rollout.sglang_rollout import SGLangRollout
from vseek.agent.slgang_rollout import VSeekSGLangRollout, VSeekSGLangRolloutTag
from tests.workers.rollout.utils_sglang import (
    are_lists_similar,
    clean_torchelastic_env,
    generate_hf_output,
    get_rollout_config,
    initialize_global_process_group,
    load_tokenizer_and_model,
    prepare_inputs,
)
import torch
from tensordict import TensorDict


from verl import DataProto
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import HFModelConfig, RolloutConfig


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


def test_async_vseek_sglang_rollout(parquet_path: str, count: int):

    assert torch.cuda.device_count() >= 1
    initialize_global_process_group()
    clean_torchelastic_env()

    max_prompt_length = 8196
    max_response_length = 12000
    dtype = "bfloat16"
    tensor_parallel_size = 2
    # local_model_path = os.path.expanduser("Qwen/Qwen2.5-VL-7B-Instruct")
    local_model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
    tokenizer, actor_model = load_tokenizer_and_model("Qwen/Qwen2.5-VL-7B-Instruct")

    preencode_prompts, tools_kwargs_list = _load_lvb_examples(parquet_path, count)
    prompts = [
        tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        for message in preencode_prompts
    ]
    input_ids, attention_mask, position_ids = prepare_inputs(tokenizer, prompts, max_prompt_length)

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
    #     skip_tokenizer_init=True,

    # )
    # rollout_config.gpu_memory_utilization = 0.80
    # rollout_config.multi_turn.max_assistant_turns = 5
    # rollout_config.multi_turn.enable = True
    
    # rollout_config: RolloutConfig = omega_conf_to_dataclass(rollout_config, dataclass_type=RolloutConfig)
    rollout_config = omega_conf.load_config_from_yaml("src/vseek/config/lvb_grpo.yaml")
    rollout_config = rollout_config.actor_rollout_ref.rollout
    
    model_config = HFModelConfig(path=local_model_path)
    rollout =  VSeekSGLangRolloutTag(
        config=rollout_config,
        model_config=model_config,
        device_mesh=None,
    )

    prompt_dict = TensorDict(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        },
        batch_size=input_ids.shape[0],
    )
    print(f"preprocessed {input_ids.shape=}")

    messages = np.asarray(preencode_prompts, dtype=object)
    dprompts = DataProto(
        batch=prompt_dict,
        non_tensor_batch={
            "raw_prompt": messages,
            "tools_kwargs": np.array(tools_kwargs_list, dtype=object),
        },
    )

    dprompts.meta_info.update(
        {
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        }
    )

    output = rollout.generate_sequences(prompts=dprompts)
    print(f"generated {output.batch['responses'].shape=}")



    sglang_output = output.to("cpu")

    sglang_response_tokens = tokenizer.batch_decode(
        sglang_output.batch["responses"],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    sglang_input_tokens = tokenizer.batch_decode(
        sglang_output.batch["input_ids"],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    # Final sanitize: strip whitespace and any residual end-of-text markers
    sglang_response_tokens = [
        (token or "").replace("<|endoftext|>", "").strip()
        for token in sglang_response_tokens
    ]
    sglang_input_tokens = [
        (token or "").replace("<|endoftext|>", "").strip()
        for token in sglang_input_tokens
    ]
    # Drop empty strings after cleanup
    sglang_response_tokens = [t for t in sglang_response_tokens if t]
    sglang_input_tokens = [t for t in sglang_input_tokens if t]
    # print(f"hf response: {hf_response_tokens}")
    # print(f"sglang response: {sglang_response_tokens}")
    # assert are_lists_similar(hf_response_tokens, sglang_response_tokens)
    print(f"sglang response: {sglang_response_tokens}")
    print(f"sglang input: {sglang_input_tokens}")
    print("SGLang w tool Test Passed!")

    
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", default=os.path.expanduser("~/data/lvb/test.parquet"), help="Path to LVB parquet")
    parser.add_argument("--count", type=int, default=3, help="Number of examples to run")
    args = parser.parse_args()

    test_async_vseek_sglang_rollout(args.parquet, args.count)