import inspect
from typing import Any

from verl import DataProto
from verl.experimental.reward.reward_loop import register
from verl.experimental.reward.reward_loop.base import RewardLoopManagerBase
from verl.utils.reward_score import default_compute_score


@register("vseekrm")
class VSeekRewardLoopManager(RewardLoopManagerBase):
    """
    The VSeek reward loop manager.
    Handles specific logic for 'puls', 'reward_type', and custom data keys.
    """

    def __init__(self, config, tokenizer, compute_score=None, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer)
        self.compute_score = compute_score or default_compute_score
        self.is_async_reward_score = inspect.iscoroutinefunction(self.compute_score)
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer

        # Extract VSeek specific configs, defaulting to standard values if missing
        self.reward_type = config.get("reward_type", "em")
        self.reward_fn_key = config.get("reward_fn_key", "data_source")

    async def run_single(self, data: DataProto) -> dict:
        assert len(data) == 1, "Only support single data item"
        data_item = data[0]
        
        # --- 1. Decode Response ---
        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        response_str = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        )

        # --- 2. Prepare Context (VSeek Specifics) ---
        # Use configured reward_fn_key (default: "data_source")
        data_source = data_item.non_tensor_batch.get(self.reward_fn_key)
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        extra_info = data_item.non_tensor_batch.get("extra_info", {})
        
        # Handle tool fields like Naive, but also specifically extract 'puls'
        tool_extra_fields = data_item.non_tensor_batch.get("tool_extra_fields", {})
        if tool_extra_fields:
            extra_info.update(tool_extra_fields.items())
        
        # VSeek Specific: Explicitly set 'puls' from tool_rewards
        extra_info['puls'] = tool_extra_fields.get("tool_rewards")

        # Standard metadata
        num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
        rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
        extra_info["num_turns"] = num_turns
        extra_info["rollout_reward_scores"] = rollout_reward_scores
        
        # VSeek Specific: Inject reward_type
        extra_info['reward_type'] = self.reward_type

        # --- 3. Compute Score ---
        extra_reward_kwargs = (
            {
                "reward_router_address": self.reward_router_address,
                "reward_model_tokenizer": self.reward_model_tokenizer,
            }
            if self.reward_router_address is not None
            else {}
        )

        if self.is_async_reward_score:
            result = await self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                **extra_reward_kwargs,
            )
        else:
            result = await self.loop.run_in_executor(
                None,
                lambda: self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    **extra_reward_kwargs,
                ),
            )

        # --- 4. Format Output ---
        reward_extra_info = {}
        score: float
        
        if isinstance(result, dict):
            score = result["score"]
            for key, value in result.items():
                reward_extra_info[key] = value
        else:
            score = result
            reward_extra_info["acc"] = score

        return {"reward_score": score, "reward_extra_info": reward_extra_info}