from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
import os
from copy import deepcopy
from json import JSONDecodeError
from typing import Any, Generator, Optional
from uuid import uuid4

import numpy as np
import ray
import sglang.srt.entrypoints.engine
import torch
import torch.distributed as dist
from sglang.srt.managers.io_struct import (
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
    UpdateWeightsFromTensorReqInput,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import (
    assert_pkg_version,
    get_ip,
    get_open_port,
    is_cuda,
    set_prometheus_multiproc_dir,
    set_ulimit,
)
from sglang.srt.weight_sync.utils import update_weights as sgl_update_weights
from tensordict import TensorDict
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, ProcessorMixin

from verl import DataProto
from verl.interactions.base import BaseInteraction
from verl.interactions.utils.interaction_registry import initialize_interactions_from_config
from verl.third_party.sglang import parallel_state as sglang_ps
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionCallSchema, OpenAIFunctionParsedSchema, OpenAIFunctionToolCall
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.net_utils import is_ipv6
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.torch_functional import get_response_mask, pad_sequence_to_length
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.schemas import (
    AsyncRolloutRequest,
    AsyncRolloutRequestStateEnum,
    FinishReasonTypeEnum,
)
from verl.workers.rollout.sglang_rollout.http_server_engine import AsyncHttpServerAdapter
from verl.workers.rollout.sglang_rollout.utils import broadcast_pyobj, get_named_tensor_buckets
from verl.workers.rollout.utils import is_valid_ipv6_address

try:
    from sglang.srt.function_call.function_call_parser import FunctionCallParser
except ImportError:
    from sglang.srt.function_call_parser import FunctionCallParser

try:
    from sglang.srt.entrypoints.openai.protocol import Tool
except ImportError:
    from sglang.srt.openai_api.protocol import Tool


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


from verl.workers.rollout.sglang_rollout.sglang_rollout import SGLangRollout

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    AutoTokenizer,
    GenerationConfig,
)

class VSeekSGLangRollout(SGLangRollout):
    def __init__(self, config: RolloutConfig, model_config: HFModelConfig, device_mesh: DeviceMesh):
        super().__init__(config, model_config, device_mesh)
    
    def _init_distributed_env(self, device_mesh_cpu, **kwargs):
        self._device_mesh_cpu = device_mesh_cpu
        os.environ.setdefault("SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK", "true")
        self.tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert self.tensor_parallel_size <= dist.get_world_size(), (
            "tensor parallel size should be less than or equal to the world size"
        )
        self.train_tp = kwargs.get("train_tp", None)
        if self.train_tp is not None:
            # deployed with megatron
            os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
            os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
            train_tp = kwargs.get("train_tp", None)
            num_tp_per_train_tp = train_tp // self.tensor_parallel_size
            sglang_ps.initialize_parallel_state(
                tensor_model_parallel_size=self.tensor_parallel_size,
                num_tp_per_train_tp=num_tp_per_train_tp,
            )

        tp_size = self.tensor_parallel_size
        world_size = int(os.getenv("WORLD_SIZE", "-1"))

        # init device mesh
        if self._device_mesh_cpu is None:
            device_mesh_kwargs = dict(
                mesh_shape=(world_size // tp_size, tp_size, 1),
                mesh_dim_names=["dp", "tp", "pp"],
            )

            self._device_mesh_cpu = init_device_mesh("cpu", **device_mesh_kwargs)

        self._rank = self._device_mesh_cpu.get_rank()
        self._tp_rank = self._device_mesh_cpu["tp"].get_local_rank()
        self._tp_size = self._device_mesh_cpu["tp"].size()
        if self._rank == 0:
            logger.info(f"_init_distributed_env: :tp_world: {self._tp_size}, global_world: {world_size}")
        # get tp_rank of this process in this tp group
        visible_devices = [None] * self._device_mesh_cpu.size(1)

        torch.distributed.all_gather_object(
            visible_devices, os.environ["CUDA_VISIBLE_DEVICES"], self._device_mesh_cpu.get_group("tp")
        )
        self.visible_devices_set = set(",".join(visible_devices).split(","))
        # os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(sorted(list(self.visible_devices_set)))


    async def _async_rollout_a_request(
        self,
        req: AsyncRolloutRequest,
        do_sample: bool = True,
        is_validate: bool = False,
        **kwargs,
    ) -> AsyncRolloutRequest:
        assert self._tp_rank == 0, "only the master process can call this function"
        _req = deepcopy(req)
        finish_reason_type = None
        output = None

        current_turns = 0
        user_turns = 0
        user_turn_rewards = []

        # Create request-level sampling parameters
        request_sampling_params = self.sampling_params.copy()
        if not do_sample:
            request_sampling_params.update(
                {
                    "n": 1,
                    "presence_penalty": 0.0,
                    "frequency_penalty": 0.0,
                    "repetition_penalty": 1.0,
                    "temperature": 0,
                    "top_p": 1,
                    "top_k": -1,
                    "ignore_eos": False,
                    "min_new_tokens": 0,
                    "max_new_tokens": 512,
                    "skip_special_tokens": True,
                    "spaces_between_special_tokens": True,
                }
            )
        elif is_validate:
            request_sampling_params.update(
                {
                    "top_k": self.config.val_kwargs.top_k,
                    "top_p": self.config.val_kwargs.top_p,
                    "temperature": self.config.val_kwargs.temperature,
                    "n": 1,  # if validate, already repeat in ray_trainer
                    "max_new_tokens": 512
                }
            )

        # Update with any additional kwargs
        request_sampling_params.update(kwargs)

        while current_turns < self.config.multi_turn.max_assistant_turns:
            #print("calling _async_rollout_a_request {}".format(current_turns))
            if _req.state == AsyncRolloutRequestStateEnum.PENDING:
                #print("calling _handle_pending_state")
                await self._handle_pending_state(_req)
                _req.state = AsyncRolloutRequestStateEnum.RUNNING
            elif _req.state == AsyncRolloutRequestStateEnum.TOOL_CALLING:
                #print("calling _handle_tool_call")
                if _req.messages[-1].tool_calls is not None:
                    parsed_tool_calls = _req.messages[-1].tool_calls
                    if self.config.skip_tokenizer_init:
                        _req.messages[-1].tool_calls = None
                    tool_call_results = await asyncio.gather(
                        *[
                            self._tool_map[tool_call.function.name].execute(
                                _req.request_id,
                                tool_call.function.arguments,
                                **_req.tools_kwargs.get(tool_call.function.name, {}).get("execute_kwargs", {}),
                            )
                            for tool_call in parsed_tool_calls
                        ]
                    )
                    _req.add_tool_response_messages(self.processing_class, [resp for resp, _, _ in tool_call_results])
                    for tool_call, (resp, reward, metrics) in zip(parsed_tool_calls, tool_call_results, strict=True):
                        _req.update_metrics(metrics, tool_call.function.name)
                    if len(_req.input_ids) >= self.config.max_model_len:
                        finish_reason_type = FinishReasonTypeEnum.STOP
                        break
                    _req.state = AsyncRolloutRequestStateEnum.RUNNING
                else:
                    raise ValueError(f"Unexpected tool calling last message state: {_req.messages[-1]}")
            elif _req.state == AsyncRolloutRequestStateEnum.RUNNING:
                # Only continue the conversation if the prompt length is not greater than max_model_len - 1,
                # since SGLang raises an error when max_new_tokens + 1 is greater to max_model_len (the extra
                # token accounts for the EOS token).
       
                # TEST: Remove Later
                
                
                prompt_length = len(_req.get_generation_prompt_ids(self.processing_class))
                #print("Compiling Prompt and inference")
                if prompt_length + 1 >= self.config.max_model_len:
                    finish_reason_type = FinishReasonTypeEnum.LENGTH
                    break
                
                # Video support is not implemented yet
                image_data = (
                    _req.multi_modal_data["image"]
                    if _req.multi_modal_data and "image" in _req.multi_modal_data
                    else None
                )
                video_data = (
                    _req.multi_modal_data["video"]
                    if _req.multi_modal_data and "video" in _req.multi_modal_data
                    else None
                )
                if video_data:
                    logger.warning(
                        "video support is not implemented yet, current length of video data is %d", len(video_data)
                    )
           
                output = await self._handle_engine_call(_req, request_sampling_params, image_data=image_data)
                if self.config.skip_tokenizer_init:
                    content_ids = output["output_ids"]
                    content = self.processing_class.decode(content_ids, skip_special_tokens=True)
                    content_ids = torch.tensor(
                        content_ids, dtype=_req.input_ids.dtype, device=_req.input_ids.device
                    ).unsqueeze(0)
                else:
                    content_ids = None
                    content = output["text"]
                #print("output: {}".format(content))
                finish_reason_type = FinishReasonTypeEnum.from_str(output["meta_info"]["finish_reason"]["type"])
                current_turns += 1
                if finish_reason_type == FinishReasonTypeEnum.LENGTH:
                    _req.add_assistant_message(self.processing_class, content=content, content_ids=content_ids)
                    break
                else:
                    #print("Has tool call {}".format(self._function_call_parser.has_tool_call(content)))
                    if self._function_call_parser and self._function_call_parser.has_tool_call(content):
                        finish_reason_type = FinishReasonTypeEnum.TOOL_CALL
                        _req.state = AsyncRolloutRequestStateEnum.TOOL_CALLING
                        
                        try:
                            normed_content, tool_calls = self._function_call_parser.parse_non_stream(content)
                            
                        except JSONDecodeError:
                            logger.error("JSONDecodeError")
                            normed_content = content
                            tool_calls = []
                        except AttributeError:
                            logger.error("AttributeError")
                            normed_content = content
                            tool_calls = []
                        parsed_tool_calls = []
                        for tool_call in tool_calls:
                            function, has_decode_error = OpenAIFunctionCallSchema.from_openai_function_parsed_schema(
                                OpenAIFunctionParsedSchema(
                                    name=tool_call.name,
                                    arguments=tool_call.parameters,
                                )
                            )
                            # Drop the tool call if its arguments has decode error
                            if has_decode_error:
                                continue
                            parsed_tool_calls.append(
                                OpenAIFunctionToolCall(
                                    id=str(tool_call.tool_index),
                                    function=function,
                                )
                            )
                        if len(parsed_tool_calls) > 0:
                            if len(parsed_tool_calls) > self.config.multi_turn.max_parallel_calls:
                                parsed_tool_calls = parsed_tool_calls[:self.config.multi_turn.max_parallel_calls]
                            _req.add_assistant_message(
                                # since the content is updated, we just pass the content not content_ids
                                self.processing_class,
                                content=content,
                                tool_calls=parsed_tool_calls,
                            )
                        else:
                            _req.add_assistant_message(self.processing_class, content=content, content_ids=content_ids)
                            finish_reason_type = FinishReasonTypeEnum.STOP
                            _req.state = AsyncRolloutRequestStateEnum.COMPLETED
                            break
                    else:
                        _req.add_assistant_message(
                            self.processing_class,
                            content=content,
                            content_ids=content_ids,
                        )
                        if (
                            _req.interaction_kwargs
                            and self.interaction_map
                            and user_turns < self.config.multi_turn.max_user_turns
                            and current_turns < self.config.multi_turn.max_assistant_turns
                        ):
                            _req.state = AsyncRolloutRequestStateEnum.INTERACTING
                        else:
                            # Add ending condition
                            finish_reason_type = FinishReasonTypeEnum.STOP
                            _req.state = AsyncRolloutRequestStateEnum.COMPLETED
                            break
            elif _req.state == AsyncRolloutRequestStateEnum.INTERACTING:
                #print("calling _handle_interacting")
                user_turns += 1
                messages = [{"role": x.role, "content": x.content} for x in _req.messages]

                # Get interaction by name from interaction_kwargs
                interaction_name = _req.interaction_kwargs.get(
                    "name", "gsm8k"
                )  # Default to gsm8k for backward compatibility
                if interaction_name not in self.interaction_map:
                    raise ValueError(
                        f"Interaction '{interaction_name}' not found in interaction_map. Available interactions: "
                        f"{list(self.interaction_map.keys())}"
                    )

                interaction = self.interaction_map[interaction_name]

                should_terminate_sequence, content, reward, metrics = await interaction.generate_response(
                    _req.request_id, messages, **_req.interaction_kwargs
                )
                user_turn_rewards.append(reward)
                # Add turn check
                if (
                    should_terminate_sequence
                    or user_turns > self.config.multi_turn.max_user_turns
                    or current_turns > self.config.multi_turn.max_assistant_turns
                ):
                    finish_reason_type = FinishReasonTypeEnum.STOP
                    _req.state = AsyncRolloutRequestStateEnum.COMPLETED
                    break
                else:
                    _req.add_user_message(self.processing_class, content)
                    if len(_req.input_ids) >= self.config.max_model_len:
                        finish_reason_type = FinishReasonTypeEnum.STOP
                        break
                    else:
                        _req.state = AsyncRolloutRequestStateEnum.RUNNING

        if current_turns >= self.config.multi_turn.max_assistant_turns:
            finish_reason_type = FinishReasonTypeEnum.STOP

        # Calculate the reward for each tool
        async def calc_reward_and_release_fn(name: str, tool: BaseTool):
            reward = await tool.calc_reward(_req.request_id, **_req.tools_kwargs[name].get("calc_reward_kwargs", {}))
            await tool.release(_req.request_id, **_req.tools_kwargs[name].get("release_kwargs", {}))
            return name, reward

        tool_reward_tasks = []
        for name in _req.tools_kwargs.keys():
            tool = self._tool_map[name]
            tool_reward_tasks.append(calc_reward_and_release_fn(name, tool))
        tool_reward_scores = await asyncio.gather(*tool_reward_tasks)
        tool_reward_scores = dict(tool_reward_scores)
        all_rewards = {**tool_reward_scores, **{"user_turn_rewards": user_turn_rewards}}
        #print("finalizing request")
        _req.finalize(self.processing_class, all_rewards, finish_reason_type)

        if self.config.calculate_log_probs:
            #print("calculating log probs")
            debug_sampling_params = {**self.sampling_params}
            debug_sampling_params["max_new_tokens"] = 0
            output = await self._engine.async_generate(
                prompt=None,
                input_ids=_req.input_ids,
                sampling_params=debug_sampling_params,
                return_logprob=True,
                logprob_start_len=0,
            )
            # len(input_token_logprobs) = len(input_tokens)-1，because logprob of 1st token is None
            _req.output_token_ids, _req.rollout_log_probs = _extract_logprob_from_output(output)
        return _req
    
    async def _handle_engine_generate(
        self, generation_prompt_ids: list[int], sampling_params: dict, image_data: Optional[list[Any]] = None
    ) -> dict:
        max_new_tokens = min(sampling_params["max_new_tokens"], self.config.max_model_len - len(generation_prompt_ids) - 1)

        kwargs = sampling_params.copy()
        kwargs["max_new_tokens"] = max_new_tokens
        kwargs["n"] = 1  # group size is supported in preprocess
        return_logprob = kwargs.pop("logprobs", False)

        output = await self._engine.async_generate(
            input_ids=generation_prompt_ids,
            sampling_params=kwargs,
            return_logprob=return_logprob,
            image_data=image_data,
        )
        return output
    

class VseekTagToToolParser:
    '''
    This class is used to parse the content to extract tool calls based on the tags.
    The tags are regex strings of the function name that are used to match the content to indicate a tool call.
    The parsed within a regex match would include the arguments for the function call.
    Here we specifically support the <search>(.*?)</search> and <search_subtitle>(.*?)</search_subtitle>  tags
    '''
    def __init__(self, config: RolloutConfig, tags: List[str]):
        self.config = config
    
    def parse(self, content: str) -> list[OpenAIFunctionToolCall]:
        return []
    
    def has_tool_call(self, content: str, ) -> bool:
        import re 
        for tag in self.tags:
            if re.search(tag[0], content):
                return True
        return False
    
    
    def parse_non_stream(self, content: str) -> tuple[str, list[OpenAIFunctionToolCall]]:
        
        # Returns a json object with name, arguments and tool_index
        tool_calls = []
        for tag in self.tags:
            if re.search(tag, content):
                match = re.search(tag, content)
                query = match.group(1)
                if '<search>' in tag:
                    mode = "base"
                elif '<search_subtitle>' in tag:
                    mode = "subtitle"
                tool_calls.append({
                    "name": "video_search",
                    "arguments": 
                        {
                            "query": query,
                            "mode": mode,
                        },
                    "tool_index": str(uuid4()),
                })
        return content, tool_calls
    
    def parse_stream(self, content: str) -> Generator[OpenAIFunctionToolCall, None, None]:
        import re
        for tag in self.tags:
            for match in re.finditer(tag, content):
                query = match.group(1)
                if '<search>' in tag:
                    mode = "base"
                elif '<search_subtitle>' in tag:
                    mode = "subtitle"
                else:
                    mode = "base"
                yield {
                    "name": "video_search",
                    "arguments": {
                        "query": query,
                        "mode": mode,
                    },
                    "tool_index": str(uuid4()),
                }
    
    
    
class VSeekSGLangRolloutTag(SGLangRollout):
    def __init__(self, config: RolloutConfig, model_config: HFModelConfig, device_mesh: DeviceMesh):
        super().__init__(config, model_config, device_mesh)
    
    def _init_distributed_env(self, device_mesh_cpu, **kwargs):
        self._device_mesh_cpu = device_mesh_cpu
        os.environ.setdefault("SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK", "true")
        self.tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert self.tensor_parallel_size <= dist.get_world_size(), (
            "tensor parallel size should be less than or equal to the world size"
        )
        self.train_tp = kwargs.get("train_tp", None)
        if self.train_tp is not None:
            # deployed with megatron
            os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
            os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
            train_tp = kwargs.get("train_tp", None)
            num_tp_per_train_tp = train_tp // self.tensor_parallel_size
            sglang_ps.initialize_parallel_state(
                tensor_model_parallel_size=self.tensor_parallel_size,
                num_tp_per_train_tp=num_tp_per_train_tp,
            )

        tp_size = self.tensor_parallel_size
        world_size = int(os.getenv("WORLD_SIZE", "-1"))

        # init device mesh
        if self._device_mesh_cpu is None:
            device_mesh_kwargs = dict(
                mesh_shape=(world_size // tp_size, tp_size, 1),
                mesh_dim_names=["dp", "tp", "pp"],
            )

            self._device_mesh_cpu = init_device_mesh("cpu", **device_mesh_kwargs)

        self._rank = self._device_mesh_cpu.get_rank()
        self._tp_rank = self._device_mesh_cpu["tp"].get_local_rank()
        self._tp_size = self._device_mesh_cpu["tp"].size()
        if self._rank == 0:
            logger.info(f"_init_distributed_env: :tp_world: {self._tp_size}, global_world: {world_size}")
        # get tp_rank of this process in this tp group
        visible_devices = [None] * self._device_mesh_cpu.size(1)

        torch.distributed.all_gather_object(
            visible_devices, os.environ["CUDA_VISIBLE_DEVICES"], self._device_mesh_cpu.get_group("tp")
        )
        self.visible_devices_set = set(",".join(visible_devices).split(","))
        # os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(sorted(list(self.visible_devices_set)))


    async def _async_rollout_a_request(
        self,
        req: AsyncRolloutRequest,
        do_sample: bool = True,
        is_validate: bool = False,
        **kwargs,
    ) -> AsyncRolloutRequest:
        assert self._tp_rank == 0, "only the master process can call this function"
        _req = deepcopy(req)
        finish_reason_type = None
        output = None

        current_turns = 0
        user_turns = 0
        user_turn_rewards = []
        self.tool_parser = VseekTagToToolParser(self.config, kwargs.get("tags", ["<search>(.*?)</search>", "<search_subtitle>(.*?)</search_subtitle>"]))

        # Create request-level sampling parameters
        request_sampling_params = self.sampling_params.copy()
        if not do_sample:
            request_sampling_params.update(
                {
                    "n": 1,
                    "presence_penalty": 0.0,
                    "frequency_penalty": 0.0,
                    "repetition_penalty": 1.0,
                    "temperature": 0,
                    "top_p": 1,
                    "top_k": -1,
                    "ignore_eos": False,
                    "min_new_tokens": 0,
                    "max_new_tokens": 512,
                    "skip_special_tokens": True,
                    "spaces_between_special_tokens": True,
                }
            )
        elif is_validate:
            request_sampling_params.update(
                {
                    "top_k": self.config.val_kwargs.top_k,
                    "top_p": self.config.val_kwargs.top_p,
                    "temperature": self.config.val_kwargs.temperature,
                    "n": 1,  # if validate, already repeat in ray_trainer
                    "max_new_tokens": 512
                }
            )

        # Update with any additional kwargs
        request_sampling_params.update(kwargs)

        while current_turns < self.config.multi_turn.max_assistant_turns:
            #print("calling _async_rollout_a_request {}".format(current_turns))
            if _req.state == AsyncRolloutRequestStateEnum.PENDING:
                #print("calling _handle_pending_state")
                await self._handle_pending_state(_req)
                _req.state = AsyncRolloutRequestStateEnum.RUNNING
            elif _req.state == AsyncRolloutRequestStateEnum.TOOL_CALLING:
                #print("calling _handle_tool_call")
                if _req.messages[-1].tool_calls is not None:
                    parsed_tool_calls = _req.messages[-1].tool_calls
                    if self.config.skip_tokenizer_init:
                        _req.messages[-1].tool_calls = None
                    tool_call_results = await asyncio.gather(
                        *[
                            self._tool_map[tool_call.function.name].execute(
                                _req.request_id,
                                tool_call.function.arguments,
                                **_req.tools_kwargs.get(tool_call.function.name, {}).get("execute_kwargs", {}),
                            )
                            for tool_call in parsed_tool_calls
                        ]
                    )
                    _req.add_tool_response_messages(self.processing_class, [resp for resp, _, _ in tool_call_results])
                    for tool_call, (resp, reward, metrics) in zip(parsed_tool_calls, tool_call_results, strict=True):
                        _req.update_metrics(metrics, tool_call.function.name)
                    if len(_req.input_ids) >= self.config.max_model_len:
                        finish_reason_type = FinishReasonTypeEnum.STOP
                        break
                    _req.state = AsyncRolloutRequestStateEnum.RUNNING
                else:
                    raise ValueError(f"Unexpected tool calling last message state: {_req.messages[-1]}")
            elif _req.state == AsyncRolloutRequestStateEnum.RUNNING:
                # Only continue the conversation if the prompt length is not greater than max_model_len - 1,
                # since SGLang raises an error when max_new_tokens + 1 is greater to max_model_len (the extra
                # token accounts for the EOS token).
       
                # TEST: Remove Later
                
                
                prompt_length = len(_req.get_generation_prompt_ids(self.processing_class))
                #print("Compiling Prompt and inference")
                if prompt_length + 1 >= self.config.max_model_len:
                    finish_reason_type = FinishReasonTypeEnum.LENGTH
                    break
                
                # Video support is not implemented yet
                image_data = (
                    _req.multi_modal_data["image"]
                    if _req.multi_modal_data and "image" in _req.multi_modal_data
                    else None
                )
                video_data = (
                    _req.multi_modal_data["video"]
                    if _req.multi_modal_data and "video" in _req.multi_modal_data
                    else None
                )
                if video_data:
                    logger.warning(
                        "video support is not implemented yet, current length of video data is %d", len(video_data)
                    )
           
                output = await self._handle_engine_call(_req, request_sampling_params, image_data=image_data)
                if self.config.skip_tokenizer_init:
                    content_ids = output["output_ids"]
                    content = self.processing_class.decode(content_ids, skip_special_tokens=True)
                    content_ids = torch.tensor(
                        content_ids, dtype=_req.input_ids.dtype, device=_req.input_ids.device
                    ).unsqueeze(0)
                else:
                    content_ids = None
                    content = output["text"]
                #print("output: {}".format(content))
                # Content is obtained, begin processing the content to extract tool calls
                finish_reason_type = FinishReasonTypeEnum.from_str(output["meta_info"]["finish_reason"]["type"])
                current_turns += 1
                if finish_reason_type == FinishReasonTypeEnum.LENGTH:
                    _req.add_assistant_message(self.processing_class, content=content, content_ids=content_ids)
                    break
                else:
                    #print("Has tool call {}".format(self._function_call_parser.has_tool_call(content)))
                    
                    # TODO: Here tool calls are based on the tags, we have an argument specifying what tags to use for the tool calls?
                    
                    if self.tool_parser.has_tool_call(content):
                        finish_reason_type = FinishReasonTypeEnum.TOOL_CALL
                        _req.state = AsyncRolloutRequestStateEnum.TOOL_CALLING
                        
                        try:
                            normed_content, tool_calls = self.tool_parser.parse_non_stream(content)
                        except Exception as e:
                            logger.error("Exception: {}".format(e))
                            normed_content = content
                            tool_calls = []
                        
                        parsed_tool_calls = []
                        for tool_call in tool_calls:
                            function, has_decode_error = OpenAIFunctionCallSchema.from_openai_function_parsed_schema(
                                OpenAIFunctionParsedSchema(
                                    name=tool_call.name,
                                    arguments=tool_call.parameters,
                                )
                            )
                            # Drop the tool call if its arguments has decode error
                            # Converted to OpenAI Function Tool Call Schema
                            if has_decode_error:
                                continue
                            parsed_tool_calls.append(
                                OpenAIFunctionToolCall(
                                    id=str(tool_call.tool_index),
                                    function=function,
                                )
                            )
                        if len(parsed_tool_calls) > 0:
                            if len(parsed_tool_calls) > self.config.multi_turn.max_parallel_calls:
                                parsed_tool_calls = parsed_tool_calls[:self.config.multi_turn.max_parallel_calls]
                            _req.add_assistant_message(
                                # since the content is updated, we just pass the content not content_ids
                                self.processing_class,
                                content=content,
                                tool_calls=parsed_tool_calls,
                            )
                        else:
                            _req.add_assistant_message(self.processing_class, content=content, content_ids=content_ids)
                            finish_reason_type = FinishReasonTypeEnum.STOP
                            _req.state = AsyncRolloutRequestStateEnum.COMPLETED
                            break
                    else:
                        _req.add_assistant_message(
                            self.processing_class,
                            content=content,
                            content_ids=content_ids,
                        )

        if current_turns >= self.config.multi_turn.max_assistant_turns:
            finish_reason_type = FinishReasonTypeEnum.STOP

        # Calculate the reward for each tool
        async def calc_reward_and_release_fn(name: str, tool: BaseTool):
            reward = await tool.calc_reward(_req.request_id, **_req.tools_kwargs[name].get("calc_reward_kwargs", {}))
            await tool.release(_req.request_id, **_req.tools_kwargs[name].get("release_kwargs", {}))
            return name, reward

        tool_reward_tasks = []
        for name in _req.tools_kwargs.keys():
            tool = self._tool_map[name]
            tool_reward_tasks.append(calc_reward_and_release_fn(name, tool))
        tool_reward_scores = await asyncio.gather(*tool_reward_tasks)
        tool_reward_scores = dict(tool_reward_scores)
        all_rewards = {**tool_reward_scores, **{"user_turn_rewards": user_turn_rewards}}
        #print("finalizing request")
        _req.finalize(self.processing_class, all_rewards, finish_reason_type)

        if self.config.calculate_log_probs:
            #print("calculating log probs")
            debug_sampling_params = {**self.sampling_params}
            debug_sampling_params["max_new_tokens"] = 0
            output = await self._engine.async_generate(
                prompt=None,
                input_ids=_req.input_ids,
                sampling_params=debug_sampling_params,
                return_logprob=True,
                logprob_start_len=0,
            )
            # len(input_token_logprobs) = len(input_tokens)-1，because logprob of 1st token is None
            _req.output_token_ids, _req.rollout_log_probs = _extract_logprob_from_output(output)
        return _req
    
    async def _handle_engine_generate(
        self, generation_prompt_ids: list[int], sampling_params: dict, image_data: Optional[list[Any]] = None
    ) -> dict:
        max_new_tokens = min(sampling_params["max_new_tokens"], self.config.max_model_len - len(generation_prompt_ids) - 1)

        kwargs = sampling_params.copy()
        kwargs["max_new_tokens"] = max_new_tokens
        kwargs["n"] = 1  # group size is supported in preprocess
        return_logprob = kwargs.pop("logprobs", False)

        output = await self._engine.async_generate(
            input_ids=generation_prompt_ids,
            sampling_params=kwargs,
            return_logprob=return_logprob,
            image_data=image_data,
        )
        return output