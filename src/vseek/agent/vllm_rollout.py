from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any, Optional

from verl.experimental.agent_loop.agent_loop import register
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.tools.schemas import ToolResponse
from verl.utils.profiler import simple_timer


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

class VSeekToolAgentLoop(ToolAgentLoop):
        
    async def _call_tool(
        self, tool_call: FunctionCall, tools_kwargs: dict[str, Any]
    ) -> tuple[ToolResponse, float, dict]:
        """Call tool and return tool response."""
        tool, instance_id = None, None
        try:
            # TODO: append malformed tool_call to the prompt: invalid function name or arguments
            tool_name = tool_call.name
            tool_args = json.loads(tool_call.arguments)
            tool = self.tools[tool_name]
            kwargs = tools_kwargs.get(tool_name, {})
            instance_id, _ = await tool.create(create_kwargs=kwargs.get("create_kwargs", {}))
            execute_kwargs = kwargs.get("execute_kwargs", {})
            tool_execution_response, tool_reward, res = await tool.execute(instance_id, tool_args, **execute_kwargs)
        except Exception as e:
            logger.warning(f"Error when executing tool: {e}")
            return (
                ToolResponse(
                    text=f"Error when executing tool: {e}",
                ),
                0.0,
                {},
            )
        finally:
            if tool and instance_id:
                await tool.release(instance_id)

        tool_response_text = tool_execution_response.text
        if tool_response_text and len(tool_response_text) > self.max_tool_response_length:
            if self.tool_response_truncate_side == "left":
                tool_response_text = tool_response_text[: self.max_tool_response_length] + "...(truncated)"
            elif self.tool_response_truncate_side == "right":
                tool_response_text = "(truncated)..." + tool_response_text[-self.max_tool_response_length :]
            else:
                length = self.max_tool_response_length // 2
                tool_response_text = tool_response_text[:length] + "...(truncated)..." + tool_response_text[-length:]

        # Create ToolResponse from tool execution result
        tool_response_kwargs = {"text": tool_response_text}

        # Add multimedia data if present
        for attr_name in ["image", "video"]:
            if hasattr(tool_execution_response, attr_name):
                attr_value = getattr(tool_execution_response, attr_name)
                if attr_value is not None:
                    tool_response_kwargs[attr_name] = attr_value

        return ToolResponse(**tool_response_kwargs), tool_reward, res

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        """Handle the processing tools state with multi-image support per tool turn."""
        add_messages: list[dict[str, Any]] = []
        new_images_this_turn: list[Any] = []

        tasks = []
        for tool_call in agent_data.tool_calls[: self.max_parallel_calls]:
            tasks.append(self._call_tool(tool_call, agent_data.tools_kwargs))

        with simple_timer("tool_calls", agent_data.metrics):
            responses = await asyncio.gather(*tasks)

        for tool_response, tool_reward, _ in responses:
            if tool_response.image or tool_response.video:
                if not getattr(self.processor, "image_processor", None):
                    raise ValueError(
                        "Multimedia data can only be processed by `processor`, but the processor is None. "
                        "This error is often caused if you are using a LLM model but your tool returns multimodal "
                        "data. Plase use a vlm as the base model."
                    )
                content: list[dict[str, Any]] = []
                if tool_response.image:
                    for _ in tool_response.image:
                        content.append({"type": "image"})
                if tool_response.video:
                    for _ in tool_response.video:
                        content.append({"type": "video"})
                if tool_response.text:
                    content.append({"type": "text", "text": tool_response.text})
                message = {"role": "tool", "content": content}
            else:
                message = {"role": "tool", "content": tool_response.text or ""}

            add_messages.append(message)

            if tool_response.image:
                # if agent_data.image_data is None:
                #     agent_data.image_data = []
                # elif not isinstance(agent_data.image_data, list):
                #     agent_data.image_data = [agent_data.image_data]

                if isinstance(tool_response.image, list):
                    # Ensure all elements in the list are valid image objects
                    for img in tool_response.image:
                        if img is not None:  # Add a check to ensure the image is not None
                            #agent_data.image_data.append(img)
                            new_images_this_turn.append(img)  # Using local variable
                else:
                    # Ensure the image is not None
                    if tool_response.image is not None:
                        #agent_data.image_data.append(tool_response.image)
                        new_images_this_turn.append(tool_response.image)  # Using local variable


            if tool_response.video:
                if tool_response.video:
                    logger.warning("Multimedia type 'video' is not currently supported. Only 'image' is supported.")
                    raise NotImplementedError(
                        "Multimedia type 'video' is not currently supported. Only 'image' is supported."
                    )

            if tool_reward is not None:
                agent_data.tool_rewards.append(tool_reward)

        if self.processor is not None:
            raw_tool_response = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    add_messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            # Ensure we pass None (not empty list) when no images
            current_images = new_images_this_turn if new_images_this_turn else None
            model_inputs = self.processor(text=[raw_tool_response], images=current_images, return_tensors="pt")
            response_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            response_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(add_messages, add_generation_prompt=True, tokenize=True),
            )
            response_ids = response_ids[len(self.system_prompt) :]

        # Check length and rollback images/messages if we cannot commit
        if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            return AgentState.TERMINATED

        # Commit messages after ensuring length is safe
        if add_messages:
            agent_data.messages.extend(add_messages)
        
        # Extend image_data only if we have new images
        if len(new_images_this_turn) > 0:
            if agent_data.image_data is None:
                agent_data.image_data = new_images_this_turn
            elif isinstance(agent_data.image_data, list):
                agent_data.image_data.extend(new_images_this_turn)
            else:
                # Convert single image to list and extend
                agent_data.image_data = [agent_data.image_data] + new_images_this_turn

        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        agent_data.user_turns += 1
        return AgentState.GENERATING
    
class _VSeekTagToolParser(ToolParser):
    def __init__(self, tokenizer, tags: list[str]) -> None:
        super().__init__(tokenizer)
        self.tags = tags
        self._compiled = [re.compile(tag, re.DOTALL) for tag in tags]

    async def extract_tool_calls(self, responses_ids: list[int]) -> tuple[str, list[FunctionCall]]:
        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(None, self.tokenizer.decode, responses_ids)

        function_calls: list[FunctionCall] = []
        cleaned_text = text
        #pattern_text = cleaned_text
        pattern_text = cleaned_text.split("</think>")[-1]
        for pattern in self._compiled:
            for match in pattern.finditer(pattern_text):
                query = match.group(1)
                tag_str = pattern.pattern
                if "<search_subtitle>" in tag_str:
                    mode = "subtitle"
                elif "<search>" in tag_str:
                    mode = "base"
                elif "<search_summary>" in tag_str:
                    query = "summary"
                    mode = "summary"
                arguments = json.dumps({"query": query, "mode": mode}, ensure_ascii=False)
                function_calls.append(FunctionCall(name="video_search", arguments=arguments))
                cleaned_text = pattern.sub("", cleaned_text)

        return cleaned_text, function_calls




class _VSeekJSONToolParser(ToolParser):
    def __init__(self, tokenizer) -> None:
        super().__init__(tokenizer)
        self._fenced_json = re.compile(r"```json\s*(\{[\s\S]*?\})\s*```", re.IGNORECASE)
        # Fallback inline matcher for simple {"name": ..., "arguments": {...}}
        self._inline_call_simple = re.compile(
            r"\{[\s\S]*?\"name\"\s*:\s*\"(?P<name>[^\"]+)\"[\s\S]*?\"arguments\"\s*:\s*(?P<args>\{[\s\S]*?\})[\s\S]*?\}",
            re.DOTALL,
        )

    async def extract_tool_calls(self, responses_ids: list[int]) -> tuple[str, list[FunctionCall]]:
        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(None, self.tokenizer.decode, responses_ids)

        function_calls: list[FunctionCall] = []
        cleaned_text = text  # keep original text including json tool calls
        
        def _ensure_arg_str(arguments: Any) -> str:
            if isinstance(arguments, str):
                return arguments
            try:
                return json.dumps(arguments, ensure_ascii=False)
            except Exception:
                return str(arguments)

        def _parse_one(obj: dict[str, Any]):
            if "tool_calls" in obj and isinstance(obj["tool_calls"], list):
                for call in obj["tool_calls"]:
                    # OpenAI format: {"type":"function","id":"...","function":{"name":"...","arguments": "..."}}
                    fn = call.get("function", {})
                    name = fn.get("name") or call.get("name")
                    arguments = fn.get("arguments") if "function" in call else call.get("arguments", {})
                    if name is None:
                        continue
                    function_calls.append(FunctionCall(name=name, arguments=_ensure_arg_str(arguments)))
            elif obj.get("type") == "function" and isinstance(obj.get("function"), dict):
                fn = obj["function"]
                name = fn.get("name")
                arguments = fn.get("arguments", {})
                if name is not None:
                    function_calls.append(FunctionCall(name=name, arguments=_ensure_arg_str(arguments)))
            elif "name" in obj and "arguments" in obj:
                function_calls.append(FunctionCall(name=obj["name"], arguments=_ensure_arg_str(obj["arguments"])))

        for m in list(self._fenced_json.finditer(text)):
            block = m.group(1)
            try:
                obj = json.loads(block)
                _parse_one(obj)
            except Exception:
                continue

        if not function_calls:
            for m in self._inline_call_simple.finditer(text):
                try:
                    name = m.group("name")
                    args = m.group("args")
                    arguments = json.loads(args)
                    function_calls.append(FunctionCall(name=name, arguments=_ensure_arg_str(arguments)))
                except Exception:
                    continue

        return cleaned_text, function_calls

@register("vseek_tag_agent")
class VSeekTagAgentLoop(VSeekToolAgentLoop):
    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        super().init_class(config, tokenizer, processor, **kwargs)
        print("Initializing tag agent")
        tags: list[str] = kwargs.get(
            "tags",
            [r"<search>(.*?)</search>", r"<search_subtitle>(.*?)</search_subtitle>"],
        )
        print(f"Tags: {tags}")
        cls.tool_parser = _VSeekTagToolParser(tokenizer, tags)

@register("vseek_tag_summary_agent")
class VSeekTagSummaryAgentLoop(VSeekToolAgentLoop):
    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        super().init_class(config, tokenizer, processor, **kwargs)
        print("Initializing tag summary agent")
        tags: list[str] = kwargs.get(
            "tags",
            [r"<search>(.*?)</search>", r"<search_subtitle>(.*?)</search_subtitle>", r"<search_summary>(.*?)</search_summary>"],
        )
        print(f"Tags: {tags}")
        cls.tool_parser = _VSeekTagToolParser(tokenizer, tags)
        
@register("vseek_json_agent")
class VSeekJSONAgentLoop(VSeekToolAgentLoop):
    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        super().init_class(config, tokenizer, processor, **kwargs)
        cls.tool_parser = _VSeekJSONToolParser(tokenizer)


