import os
from typing import Any, Dict, List
import uuid
import asyncio
import io
import re

from transformers import AutoProcessor
from vllm import SamplingParams
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
import cv2
import base64
from PIL import Image

class LocalVLLMBase:
    def __init__(self, config) -> None:
        # GPU selection
        self.config = config
        self.gpu_number = getattr(config.inference, "gpu_number", 0)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_number)

        # Model / generation settings
        self.model_path = config.llm.model
        self.is_internvl = "InternVL" in str(self.model_path)
        self.max_image_width = config.inference.max_image_width
        self.max_image_height = config.inference.max_image_height
        self.max_images = getattr(config.inference, "max_images_per_turn", 16)
        self.temperature = float(getattr(config.inference, "temperature", 0.0))
        self.thinking_enabled = getattr(config.inference, "thinking_enabled", True)
        # Processor and LLM
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)

        # InternVL 4B expects 448x448 vision inputs in vLLM.
        if self.is_internvl:
            mm_processor_kwargs = {
                "size": {"height": 448, "width": 448},
                "crop_size": {"height": 448, "width": 448},
                "nframes": int(self.max_images),
                "fps": 1,
            }
        else:
            mm_processor_kwargs = {
                "max_pixels": int(self.max_image_width) * int(self.max_image_height),
                "nframes": int(self.max_images),
                "fps": 1,
            }

        engine_args = AsyncEngineArgs(
            model=self.model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.7,
            enforce_eager=True,
            mm_processor_kwargs=mm_processor_kwargs,
            trust_remote_code=True,
        )
        
        self.llm = AsyncLLMEngine.from_engine_args(engine_args)
        
        self.sampling_params = SamplingParams(
            temperature=max(0.0, self.temperature),
            top_p=0.001,
            repetition_penalty=1.05,
            max_tokens=self.config.inference.max_output_tokens,
            stop_token_ids=[],
        )

    def build_prompt_and_mm(self, messages: List[Dict[str, Any]]):
        if self.is_internvl:
            image_inputs = []
            template_messages: List[Dict[str, Any]] = []
            for msg in messages:
                role = msg.get("role")
                content = msg.get("content")
                if not isinstance(content, list):
                    template_messages.append(msg)
                    continue

                rebuilt_content = []
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    item_type = item.get("type")
                    if item_type == "image_url":
                        image_url = item.get("image_url")
                        if isinstance(image_url, dict):
                            image_url = image_url.get("url")
                        if isinstance(image_url, str):
                            match = re.match(r"^data:image/[^;]+;base64,(.+)$", image_url)
                            if match is not None:
                                img_bytes = base64.b64decode(match.group(1))
                                image_inputs.append(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
                                # InternVL chat templates require explicit multimodal placeholders.
                                rebuilt_content.append({"type": "image"})
                    elif item_type == "text":
                        rebuilt_content.append({"type": "text", "text": item.get("text", "")})

                template_messages.append({"role": role, "content": rebuilt_content})

            try:
                prompt = self.processor.apply_chat_template(
                    template_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    thinking_enabled=self.thinking_enabled,
                )
            except TypeError:
                prompt = self.processor.apply_chat_template(
                    template_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

            mm_data: Dict[str, Any] = {}
            if image_inputs:
                mm_data["image"] = image_inputs
            return prompt, mm_data, None

        from qwen_vl_utils import process_vision_info

        prompt = self.processor.apply_chat_template(messages,
                                                    tokenize=False,
                                                    add_generation_prompt=True,
                                                    thinking_enabled=self.thinking_enabled)
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        mm_data: Dict[str, Any] = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs
        return prompt, mm_data, video_kwargs

    async def generate_text(self, messages: List[Dict[str, Any]]) -> str:
        prompt, mm_data, video_kwargs = self.build_prompt_and_mm(messages)
        
        request_id = str(uuid.uuid4())
        
        # Prepare arguments for AsyncLLMEngine.generate
        # We pass prompt, sampling_params, request_id, and optionally multi_modal_data
        
        kwargs = {}
        # Only pass multi_modal_data if it's not empty, to avoid issues if the engine doesn't support it when None
        if mm_data:
            # Note: Some vLLM versions might expect 'inputs' dict instead of 'prompt' + 'multi_modal_data'
            # But since 'inputs' kwarg failed, we try explicit args.
            # If this fails, we might need to check vLLM version or use inputs positional arg.
            kwargs["multi_modal_data"] = mm_data
        
        if video_kwargs:
            kwargs["mm_processor_kwargs"] = video_kwargs

        try:
            results_generator = self.llm.generate(
                prompt=prompt,
                sampling_params=self.sampling_params,
                request_id=request_id,
                **kwargs
            )
        except TypeError as e:
            # Fallback: try passing prompt as positional argument if keyword fails
            # or if inputs is expected as positional
            if "multi_modal_data" in str(e) or "mm_processor_kwargs" in str(e):
                 # If kwargs are not supported, try passing inputs dict as positional prompt
                 # This works in some vLLM versions where the first arg 'inputs' can be a dict
                 inputs_dict = {"prompt": prompt}
                 if mm_data:
                     inputs_dict["multi_modal_data"] = mm_data
                 if video_kwargs:
                     inputs_dict["mm_processor_kwargs"] = video_kwargs
                 
                 results_generator = self.llm.generate(
                    inputs_dict,
                    self.sampling_params,
                    request_id,
                )
            elif "inputs" in str(e) or "prompt" in str(e):
                 # Construct inputs dict if that's what it wants (but passed positionally?)
                 # Or just pass prompt positionally
                 results_generator = self.llm.generate(
                    prompt,
                    self.sampling_params,
                    request_id,
                    **kwargs
                )
            else:
                raise e
        
        final_output = None
        async for request_output in results_generator:
            final_output = request_output
            
        return final_output.outputs[0].text
    
    def _encode_frame(self, frame, max_width=512, max_height=512, quality=85):
        # Resize frame to reduce aspect ratio and make it easier to parse
        height, width = frame.shape[:2]
        # Calculate scaling factor to fit within max dimensions while maintaining aspect ratio
        scale = min(max_width / width, max_height / height)
        
        # Only resize if the image is larger than max dimensions
        if scale < 1.0:
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Encode a uint8 numpy array (image) as a JPEG and then base64 encode it.
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        ret, buffer = cv2.imencode(".jpg", frame, encode_params)
        if not ret:
            raise ValueError("Could not encode frame")
        return base64.b64encode(buffer).decode("utf-8")



