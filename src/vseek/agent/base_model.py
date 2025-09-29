import os
from typing import Any, Dict, List

from transformers import AutoProcessor
from vllm import LLM, SamplingParams


class LocalVLLMBase:
    def __init__(self, config) -> None:
        # GPU selection
        self.gpu_number = getattr(config.inference, "gpu_number", 0)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_number)

        # Model / generation settings
        self.model_path = config.llm.model
        self.max_image_width = config.inference.max_image_width
        self.max_image_height = config.inference.max_image_height
        self.max_images = getattr(config.inference, "max_images_per_turn", 16)
        self.temperature = float(getattr(config.inference, "temperature", 0.0))

        # Processor and LLM
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.llm = LLM(
            model=self.model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.8,
            enforce_eager=True,
            limit_mm_per_prompt={"video": 1},
            mm_processor_kwargs={
                "max_pixels": int(self.max_image_width) * int(self.max_image_height),
                "nframes": int(self.max_images),
                "fps": 1,
            },
            trust_remote_code=True,
            device=self.gpu_number,
        )
        self.sampling_params = SamplingParams(
            temperature=max(0.0, self.temperature),
            top_p=0.001,
            repetition_penalty=1.05,
            max_tokens=512,
            stop_token_ids=[],
        )

    def build_prompt_and_mm(self, messages: List[Dict[str, Any]]):
        from qwen_vl_utils import process_vision_info

        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        mm_data: Dict[str, Any] = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs
        return prompt, mm_data, video_kwargs

    def generate_text(self, messages: List[Dict[str, Any]]) -> str:
        prompt, mm_data, video_kwargs = self.build_prompt_and_mm(messages)
        llm_inputs = {
            "prompt": prompt,
            "multi_modal_data": mm_data,
            "mm_processor_kwargs": video_kwargs,
        }
        outputs = self.llm.generate([llm_inputs], sampling_params=self.sampling_params)
        return outputs[0].outputs[0].text


