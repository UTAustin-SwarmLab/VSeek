import os
from typing import Any, Dict, List

from transformers import AutoProcessor
from vllm import LLM, SamplingParams
import cv2
import base64

class LocalVLLMBase:
    def __init__(self, config) -> None:
        # GPU selection
        self.config = config
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
        )
        self.sampling_params = SamplingParams(
            temperature=max(0.0, self.temperature),
            top_p=0.001,
            repetition_penalty=1.05,
            max_tokens=self.config.inference.max_output_tokens,
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



