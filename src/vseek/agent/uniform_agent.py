import os
import numpy as np
from PIL import Image

from vseek.data.exp_io import DataInput
from vseek.data.vseek_dm import ReasoningTrajectory
from vseek.agent.base_model import LocalVLLMBase
from vseek.agent.utils.parse_response import parse_response_with_regex


class UniformSampleAgent(LocalVLLMBase):
    def __init__(
        self,
        config,
    ):
        super().__init__(config=config)

        self.max_image_width = config.inference.max_image_width
        self.max_image_height = config.inference.max_image_height
        self.image_quality = config.inference.image_quality
        self.temperature = getattr(config.inference, "temperature", 0.0)
        self.max_images = getattr(config.inference, "max_images_per_turn", 16)


    def _encode_frame(self, frame):
        return super()._encode_frame(
            frame,
            max_width=self.max_image_width,
            max_height=self.max_image_height,
            quality=self.image_quality,
        )

    def run(self, data_input: DataInput) -> ReasoningTrajectory:
        # Prefer native video pathway if available; otherwise fall back to frames-as-images.
        video_path = getattr(data_input.video, "video_path", None)

        system_prompt = """
            You are a helpful assistant. Look at the provided images sampled uniformly from a video and answer the question concisely.
            You must answer the question with the correct option within the <answer></answer> tags. 
            Do not provide empty fields and you must provide an answer to the best of your ability.

            **EXAMPLES**:
            (the agent receives a video in the form of series of images)
            Question: What is the first ingredient the chef adds to the mixing bowl? Options: A. Mint , B. Sugar, C. Salt, D. Flour
            D
           
           (the agent receives a video in the form of series of images)
            Question: What is the color of the unicorn? Options: A. Green, B. Blue, C. Pink, D. White
            C
        """
        
        system_prompt_cot = """
            You are a helpful assistant. Look at the provided images sampled uniformly from a video and answer the question concisely.
            You must answer the question with the correct option within the <answer></answer> tags. 
            Do not provide empty fields and you must provide an answer to the best of your ability.

            **EXAMPLES**:
            (the agent receives a video in the form of series of images)
            Question: What is the first ingredient the chef adds to the mixing bowl? Options: A. Mint , B. Sugar, C. Salt, D. Flour
            D
           
           (the agent receives a video in the form of series of images)
            Question: What is the color of the unicorn? Options: A. Green, B. Blue, C. Pink, D. White
            C
        """


        if video_path:
            print("Using video pathway")
            user_content = f"Question: {data_input.question} Options: {data_input.options}"
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": user_content},
                    {"type": "video", "video": VIDEO_PATH, "total_pixels": args.max_pixels, "nframes": args.nframes, "fps": args.fps},
                ]},
            ]

            content = self.generate_text(messages)
            # content = data["choices"][0]["message"]["content"]
            # user_content = [
            #     {"type": "text", "text": f"Question: {data_input.question} Options: {data_input.options}"},
            # ]
            # chat_response = self.client.chat.completions.create(
            #     model=self.model,
            #     messages=[
            #         {"role": "system", "content": system_prompt},
            #         {"role": "user", "content": user_content},
            #     ],
            #     multi_modal_data={"video": [{"video_path": video_path, "fps": 1, "max_frames": int(self.max_images)}]},
            #     max_tokens=1,
            #     temperature=self.temperature,
            # )
        else:
            print("Using frames pathway")
            frames = data_input.video.all_frames
            if not frames:
                return ReasoningTrajectory(
                    reasoning_trajectory=[], is_error=True, error_message="No frames available for inference"
                )

            k = int(self.max_images) if self.max_images is not None else len(frames)
            k = max(1, min(k, len(frames)))
            indices = np.linspace(0, len(frames) - 1, k, dtype=int)
            selected = [frames[i] for i in indices]

            encoded_images = [self._encode_frame(f) for f in selected]
            user_content = []

            for enc in encoded_images:
                user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{enc}"}})
            
            user_content.append({"type": "text", "text": f"Question: {data_input.question} Options: {data_input.options}"})
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
            content = self.generate_text(messages)
            # content = chat_response.choices[0].message.content

        # agent_output = parse_response_with_regex(content)
        
        # if agent_output is None or agent_output.answer is None:
        #     return ReasoningTrajectory(
        #         reasoning_trajectory=[], is_found_answer=False, answer=""
        #     )
        
        # return ReasoningTrajectory(
        #     reasoning_trajectory=[agent], is_found_answer=True, answer=agent_output.answer
        # )
        return ReasoningTrajectory(
            reasoning_trajectory=[], is_found_answer=True, answer=content
        )



