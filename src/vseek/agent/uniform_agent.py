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
        self.config = config
        self.max_image_width = config.inference.max_image_width
        self.max_image_height = config.inference.max_image_height
        self.image_quality = config.inference.image_quality
        self.temperature = getattr(config.inference, "temperature", 0.0)
        self.max_images = getattr(config.inference, "max_images_per_turn", 16)
        self.agent_prompt_type = getattr(config.inference, "agent_prompt_type", "base")

    def _encode_frame(self, frame):
        return super()._encode_frame(
            frame,
            max_width=self.max_image_width,
            max_height=self.max_image_height,
            quality=self.image_quality,
        )

    async def run(self, data_input: DataInput) -> ReasoningTrajectory:
        # Prefer native video pathway if available; otherwise fall back to frames-as-images.
        # video_path = getattr(data_input.video, "video_path", None)
        video_path = None
        if self.agent_prompt_type == "base" or self.agent_prompt_type == "single":
            system_prompt = """
                You are a helpful assistant. Look at the provided images sampled uniformly from a video and must choose the correct option from the given options to answer the question.
                You must only output the number of the correct option or the letter corresponding to the provided options without thinking. Eg. 3 or B
                If the options are numbered, you must output the number. If the options are letters, you must output the letter.
                
            """
        elif self.agent_prompt_type == "cot":
            system_prompt = """
                You are a helpful assistant. Look at the provided images sampled uniformly from a video and answer the question concisely.
                **INSTRUCTIONS**:
                1. First think very concisely within 100 words, about the question and the provided images within the <think> and </think> tags.
                2. You must provide the final answer the question with the correct option after ###.
                3. Do not provide empty fields and you must provide an answer to the best of your ability.
                4. For example.
                <think>I have found the mixing bowl, but no ingredients have been added yet. I need to find the next action where something is put into the bowl. I found it, based on the question, the action is mixing, hence the answer option is 3.</think>
                ### 3
                """

        if video_path:
            print("Using video pathway")
            user_content = f"Question: {data_input.question} Options: {data_input.options}"
            print(user_content)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": user_content},
                    {"type": "video", "video": data_input.video.video_path, "max_frames": self.config.inference.max_images_per_turn, "fps": 1},
                ]},
            ]

            content = await self.generate_text(messages)
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
                user_content.append({"type": "image_url", "image_url": f"data:image/jpeg;base64,{enc}"})
            
            user_content.append({"type": "text", "text": f"Question: {data_input.question} Options: {data_input.options}"})
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
            content = await self.generate_text(messages)
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



