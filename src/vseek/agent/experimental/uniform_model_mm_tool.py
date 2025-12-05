# An experiment with an agentic setup that populates the multi-modal data from the video as tool calls. This is to experiment with whether the tool calls itself are reponsible for poorer performance

import numpy as np
from PIL import Image

from vseek.data.exp_io import DataInput
from vseek.data.vseek_dm import ReasoningTrajectory
from vseek.agent.base_model import LocalVLLMBase
from vseek.agent.uniform_agent import UniformSampleAgent
from vseek.agent.utils.parse_response import parse_response_with_regex


class ToolUniformSampleAgent(UniformSampleAgent):


    async def run(self, data_input: DataInput) -> ReasoningTrajectory:
        # Prefer native video pathway if available; otherwise fall back to frames-as-images.
        # video_path = getattr(data_input.video, "video_path", None)
        video_path = None
        if self.agent_prompt_type == "base" or self.agent_prompt_type == "single":
            system_prompt = """
                You are a helpful assistant. Look at the provided images sampled uniformly from a video and must choose the correct option from the given options to answer the question.
                You must only output the number of the correct option without thinking. Eg. 3
                
            """
        elif self.agent_prompt_type == "cot":
            system_prompt = """
                You are a helpful assistant. Look at the provided images sampled uniformly from a video and answer the question concisely.
                **INSTRUCTIONS**:
                1. First think very concisely within 100 words, about the question and the provided images within the <think> and </think> tags.
                2. You must provide the final answer the question with the correct option after ###.
                3. Do not provide empty fields and you must provide an answer to the best of your ability.
                4. For example.
                <think>I have found the mixing bowl, but no ingredients have been added yet. I need to find the next action where something is put into the bowl.</think>
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
                {"role": "tool", "content": user_content},
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



