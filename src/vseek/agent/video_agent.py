from pathlib import Path

import torch

from vseek.agent.utils.parse_response import parse_response_with_regex
from vseek.data.exp_io import DataInput
from vseek.data.vseek_dm import ReasoningTrajectory
from vseek.setting import ViClipSetting, VLLMSetting
from vseek.utils.prompt import load_prompt_template
from vseek.video_embedding.video_clip import ViClip
from vseek.vlm.vllm_client import VLLMClient

VLLM_SETTING = VLLMSetting()
VICLIP_SETTING = ViClipSetting()
AGENT_MODULE_ROOT_PATH = Path(__file__).parent
AGENT_PROMPT_ROOT_PATH = AGENT_MODULE_ROOT_PATH.joinpath("prompts")


class VSeekAgent(VLLMClient):
    def __init__(
        self,
        api_key="EMPTY",
        api_base=None,
        model=None,
        max_image_width=256,
        max_image_height=256,
        image_quality=85,
    ):
        super().__init__(api_key=api_key, api_base=api_base, model=model)
        self.viclip = ViClip(
            pretrained_model_path=VICLIP_SETTING.vclip_model_path,
            gpu_number=VICLIP_SETTING.gpu_number,
        )
        self.max_image_width = max_image_width
        self.max_image_height = max_image_height
        self.image_quality = image_quality

    def _encode_frame(self, frame):
        """Override parent method to use custom image dimensions and quality."""
        return super()._encode_frame(
            frame,
            max_width=self.max_image_width,
            max_height=self.max_image_height,
            quality=self.image_quality,
        )

    def run(
        self,
        data_input: DataInput,
        max_parse_attempts: int = 3,
        max_reasoning_attempts: int = 5,
    ) -> ReasoningTrajectory:
        video_frames = data_input.video

        system_prompt = load_prompt_template(
            AGENT_PROMPT_ROOT_PATH.joinpath("vseek_v0.0.2.txt")
        )

        iteration = 0
        parse_attempts = 0
        reasoning_trajectory = []
        encoded_images = []
        assistant_content = None
        message_content = [{"role": "system", "content": system_prompt}]
        while True:
            if iteration == 0:
                video_window_idx = 0
                user_content = [
                    {
                        "type": "text",
                        "text": f"Question: {data_input.question} Options: {data_input.options} \n",
                    }
                ]
            else:
                encoded_images = []
                for idx in video_window_idx:
                    encoded_images += [
                        self._encode_frame(frame)
                        for frame in data_input.video.get_frame_chunk(idx)
                    ]
            # # Build the user message: a text prompt plus one image for each frame.

            for encoded in encoded_images:
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                    }
                )
            message_content.append({"role": "user", "content": user_content})

            chat_response = self.client.chat.completions.create(
                model=self.model,
                messages=message_content,
                max_tokens=500,
                temperature=0.5,
                logprobs=True,
                top_logprobs=20,
            )
            content = chat_response.choices[0].message.content

            # Parse the response using tag-based format
            agent_output = parse_response_with_regex(content)
            message_content.append({"role": "assistant", "content": content})
            if agent_output is None:
                if parse_attempts >= max_parse_attempts:
                    return ReasoningTrajectory(
                        reasoning_trajectory=reasoning_trajectory,
                        is_error=True,
                        error_message="Error parsing response with tag-based parser after max attempts",
                    )
                parse_attempts += 1  # engineering iteration
                continue  # Try again with next iteration

            # TODO: agent needs to handle the search_subtitle case
            if agent_output:
                reasoning_trajectory.append(agent_output)

                if agent_output.answer:
                    return ReasoningTrajectory(
                        reasoning_trajectory=reasoning_trajectory,
                        is_found_answer=True,
                        answer=agent_output.answer,
                    )
                elif agent_output.search or agent_output.subtitle:
                    # search

                    if agent_output.search:
                        embeddings = video_frames.embeddings
                        search_indices = self.search_video(
                            embeddings, agent_output.search
                        )
                    elif agent_output.subtitle:
                        search_indices = self.search_subtitle(
                            video_frames.subtitles, agent_output.subtitle
                        )
                    print("Search indices: ", search_indices)
                    # Use the most relevant video segment for next iteration
                    if search_indices:
                        video_window_idx = search_indices[:1]

                    iteration += 1
                    if iteration >= max_reasoning_attempts:
                        return ReasoningTrajectory(
                            reasoning_trajectory=reasoning_trajectory,
                            is_found_answer=False,
                        )

    def search_video(
        self, embeddings: list[torch.Tensor], search_query: str
    ) -> list[int]:
        """Search for the most similar visual embeddings to the text query.

        Args:
            embeddings: List of visual embeddings (tensors)
            search_query: Text query to search for

        Returns:
            List of indices sorted by similarity (highest first)
        """
        # Get text embedding for the search query
        text_embedding = self.viclip.get_text_embedding(search_query)

        # Get the device of the text embedding (likely GPU)
        device = text_embedding.device

        # Ensure text embedding is 1D [embedding_dim]
        if text_embedding.dim() > 1:
            text_embedding = text_embedding.squeeze()

        # Stack all visual embeddings into a single tensor
        # Move to same device as text embedding and normalize
        visual_embeddings = []
        for emb in embeddings:
            emb_device = emb.to(device)  # Move to same device as text embedding
            # Ensure embedding is 1D
            if emb_device.dim() > 1:
                emb_device = emb_device.squeeze()
            emb_norm = emb_device / emb_device.norm(dim=-1, keepdim=True)
            visual_embeddings.append(emb_norm)

        visual_embeddings_tensor = torch.stack(
            visual_embeddings
        )  # Shape: [N, embedding_dim]

        # Compute cosine similarities on GPU
        # text_embedding is already normalized from get_text_embedding
        # visual_embeddings_tensor: [N, D], text_embedding: [D] -> similarities: [N]
        similarities = torch.matmul(visual_embeddings_tensor, text_embedding)

        # Get indices sorted by similarity (descending order)
        sorted_indices = torch.argsort(similarities, descending=True)

        return sorted_indices.cpu().tolist()

    def search_subtitle(self, subtitles: list[str], search_query: str) -> list[int]:
        """Search for the most similar subtitle to the text query.

        Args:
            subtitles: List of subtitles (list of strings)
            search_query: Text query to search for

        Returns:
            List of indices sorted by similarity (highest first)
        """
        # Get text embedding for the search query
        text_embedding = self.viclip.get_text_embedding(search_query)

        # Get the device of the text embedding (likely GPU)
        device = text_embedding.device

        # Ensure text embedding is 1D [embedding_dim]
        if text_embedding.dim() > 1:
            text_embedding = text_embedding.squeeze()

        # Normalize the text embedding
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

        # Get text embeddings for all subtitles
        subtitle_embeddings = []
        print("len(subtitles): ", len(subtitles))
        for subtitle in subtitles:
            emb = self.viclip.get_text_embedding(subtitle)
            emb_device = emb.to(device)  # Move to same device as query embedding
            # Ensure embedding is 1D
            if emb_device.dim() > 1:
                emb_device = emb_device.squeeze()
            emb_norm = emb_device / emb_device.norm(dim=-1, keepdim=True)
            subtitle_embeddings.append(emb_norm)

        subtitle_embeddings_tensor = torch.stack(
            subtitle_embeddings
        )  # Shape: [N, embedding_dim]

        # Compute cosine similarities on GPU
        # subtitle_embeddings_tensor: [N, D], text_embedding: [D] -> similarities: [N]
        similarities = torch.matmul(subtitle_embeddings_tensor, text_embedding)

        # Get indices sorted by similarity (descending order)
        sorted_indices = torch.argsort(similarities, descending=True)

        return sorted_indices.cpu().tolist()

        return sorted_indices.cpu().tolist()
        return sorted_indices.cpu().tolist()
