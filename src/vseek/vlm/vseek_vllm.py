import re

import torch
from langchain_core.output_parsers import PydanticOutputParser

from vseek.data.exp_io import DataInput
from vseek.data.vseek_dm import AgentOutput, ReasoningTrajectory
from vseek.setting import ViClipSetting, VLLMSetting
from vseek.video_embedding.video_clip import ViClip
from vseek.vlm.vllm_client import VLLMClient

VLLM_SETTING = VLLMSetting()
VICLIP_SETTING = ViClipSetting()


class VSeekAgent(VLLMClient):
    def __init__(
        self,
        api_key=None,
        api_base=None,
        model=None,
    ):
        super().__init__(api_key=api_key, api_base=api_base, model=model)
        self.viclip = ViClip(
            pretrained_model_path=VICLIP_SETTING.vclip_model_path,
            gpu_number=VICLIP_SETTING.gpu_number,
        )

    def _parse_response_with_regex(
        self, content: str, just_thought: bool = False
    ) -> AgentOutput | None:
        """
        Fallback regex-based parser for when JSON parsing fails.
        Extracts thought, answer, and search fields from the response text.
        """
        try:
            # Clean the content - remove extra whitespace and newlines
            content = content.strip()

            # Remove markdown code block formatting if present
            # Handle cases like ```json\n{...}\n``` or ```\n{...}\n```
            if content.startswith("```"):
                # Find the first newline after ```
                first_newline = content.find("\n")
                if first_newline != -1:
                    # Remove everything up to and including the first newline
                    content = content[first_newline + 1 :]

                # Remove trailing ``` if present
                if content.endswith("```"):
                    content = content[:-3]
                elif content.endswith("\n```"):
                    content = content[:-4]

                content = content.strip()

            # Try to find the JSON-like structure within the content
            # Look for patterns like "field": "value" or "field": null

            # Extract thought field
            thought_pattern = r'"thought"\s*:\s*"([^"]*(?:\\.[^"]*)*)"'
            thought_match = re.search(thought_pattern, content, re.DOTALL)
            thought = thought_match.group(1) if thought_match else None

            if thought:
                # Unescape JSON escaped characters
                thought = (
                    thought.replace('\\"', '"')
                    .replace("\\n", "\n")
                    .replace("\\\\", "\\")
                )

            # Extract answer field (can be null or a string)
            answer_pattern = r'"answer"\s*:\s*(?:"([^"]*(?:\\.[^"]*)*)"|(null))'
            answer_match = re.search(answer_pattern, content, re.DOTALL)
            answer = None
            if answer_match:
                if answer_match.group(2) == "null":
                    answer = None
                elif answer_match.group(1) is not None:
                    answer = (
                        answer_match.group(1)
                        .replace('\\"', '"')
                        .replace("\\n", "\n")
                        .replace("\\\\", "\\")
                    )

            # Extract search field (can be null or a string)
            search_pattern = r'"search"\s*:\s*(?:"([^"]*(?:\\.[^"]*)*)"|(null))'
            search_match = re.search(search_pattern, content, re.DOTALL)
            search = None
            if search_match:
                if search_match.group(2) == "null":
                    search = None
                elif search_match.group(1) is not None:
                    search = (
                        search_match.group(1)
                        .replace('\\"', '"')
                        .replace("\\n", "\n")
                        .replace("\\\\", "\\")
                    )

            # Validate that we have the required fields
            if not thought or len(thought.strip()) < 5:
                print("Regex parser: Invalid or missing thought field")
                return None

            # Check answer length constraint (AgentOutput has max_length=200)
            if answer and len(answer.strip()) > 200:
                print(
                    f"Regex parser: Answer field too long ({len(answer.strip())} chars), truncating to 200 chars"
                )
                answer = answer.strip()[:197] + "..."  # Truncate and add ellipsis

            # Check search length constraint (AgentOutput has max_length=200)
            if search and len(search.strip()) > 200:
                print(
                    f"Regex parser: Search field too long ({len(search.strip())} chars), truncating to 200 chars"
                )
                search = search.strip()[:197] + "..."  # Truncate and add ellipsis

            # Ensure exactly one of answer or search is provided (XOR validation)
            has_answer = answer is not None and len(answer.strip()) > 0
            has_search = search is not None and len(search.strip()) > 0

            if has_answer == has_search:  # Both true or both false
                print(
                    f"Regex parser: XOR validation failed - has_answer: {has_answer}, has_search: {has_search}"
                )
                if not just_thought:
                    return None
                else:
                    return AgentOutput(
                        thought=thought.strip(),
                        answer="",
                        search="",
                    )

            # Create AgentOutput object
            return AgentOutput(
                thought=thought.strip(),
                answer=answer.strip() if answer else None,
                search=search.strip() if search else None,
            )

        except Exception as e:
            print(f"Regex parsing failed: {e}")
            return None

    def run(
        self,
        data_input: DataInput,
        max_parse_attempts: int = 3,
        max_reasoning_attempts: int = 10,
    ) -> ReasoningTrajectory:
        video_frames = data_input.video
        parser = PydanticOutputParser(pydantic_object=AgentOutput)

        parser_template = parser.get_format_instructions()

        system_prompt = f"""
        You are a video analysis assistant. Analyze the video frames and respond in the specified JSON format.

        INSTRUCTIONS:
        1. Look at the video frames carefully
        2. For questions you can answer directly from the video, provide an "answer"
        3. For questions requiring more information, provide a "search" query
        4. The output should be formatted as a JSON instance that conforms to the JSON schema below.

        {parser_template}

        EXAMPLES:
        - Always fill exactly ONE field (answer OR search), never both"""
        iteration = 0
        parse_attempts = 0
        reasoning_trajectory = []
        while True:
            if iteration == 0:
                video_window_idx = 0
                user_content = [
                    {
                        "type": "text",
                        "text": f"QUESTION: {data_input.question}\n\nAnalyze these video frames in sequence:",
                    }
                ]
            encoded_images = [
                self._encode_frame(frame)
                for frame in data_input.video.get_frame_chunk(video_window_idx)
            ]
            # Build the user message: a text prompt plus one image for each frame.

            for encoded in encoded_images:
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                    }
                )

            chat_response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=400,
                temperature=0.0,
                logprobs=True,
                top_logprobs=20,
            )
            content = chat_response.choices[0].message.content

            # Parse the response using the pydantic parser
            try:
                # try with pydantic parser
                agent_output = parser.parse(content)

            except ValueError:
                # Only thought is provided
                agent_output = self._parse_response_with_regex(
                    content=content, just_thought=True
                )

            except Exception as e:
                print(f"Error parsing response with Pydantic (General Exception): {e}")
                print("Attempting regex-based fallback parsing...")

                # Try regex-based fallback parsing
                agent_output = self._parse_response_with_regex(content)

                if agent_output is None:
                    if parse_attempts >= max_parse_attempts:
                        return ReasoningTrajectory(
                            reasoning_trajectory=reasoning_trajectory,
                            is_error=True,
                            error_message=f"Error parsing response with both Pydantic and regex: {e}",
                        )
                    parse_attempts += 1  # engineering iteration
                    continue  # Try again with next iteration

            if agent_output:
                reasoning_trajectory.append(agent_output)

                if agent_output.answer:
                    return ReasoningTrajectory(
                        reasoning_trajectory=reasoning_trajectory,
                        is_found_answer=True,
                    )
                else:
                    # search
                    embeddings = video_frames.embeddings
                    search_indices = self.search(embeddings, agent_output.search)

                    # Use the most relevant video segment for next iteration
                    if search_indices:
                        video_window_idx = search_indices[0]

                    iteration += 1
                    if iteration >= max_reasoning_attempts:
                        return ReasoningTrajectory(
                            reasoning_trajectory=reasoning_trajectory,
                            is_found_answer=False,
                        )

    def search(self, embeddings: list[torch.Tensor], search_query: str) -> list[int]:
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
