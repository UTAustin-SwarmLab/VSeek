import re

import torch

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
        Parse response using tag-based format from system prompt.
        Extracts thought, answer, and search fields from <think>, <answer>, and <search> tags.
        """
        try:
            # Clean the content - remove extra whitespace and newlines
            content = content.strip()

            # Extract thought field from <think> tags
            thought_pattern = r"<think>(.*?)</think>"
            thought_match = re.search(thought_pattern, content, re.DOTALL)
            thought = thought_match.group(1).strip() if thought_match else None

            # Extract answer field from <answer> tags
            answer_pattern = r"<answer>(.*?)</answer>"
            answer_match = re.search(answer_pattern, content, re.DOTALL)
            answer = answer_match.group(1).strip() if answer_match else None

            # Extract search field from <search> tags
            search_pattern = r"<search>(.*?)</search>"
            search_match = re.search(search_pattern, content, re.DOTALL)
            search = search_match.group(1).strip() if search_match else None

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

        # system_prompt = f"""
        # You are a video analysis assistant. You would be given a question. You are going to use a video to answer the question in a step-by-step manner.

        # **INSTRUCTIONS**:
        # 1. At each step, based on the video frames obtained so far, you need to think carefully about the question and the current video frames, to decide whether you can answer the question directly or you need to search for the answer within the <think> and </think> tags.
        # 2. For questions you can answer directly from the video, you must provide an answer within the <answer> and </answer> tags.
        # 3. If you think you need more information, you must provide a search query within the <search> and </search> tags. Each search query should be a concise (1–3 sentences), optimized video search query ONLY if you cannot answer. Include specific entities/objects/ and actions when available.


        # **EXAMPLES**:
        
        # Question: Is there a unicorn prancing until a pizza eats a strawberry? Please answer in yes or no.
        # <think>I first need to search for relevent video frames that has a unicorn prancing and a pizza eating a strawberry</think>
        # <search>a unicorn is prancing</search>
        # <think>These frames don't show a unicorn prancing and pizza eating a strawberry. I now need to search for frames that show a pizza eating a strawberry</think>
        
        # Question: What is the color of the unicorn?
        # <answer>The color of the unicorn is white.</answer>
        # <search>a unicorn is prancing until a pizza eats a strawberry</search>
        # - Always fill exactly ONE field (answer OR search), never both"""
        system_prompt = f"""

        You are a video analysis assistant. You will be given a question and access to a video. Your task is to answer the question in a step-by-step manner by analyzing the video.
        **INSTRUCTIONS**:
        1. Read the user's question carefully.
        2. At each step, based on the video frames obtained so far, you need to think carefully about the question and the current video frames, to decide whether you can answer the question directly or you need to search for the answer within the <think> and </think> tags.
        3. Inside <think> tags, reason about whether the current frames are sufficient to answer the question. If there are no frames that are relevant to the question, you need to search for the answer <search> and </search> tags.
        4. If you can definitively answer the question with the current information, provide the final answer inside <answer> and </answer> tags.
        5. If you need more information to answer the question, issue a precise video search query inside <search> and </search> tags. Your query should help you find the next relevant segment of the video.
        6. Alternatively, if there are subtitles mentioned in the question, you can use the subtitles to retrieve the frames corresponding to the subtitle by using the <search_subtitle> and </search_subtitle> tags.
        Each search query should be a concise (1–3 sentences), optimized video search query ONLY if you cannot answer. Include specific entities/objects/ and actions when available.
        7. Crucially, you must output exactly ONE tag (<answer> or <search> or <search_subtitle>) per turn. Do not provide both.
        8. You will have access to the history of each turn so far.

        **EXAMPLES**:
        EXAMPLE 1:
        Question: What is the first ingredient the chef adds to the mixing bowl? Options: A. , B. Sugar, C. Salt, D. Flour

        Turn 1:
        <think>To answer the question, I first need to locate the part of the video where the chef is using a mixing bowl.</think>
        <search>a chef with a large mixing bowl</search>
        (After the first search, the agent receives frames of the chef placing an empty bowl on the counter.)
        Turn 2:
        <think>I have found the mixing bowl, but no ingredients have been added yet. I need to find the next action where something is put into the bowl.</think>
        <search>chef adding an ingredient to the bowl</search>
        (After the second search, the agent receives frames of the chef pouring flour into the bowl.)
        Turn 3:
        <think>These frames clearly show the chef adding a white powder, which appears to be flour, into the bowl. This is the first ingredient. I can now answer the question.</think>
        <answer>D.</answer>

        EXAMPLE 2:
        Question: What color is the car parked in the driveway? Options: A. Blue, B. Red, C. Green, D. Yellow

        Turn 1:
        <think>The question asks for the color of the car in the driveway. I have sufficient information to answer hence I need to locate the part of the video where the car is parked in the driveway.</think>
        <search>a car parked in a driveway</search>
        (After the first search, the agent receives frames of the car parked in the driveway.)
        Turn 2:
        <think>I have found the car parked in the driveway. I need to locate the part of the video where the color of the car is visible. The care is blue hence I can answer the question with the option A.</think>
        <answer>A.</answer>

        EXAMPLE 3: Temporal Reasoning 
        Question: What does the person do right after picking up the red ball? Options: A.Put it in a box , B. Throw it to a dog, C. Put it on a shelf, D. Put it in a bag

        Turn 1:
        <think>I need to first find the moment the person picks up the red ball.</think>
        <search>a person picking up a red ball</search>
        (After the search, the agent receives frames of a person bending over and grabbing a red ball.)
        Turn 2:
        <think>I have found the event where the person picks up the red ball. Now I need to observe the immediate next action to answer the question. The immediate next action is throwing the ball to a dog hence I can answer the question with the option B.</think>
        <answer>B.</answer>
        
        EXAMPLE 4:
        Question: What is the on the right of the dark haired woman wearing a hat when the subtitle 'you're interested in. is mentioned? Options: A. Door B. Computer, C. Lamp, D. Phone
        Turn 1:
        <think>I need to first find the moment the subtitle 'you're interested in.' is mentioned.</think>
        <search_subtitle>you're interested in.</search_subtitle>
        (After the search, the agent receives frames of a dark haired woman wearing a hat.)
        Turn 2:
        <think>I have found the event where the subtitle 'you're interested in.' is mentioned. Now I need to observe the immediate next action to answer the question. The immediate next action is the dark haired woman wearing a hat hence I can answer the question with the option A.</think>
        <answer>A.</answer>

        """
        
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
                
                encoded_images = [
                        self._encode_frame(frame)
                        for frame in data_input.video.get_frame_chunk(video_window_idx)
                    ]
            # # Build the user message: a text prompt plus one image for each frame.

            for encoded in encoded_images:
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                    }
                )
            message_content.append(user_content)

            chat_response = self.client.chat.completions.create(
                model=self.model,
                messages=message_content,
                max_tokens=400,
                temperature=0.0,
                logprobs=True,
                top_logprobs=20,
            )
            content = chat_response.choices[0].message.content

            # Parse the response using tag-based format
            agent_output = self._parse_response_with_regex(content)
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
