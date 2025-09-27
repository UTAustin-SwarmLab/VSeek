import torch
import requests

from vseek.agent.utils.parse_response import parse_response_with_regex
from vseek.data.exp_io import DataInput
from vseek.data.vseek_dm import ReasoningTrajectory
from vseek.vlm.vllm_client import VLLMClient


class VSeekAgent(VLLMClient):
    def __init__(
        self,
        config
    ):
        super().__init__(api_key=config.llm.openai_api_key, api_base=config.llm.server_url, model=config.llm.model)

        self.max_image_width = config.inference.max_image_width
        self.max_image_height = config.inference.max_image_height
        self.image_quality = config.inference.image_quality
        self.topk = config.inference.topk
        self.temperature = config.inference.temperature
        self.server_url = config.llm.server_url
        self.model = config.llm.model
        self.openai_api_key = config.llm.openai_api_key

        self.retriever_server_url = config.retriever.server_url
        # Tradeoff between topk and number of effective frames from each window

    def _encode_frame(self, frame):
        """Override parent method to use custom image dimensions and quality."""
        return super()._encode_frame(
            frame, 
            max_width=self.max_image_width, 
            max_height=self.max_image_height, 
            quality=self.image_quality
        )

    def run(
        self,
        data_input: DataInput,
        max_parse_attempts: int = 3,
        max_reasoning_attempts: int = 5,
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
        # system_prompt = """

        # You are a video analysis assistant. You will be given a question and access to a video. Your task is to answer the question in a step-by-step manner by analyzing the video.
        # **INSTRUCTIONS**:
        # 1. Read the user's question carefully.
        # 2. At each step, based on the video frames obtained so far, you must think carefully about the question and the current video frames, to decide whether you can answer the question directly or you need to search for the answer within the <think> and </think> tags.
        # 3. You must think inside <think> </think? tags, and reason about whether the current frames are sufficient to answer the question. If there are no frames that are relevant to the question, you need to search for the answer <search> and </search> tags.
        # 4. If you can definitively answer the question with the current information, provide the final answer inside <answer> and </answer> tags.
        # 5. If you need more information to answer the question, issue a precise video search query inside <search> and </search> tags. Your query should help you find the next relevant segment of the video. Ensure that the search query is not too general, long, vague or repeated across turns.
        # 6. Alternatively, if there are subtitles mentioned in the question, you can use the subtitles to retrieve the frames corresponding to the subtitle by using the <search_subtitle> and </search_subtitle> tags.
        # Each search query should be a concise (1–3 sentences), optimized video search query ONLY if you cannot answer. Include specific entities/objects/ and actions when available.
        # 7. Crucially, you must output exactly ONE tag from (<answer> or <search> or <search_subtitle>) per turn and a <think> tag. Do not provide both.
        # 8. You will have access to the history of each turn so far. You can take a maximum of 4 turns to answer the question.

        # **EXAMPLES**:
        # EXAMPLE 1:
        # Question: What is the first ingredient the chef adds to the mixing bowl? Options: A. , B. Sugar, C. Salt, D. Flour

        # Turn 1:
        # <think>To answer the question, I first need to locate the part of the video where the chef is using a mixing bowl.</think>
        # <search>a chef with a large mixing bowl</search>
        # (After the first search, the agent receives frames of the chef placing an empty bowl on the counter.)
        # Turn 2:
        # <think>I have found the mixing bowl, but no ingredients have been added yet. I need to find the next action where something is put into the bowl.</think>
        # <search>chef adding an ingredient to the bowl</search>
        # (After the second search, the agent receives frames of the chef pouring flour into the bowl.)
        # Turn 3:
        # <think>These frames clearly show the chef adding a white powder, which appears to be flour, into the bowl. This is the first ingredient. I can now answer the question.</think>
        # <answer>D</answer>

        # EXAMPLE 2: Temporal Reasoning 
        # Question: What does the person do right after picking up the red ball? Options: A.Put it in a box , B. Throw it to a dog, C. Put it on a shelf, D. Put it in a bag

        # Turn 1:
        # <think>I need to first find the moment the person picks up the red ball.</think>
        # <search>a person picking up a red ball</search>
        # (After the search, the agent receives frames of a person bending over and grabbing a red ball.)
        # Turn 2:
        # <think>I have found the event where the person picks up the red ball. Now I need to observe the immediate next action to answer the question. The immediate next action is throwing the ball to a dog hence I can answer the question with the option B.</think>
        # <answer>B</answer>
        
        # EXAMPLE 3:
        # Question: What is the on the right of the dark haired woman wearing a hat when the subtitle 'you're interested in. is mentioned? Options: A. Door B. Computer, C. Lamp, D. Phone
        # Turn 1:
        # <think>I need to first find the moment the subtitle 'you're interested in.' is mentioned.</think>
        # <search_subtitle>you're interested in.</search_subtitle>
        # (After the search, the agent receives frames of a dark haired woman wearing a hat.)
        # Turn 2:
        # <think>I have found the event where the subtitle 'you're interested in.' is mentioned. Now I need to observe the immediate next action to answer the question. The immediate next action is the dark haired woman wearing a hat hence I can answer the question with the option A.</think>
        # <answer>A</answer>

        # """

        system_prompt = """

        You are a video analysis assistant. You will be given a question and access to a video. Your task is to answer the question in a step-by-step manner by analyzing the video.
        **INSTRUCTIONS**:
        1. Read the user's question carefully.
        2. At each step, based on the video frames obtained so far, you must think carefully about the question and the current video frames, to decide whether you can answer the question directly or you need to search for the answer within the <think> and </think> tags.
        3. You must think inside <think> </think? tags, and reason about whether the current frames are sufficient to answer the question. If there are no frames that are relevant to the question, you need to search for the answer <search> and </search> tags.
        4. If you can definitively answer the question with the current information, provide the final answer inside <answer> and </answer> tags.
        5. If you need more information to answer the question, issue a precise video search query inside <search> and </search> tags. Your query should help you find the next relevant segment of the video. Ensure that the search query is not too general, long, vague or repeated across turns.
        Each search query should be a concise (1–3 sentences), optimized video search query ONLY if you cannot answer. Include specific entities/objects/ and actions when available.
        6. Crucially, you must output exactly ONE tag from (<answer> or <search>) per turn and a <think> tag. Do not provide both.
        7. You will have access to the history of each turn so far. You can take a maximum of 4 turns to answer the question.

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
        <answer>D</answer>

        EXAMPLE 2: Temporal Reasoning 
        Question: What does the person do right after picking up the red ball? Options: A.Put it in a box , B. Throw it to a dog, C. Put it on a shelf, D. Put it in a bag

        Turn 1:
        <think>I need to first find the moment the person picks up the red ball.</think>
        <search>a person picking up a red ball</search>
        (After the search, the agent receives frames of a person bending over and grabbing a red ball.)
        Turn 2:
        <think>I have found the event where the person picks up the red ball. Now I need to observe the immediate next action to answer the question. The immediate next action is throwing the ball to a dog hence I can answer the question with the option B.</think>
        <answer>B</answer>
        
        EXAMPLE 3: Repetition
        Question: What does the person do right after picking up the red ball? Options: A.Put it in a box , B. Throw it to a dog, C. Put it on a shelf, D. Put it in a bag

        Turn 1:
        <think>I need to first find the moment the person picks up the red ball.</think>
        <search>a person picking up a red ball</search>
        (After the search, the agent does not receive any relevant frames.)
        Turn 2:
        <think>I don't have enough context as the provided video does not include any relevant frames. I need to write a better search query. </think>
        <search>a person wearing tshirtpicking up a red ball in the room</search>
        <think>I have found the event where the person picks up the red ball. The immediate next action is throwing the ball to a dog hence I can answer the question with the option B.</think>
        <answer>B</answer>
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
                temperature=self.temperature,
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
                    # search via retriever server
                    if agent_output.search:
                        search_indices = self.search_video(agent_output.search, getattr(data_input, "video_id", None), self.topk)
                    elif agent_output.subtitle:
                        search_indices = self.search_subtitle(agent_output.subtitle, getattr(data_input, "video_id", None), self.topk)
                    print("Search indices: ", search_indices)
                    # Use the most relevant video segment for next iteration
                    if search_indices:
                        video_window_idx = search_indices[:self.topk]

                    iteration += 1
                    if iteration >= max_reasoning_attempts:
                        return ReasoningTrajectory(
                            reasoning_trajectory=reasoning_trajectory,
                            is_found_answer=False,
                        )

    def search_video(self, search_query: str, video_id: str | None = None, topk: int | None = None) -> list[int]:
        params = {"query": search_query}
        if topk is not None:
            params["topk"] = topk
        if video_id:
            params["video_id"] = video_id
        try:
            resp = requests.get(f"{self.retriever_server_url}/search", params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return [int(i) for i in data.get("frame_indices", [])]
        except Exception:
            return []

    def search_subtitle(self, search_query: str, video_id: str | None = None, topk: int | None = None) -> list[int]:
        params = {"query": search_query}
        if topk is not None:
            params["topk"] = topk
        if video_id:
            params["video_id"] = video_id
        try:
            resp = requests.get(f"{self.retriever_server_url}/search_subtitle", params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            indices =  data.get("subtitle_indices")
            return [int(i) for i in indices]
        except Exception:
            return []
