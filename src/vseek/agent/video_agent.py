import torch
import numpy as np
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
        self.max_images_per_turn = getattr(config.inference, "max_images_per_turn", 20)

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
        6. SUBTITLE SEARCH (important): Many questions refer to spoken words. If the question quotes text, refers to what someone "says"/"said"/"mentions"/"asks", or explicitly mentions "subtitle"/"subtitles", you MUST use <search_subtitle> and </search_subtitle> with the EXACT quoted/spoken line as the query — do NOT use <search> for these. <search_subtitle> returns the frames where that line is spoken. Rule of thumb: if the cue is a spoken line or dialogue, use <search_subtitle>; if the cue is a visible object or action, use <search>.
        7. Crucially, you must output exactly ONE non empty field from (<answer> or <search> or <search_subtitle>) per turn and a non empty <think> field. Do not provide more than one and do not provide empty fields.
        8. You will have access to the history of each turn so far. You can take a maximum of 4 turns to answer the question.
        9. The options are numbered from 0 to N-1. You MUST answer the question with only one of the option number. You must not provide any other text in the <answer> tag.

        **EXAMPLES**:
        EXAMPLE 1:
        Question: What is the first ingredient the chef adds to the mixing bowl? Options: 0. Mint , 1. Sugar, 2. Salt, 3. Flour

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
        <answer>3</answer>

        EXAMPLE 2: Temporal Reasoning 
        Question: What does the person do right after picking up the red ball? Options: 0.Put it in a box , 1. Throw it to a dog, 2. Put it on a shelf, 3. Put it in a bag

        Turn 1:
        <think>I need to first find the moment the person picks up the red ball.</think>
        <search>a person picking up a red ball</search>
        (After the search, the agent receives frames of a person bending over and grabbing a red ball.)
        Turn 2:
        <think>I have found the event where the person picks up the red ball. Now I need to observe the immediate next action to answer the question. The immediate next action is throwing the ball to a dog hence I can answer the question with the option 1.</think>
        <answer>1</answer>
        
        EXAMPLE 3: Handling Repetition Cases
        Question: What does the person do right after picking up the red ball? Options: 0.Put it in a box , 1. Throw it to a dog, 2. Put it on a shelf, 3. Put it in a bag

        Turn 1:
        <think>I need to first find the moment the person picks up the red ball.</think>
        <search>a person picking up a red ball</search>
        (After the search, the agent does not receive any relevant frames.)
        Turn 2:
        <think>I don't have enough context as the provided video does not include any relevant frames. I need to write a better search query. </think>
        <search>a person wearing tshirtpicking up a red ball in the room</search>
        <think>I have found the event where the person picks up the red ball. The immediate next action is throwing the ball to a dog hence I can answer the question with the option B.</think>
        <answer>1</answer>

        EXAMPLE 4: Subtitle-grounded question (use <search_subtitle>, NOT <search>)
        Question: What is to the right of the woman in the hat when the subtitle "you're interested in" is spoken? Options: 0. Door, 1. Computer, 2. Lamp, 3. Phone

        Turn 1:
        <think>The question quotes a spoken line ("you're interested in"), so the cue is dialogue, not a visible object. I must locate where that subtitle is spoken using subtitle search.</think>
        <search_subtitle>you're interested in</search_subtitle>
        (After the search, the agent receives frames from the moment that line is spoken: a woman in a hat with a door to her right.)
        Turn 2:
        <think>These frames show the moment the subtitle is spoken. To the right of the woman there is a door, so the answer is option 0.</think>
        <answer>0</answer>
        """
        
        iteration = 0
        parse_attempts = 0
        reasoning_trajectory = []
        encoded_images = []
        assistant_content = None
        message_content = [{"role": "system", "content": system_prompt}]
        total_images=0
        while True:
            if iteration == 0:
                video_window_idx = [0]
                user_content = [
                    {
                        "type": "text",
                        "text": f"Question: {data_input.question} Options: {data_input.options} \n",
                    }
                ]
            else:
                encoded_images = []
                user_content = []
                for idx in video_window_idx:
                    encoded_images += [
                        self._encode_frame(frame)
                        for frame in data_input.video.get_frame_chunk(idx)
                    ]
                
                # Subsample the encoded images to the max_images_per_turn uniformly
                if len(encoded_images) > self.max_images_per_turn:
                    n = len(encoded_images)
                    k = int(self.max_images_per_turn)
                    if k <= 0:
                        encoded_images = []
                    elif k == 1:
                        encoded_images = [encoded_images[n // 2]]
                    else:
                        indices = np.linspace(0, n - 1, k, dtype=int).tolist()
                        encoded_images = [encoded_images[i] for i in indices]
                
                prev_total_images = total_images
                total_images += len(encoded_images)
                
                if total_images > 150:
                    extra_images = total_images - 150
                    encoded_images = encoded_images[:-extra_images]
                    total_images = prev_total_images + len(encoded_images)
                
                print("total_images: ", total_images)

            # # Build the user message: a text prompt plus one image for each frame.

            for encoded in encoded_images:
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                    }
                )
            print(len(user_content))
            message_content.append({"role": "user", "content": user_content})

            # Find number of images in message_content
            num_message_images = 0
            for message in message_content:
                if message["role"] == "user":
                    for content in message["content"]:
                        if content["type"] == "image_url":
                            num_message_images += 1
            print(f"Number of images in message_content: {num_message_images}")
            # Print number of images in user_content
                
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
                        video_window_idx = sorted(search_indices[:self.topk])

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
