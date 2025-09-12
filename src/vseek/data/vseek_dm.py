from typing import List

from pydantic import BaseModel, Field


class Search(BaseModel):
    search_query: str = Field(
        description="The search query to be used to search for videos"
    )
    video_index: List[int] = Field(
        description="The index of the video to be used to search for videos"
    )


class VLMOutput(BaseModel):
    thought: str = Field(description="The thought of the VLM")
    search: Search = Field(description="The search to be used to search for videos")
    answer: str = Field(description="The answer to the search query")


class ReasoningTrajectory(BaseModel):
    reasoning_trajectory: List[VLMOutput] = Field(
        description="The reasoning trajectory of the VLM"
    )
