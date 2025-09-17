from typing import List, Optional

from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, constr, model_validator


class Search(BaseModel):
    search_query: str = Field(
        description="The search query to be used to search for videos"
    )
    video_index: List[int] = Field(
        description="The index of the video to be used to search for videos"
    )


class AgentThought(BaseModel):
    thought: constr(strip_whitespace=True, min_length=5, max_length=400) = Field(
        description=(
            "A concise (1–3 sentences) high-level reasoning of how you interpret the question "
            "and decide whether you can answer directly or need to search. No step-by-step chain-of-thought."
        )
    )

    @staticmethod
    def get_prompt() -> str:
        parser = PydanticOutputParser(pydantic_object=AgentThought)
        parser_template = parser.get_format_instructions()
        return f"You're the CoT video agent. If you can answer the question directly, provide an answer. If you cannot answer the question directly, provide a search query to find the answer.\n\n{parser_template}"


class AgentOutput(BaseModel):
    thought: constr(strip_whitespace=True, min_length=5, max_length=1000) = Field(
        description=(
            "A concise (1–3 sentences) high-level reasoning of how you interpret the question "
            "and decide whether you can answer directly or need to search. No step-by-step chain-of-thought."
        )
    )

    # Provide EXACTLY ONE of {answer, search}
    answer: Optional[constr(strip_whitespace=True, min_length=1, max_length=200)] = (
        Field(
            default=None,
            description=(
                "Direct, self-contained answer to the question. Provide this ONLY if you are confident. "
                "If you provide 'answer', 'search' MUST be null."
            ),
        )
    )

    search: Optional[constr(strip_whitespace=True, min_length=3, max_length=200)] = (
        Field(
            default=None,
            description=(
                "A concise (1–3 sentences) final, optimized video search query ONLY if you cannot answer. "
                "Include specific entities/objects/actions/time ranges when available; "
                "If you provide 'search', 'answer' MUST be null."
            ),
        )
    )
    
     # Provide EXACTLY ONE of {answer, search}
    subtitle: Optional[constr(strip_whitespace=True, min_length=1, max_length=200)] = (
        Field(
            default=None,
            description=(
                "Direct, self-contained answer to the question. Provide this ONLY if you are confident. "
                "If you provide 'answer', 'search' MUST be null."
            ),
        )
    )

    # --- Pydantic v2 validator ---

    @model_validator(mode="after")
    def _xor_answer_search(self):
        has_answer = self.answer is not None and len(self.answer.strip()) > 0
        has_search = self.search is not None and len(self.search.strip()) > 0
        if has_answer == has_search:
            raise ValueError(
                "Exactly one of 'answer' or 'search' must be provided (XOR)."
            )
        return self


class ReasoningTrajectory(BaseModel):
    reasoning_trajectory: List[Optional[AgentOutput]] = Field(
        description="The reasoning trajectory of the VLM"
    )
    is_found_answer: bool = Field(
        default=False, description="Whether the reasoning trajectory is found answer"
    )
    is_error: bool = Field(
        default=False, description="Whether the reasoning trajectory is error"
    )
    error_message: Optional[str] = Field(
        default=None, description="The error message of the reasoning trajectory"
    )
    answer: Optional[str] = Field(
        default=None, description="The answer of the reasoning trajectory"
    )
