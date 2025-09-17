import re

from vseek.data.vseek_dm import AgentOutput


def parse_response_with_regex(
    content: str, just_thought: bool = False
) -> AgentOutput | None:
    """
    Parse response using tag-based format from system prompt.
    Extracts thought, answer, and search fields from <think>, <answer>, and <search> tags.
    """
    try:
        # Clean the content - remove extra whitespace and newlines
        content = content.strip()

        print(content)
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

        search_pattern = r"<search_subtitle>(.*?)</search_subtitle>"
        search_match = re.search(search_pattern, content, re.DOTALL)
        search_subtitle = search_match.group(1  ).strip() if search_match else None

        # Validate that we have the required fields
        if not thought or len(thought.strip()) < 5:
            print("Regex parser: Invalid or missing thought field")
            return None

        if thought and len(thought.strip()) > 1000:
            print(
                f"Regex parser: Thought field too long ({len(thought.strip())} chars), truncating to 200 chars"
            )
            thought = thought.strip()[:997] + "..."  # Truncate and add ellipsis

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
            
        if search_subtitle and len(search_subtitle.strip()) > 200:
            print(
                f"Regex parser: Search subtitle field too long ({len(search_subtitle.strip())} chars), truncating to 200 chars"
            )
            search_subtitle = search_subtitle.strip()[:197] + "..."  # Truncate and add ellipsis

        # Ensure exactly one of answer or search is provided (XOR validation)
        has_answer = answer is not None and len(answer.strip()) > 0
        has_search = search is not None and len(search.strip()) > 0
        has_search_subtitle = search_subtitle is not None and len(search_subtitle.strip()) > 0
        
        # Check if exactly one of answer, search, or search_subtitle is provided
        true_count = sum([has_answer, has_search, has_search_subtitle])
        
        if true_count != 1:  # Not exactly one true
            print(
                f"Regex parser: XOR validation failed - has_answer: {has_answer}, has_search: {has_search}, has_search_subtitle: {has_search_subtitle} (count: {true_count})"
            )
            if not just_thought:
                return None
            else:
                return AgentOutput(
                    thought=thought.strip(),
                    answer="",
                    search="",
                    search_subtitle="",
                )

        # Create AgentOutput object
        return AgentOutput(
            thought=thought.strip(),
            answer=answer.strip() if answer else None,
            search=search.strip() if search else None,
            search_subtitle=search_subtitle.strip() if search_subtitle else None,
        )

    except Exception as e:
        print(f"Regex parsing failed: {e}")
        return None
