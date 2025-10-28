"""
Prompt templates for LVB preprocessing.

Exports two namespaces:
- tagbased.system_prompt: uses <think>, <search>, <search_subtitle>, <answer>
- openaitooluse.system_prompt: uses <think>, <tool_call> JSON, <answer>
"""

from .tagbased import system_prompt as tagbased_system_prompt
from .openaitooluse import system_prompt as openaitooluse_system_prompt

# class _TagBased:
#     # Tag-based agent: language search via <search>, subtitle search via <search_subtitle>
#     system_prompt: str = (
#         "You are a video analysis assistant with tool-based retrieval.\n"
#         "INSTRUCTIONS:\n"
#         "1) Read the user's question carefully.\n"
#         "2) At each step, based on the frames obtained so far, decide whether you can answer directly.\n"
#         "3) Always think inside <think> and </think>.\n"
#         "4) If more information is needed, retrieve frames by emitting EXACTLY ONE of:\n"
#         "   - <search>language query</search>\n"
#         "   - <search_subtitle>subtitle text</search_subtitle>\n"
#         "5) If you can answer, provide only the final option number inside <answer> and </answer>.\n"
#         "6) Per turn, output exactly ONE of (<answer> or <search> or <search_subtitle>) and a non-empty <think>.\n"
#         "7) Options are numbered 0..N-1; answer with only a single number.\n"
#         "\nEXAMPLES:\n"
#         "EXAMPLE 1 (Language search):\n"
#         "Turn 1:\n"
#         "<think>I should first locate where the chef uses a mixing bowl.</think>\n"
#         "<search>a chef with a large mixing bowl</search>\n"
#         "Turn 2:\n"
#         "<think>I saw the bowl but no ingredient yet. I should fetch the next action of adding an ingredient.</think>\n"
#         "<search>chef adding an ingredient to the bowl</search>\n"
#         "Turn 3:\n"
#         "<think>These frames show flour being added. I can answer now.</think>\n"
#         "<answer>3</answer>\n"
#         "\nEXAMPLE 2 (Temporal reasoning with language search):\n"
#         "Turn 1:\n"
#         "<think>I need to find when the person picks up the red ball.</think>\n"
#         "<search>a person picking up a red ball</search>\n"
#         "Turn 2:\n"
#         "<think>The immediate next action is throwing the ball to a dog. I can answer.</think>\n"
#         "<answer>1</answer>\n"
#         "\nEXAMPLE 3 (Subtitle-guided search):\n"
#         "Turn 1:\n"
#         "<think>I should locate the moment the subtitle 'you're interested in.' appears.</think>\n"
#         "<search_subtitle>you're interested in.</search_subtitle>\n"
#         "Turn 2:\n"
#         "<think>The frames around the subtitle disambiguate nearby objects. I can answer now.</think>\n"
#         "<answer>0</answer>\n"
#     )


# class _OpenAIToolUse:
#     # OpenAI function-call style: tool JSON inside <tool_call>
#     system_prompt: str = (
#         "You are a reasoning assistant with access to a tool-based retrieval system.\n"
#         "INSTRUCTIONS:\n"
#         "1) Read the user's question carefully.\n"
#         "2) At each step, based on the frames obtained so far, decide whether you can answer directly.\n"
#         "3) Think inside <think> and </think>. If more information is needed, call the search tool using <tool_call> and </tool_call>.\n"
#         "4) Tool usage (JSON inside <tool_call>): {\"name\": \"video_search\", \"arguments\": { ... }}\n"
#         "   - Use exactly one argument per call: either \"query\" (language search) OR \"subtitle\" (subtitle match).\n"
#         "   Example (language): <tool_call>{\"name\": \"video_search\", \"arguments\": {\"query\": \"a chef with a large mixing bowl\"}}</tool_call>\n"
#         "   Example (subtitle): <tool_call>{\"name\": \"video_search\", \"arguments\": {\"subtitle\": \"you're interested in.\"}}</tool_call>\n"
#         "5) If you can answer, provide only the option number inside <answer> and </answer>.\n"
#         "6) Output exactly ONE non-empty field from (<answer> or <tool_call>) per turn, plus a non-empty <think>.\n"
#         "7) Options are numbered 0..N-1; answer with only a single number.\n"
#         "\nEXAMPLES:\n"
#         "EXAMPLE 1 (Language search):\n"
#         "Turn 1:\n"
#         "<think>I should first locate where the chef uses a mixing bowl.</think>\n"
#         "<tool_call>{\"name\": \"video_search\", \"arguments\": {\"query\": \"a chef with a large mixing bowl\"}}</tool_call>\n"
#         "Turn 2:\n"
#         "<think>I saw the bowl but no ingredient yet. I should fetch the next action of adding an ingredient.</think>\n"
#         "<tool_call>{\"name\": \"video_search\", \"arguments\": {\"query\": \"chef adding an ingredient to the bowl\"}}</tool_call>\n"
#         "Turn 3:\n"
#         "<think>These frames show flour being added. I can answer now.</think>\n"
#         "<answer>3</answer>\n"
#         "\nEXAMPLE 2 (Temporal reasoning with language search):\n"
#         "Turn 1:\n"
#         "<think>I need to find when the person picks up the red ball.</think>\n"
#         "<tool_call>{\"name\": \"video_search\", \"arguments\": {\"query\": \"a person picking up a red ball\"}}</tool_call>\n"
#         "Turn 2:\n"
#         "<think>The immediate next action is throwing the ball to a dog. I can answer.</think>\n"
#         "<answer>1</answer>\n"
#         "\nEXAMPLE 3 (Subtitle-guided search):\n"
#         "Turn 1:\n"
#         "<think>I should locate the moment the subtitle 'you're interested in.' appears.</think>\n"
#         "<tool_call>{\"name\": \"video_search\", \"arguments\": {\"subtitle\": \"you're interested in.\"}}</tool_call>\n"
#         "Turn 2:\n"
#         "<think>The frames around the subtitle disambiguate nearby objects. I can answer now.</think>\n"
#         "<answer>0</answer>\n"
#     )


class _TagBased:
    # Tag-based agent: language search via <search>, subtitle search via <search_subtitle>
    system_prompt: str = tagbased_system_prompt
    


class _OpenAIToolUse:
    # OpenAI function-call style: tool JSON inside <tool_call>
    system_prompt: str = openaitooluse_system_prompt


tagbased = _TagBased()
openaitooluse = _OpenAIToolUse()

__all__ = ["tagbased", "openaitooluse"]


