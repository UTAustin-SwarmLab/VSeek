def find_prompt(prompt):
    full_prompt = f"""
    You are an intelligent agent that extracts atomic propositions from a video QA sample
    and generates a temporal-logic-like specification.

    IMPORTANT:
    - The input question may already contain options.
    - Use "Correct Answer" text as the source of truth whenever it is provided.
    - If only an answer index is provided, use it only to map to the option in the question.

    Stage 1: Proposition Extraction
    - Extract atomic propositions from:
      (a) the question context and
      (b) the correct answer text.
    - Each proposition must describe exactly one atomic fact/event/state.
    - Never merge multiple events in one proposition.
    - Do not hallucinate facts not grounded in the question or answer text.
    - Be concrete and specific (avoid vague text like "something happens").
    - Keep proposition text concise and directly usable in logic.
    - If subtitle/caption content is referenced, use: subtitle_'<exact text>'
      (no extra words like appears/says/mentions).

    Stage 2: Specification Generation
    Generate one specification using ONLY propositions from Stage 1.

    Hard rules:
    - Use each proposition exactly once.
    - Use ONLY operators: AND, OR, NOT, UNTIL.
    - Do not introduce new propositions.
    - The specification must reflect the question's temporal or logical intent.

    Temporal-operator decision guide:
    1) Use UNTIL only when the question expresses a true sequence trigger:
       - "before X" -> answer_event UNTIL X
       - "after X" / "what happens when X" -> X UNTIL answer_event
       - "between X and Y" -> X UNTIL (answer_event AND Y)
    2) Use AND for co-existing facts/attributes (identity + property, subject + color, etc.).
    3) Use OR only for explicit disjunction in the question/answer.
    4) Use NOT only for explicit negation in the question/answer.
    5) If no temporal cue exists, avoid UNTIL.

    Quality checks before final output:
    - For static attribute QA (color, count, identity, location), prefer AND or a single proposition;
      do not force UNTIL.
    - Verify operator choice matches wording: before/after/when/between/while/because.
    - Ensure the answer event/fact from Correct Answer is represented explicitly.

    Examples

    Example 1
    Question: "How did the girl feel before turning on the computer?"
    Correct Answer: "The girl feels happy"
    Output:
    {{
      "proposition": ["girl feels happy", "girl turns on computer"],
      "specification": "\"girl feels happy\" UNTIL \"girl turns on computer\""
    }}

    Example 2
    Question: "What color is the man's shirt while he is cooking?"
    Correct Answer: "Blue"
    Output:
    {{
      "proposition": ["man cooks", "mans_shirt_is_blue"],
      "specification": "\"man cooks\" AND \"mans_shirt_is_blue\""
    }}

    Example 3
    Question: "What happens when the red robot starts blinking?"
    Correct Answer: "The green robot turns on and moves"
    Output:
    {{
      "proposition": ["red robot starts blinking", "green robot turns on and moves"],
      "specification": "\"red robot starts blinking\" UNTIL \"green robot turns on and moves\""
    }}

    Example 4
    Question: "What happened before the subtitle 'Hello Mr. Anderson' appeared?"
    Correct Answer: "The man gets up"
    Output:
    {{
      "proposition": ["man gets up", "subtitle_'Hello Mr. Anderson'"],
      "specification": "\"man gets up\" UNTIL \"subtitle_'Hello Mr. Anderson'\""
    }}

    Now process this input:
    {{
      "prompt": "{prompt}"
    }}

    Return ONLY valid JSON:
    {{
      "proposition": ["<proposition1>", "<proposition2>", "..."],
      "specification": "<specification_in_temporal_logic>"
    }}

    Ensure at least one proposition and a non-empty specification.
    """
    return full_prompt