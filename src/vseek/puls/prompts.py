def find_prompt(prompt):
    full_prompt = f"""
    You are an intelligent agent designed to extract primitives from a question and its correct answer, and generate a temporal logic specification that captures the sequence of primitives implied by the question. 
    A primitive is a single entity containing either an object, action or a combination of a single object and action. 
    You will operate in two stages: (1) primitive extraction and (2) logic specification generation.

    Stage 1: Proposition Extraction

    Given an input question and its correct answer about a video, extract the atomic primitives that contain a single object, activity, event or object-object relationsip referenced in the question and/or its correct answer. You must extract all primitives explicity referenced in the question and avoid making assumptions or inferring any additional events. You must avoid pieicing multiple atomic primitives into a single primitive using keywords that combine multiple primitives such as 'and', 'or', 'not', 'until'. Note that the list of keywords are not exhaustive and you must avoid piecing multiple primitives into a single primitive using any of the keywords.
    Do not include ambiguous primitives that lack specificity. For instance, phrases like "guy does something" are ambiguous and should be omitted. Instead, focus on concrete actions or relationships. For example, given the prompt "In a bustling park, a child kicks a ball. What happens when the ball hits the bench?", the correct primitives are ["child kicks ball", "ball hits bench"].
    If a primitive mentions subtitles/captions, the format of the primitive is the word "subtile_" followed by the subtitle in single quotes. Do NOT add words like "appears"/"says"/"mentions" after the subtitle; follow this format to create the individual primitive. For example, given the prompt "After the man gets up, what happens after the subtitle 'Hello Mr. Anderson' appeared?", the correct primitives are ["man gets up", "subtitle_'Hello Mr. Anderson'"].

    Stage 2: TL Specification Generation

    Using only the list of the propositions extracted in Stage 1, generate a single Temporal Logic (TL) specification that catpures the sequence of logical structure implied by the question. 

    Rules:
    - The formula must use each proposition **exactly once**
    - Use only the TL operators: `AND`, `OR`, `NOT`, `UNTIL`
    - Do **not** infer new events or rephrase propositions.
    - The formula should reflect the temporal or logical relationships between the propositions under which the question would be understandable.


    **Examples**

    Example 1: "In a sunny meadow, a child plays with a kite and runs around. What does the child do after falling?
    Options: A. Runs around B. Plays with kite C. Calls for help D. Runs around and plays with kite
    Correct Answer: C
    Output:
    {{
    "proposition": ["child plays with kite", "child runs around", "child falls", "child calls for help"],
    "specification": "(child plays with kite AND child runs around) UNTIL child falls UNTIL child calls for help"
    }}

    Example 2: "In a dimly lit room, two robots stand silently. What happens when the red robot starts blinking or the green robot does not turn off?"
    Options: A. The red robot stops blinking B. The green robot turns on and moves C. The red robot turns its head D. The green robot starts blinking
    Correct Answer: B
    Output:
    {{
    "proposition": ["robots stand silently", "red robot starts blinking", "green robot turns off", "green robot turns on and moves"],
    "specification": "robots stand silently UNTIL (red robot starts blinking OR NOT green robot turns off) UNTIL green robot turns on and moves"
    }}

    Example 3: "Inside a cave, a man holds a lantern. What happens when the man sees the dragon?"
    Options: A. The man puts out the lantern B. The man runs away C. The man throws the lantern D. The man hides behind the rock
    Correct Answer: D
    Output:
    {{
    "proposition": ["man holds lantern", "man sees dragon", "man hides behind the rock"],
    "specification": "man holds lantern UNTIL man sees dragon UNTIL man hides behind the rock"
    }}

    Example 4: "What happened on the screen before a man in black armor with glasses spoke into the microphone in front of a golden sky and the subtitles said 'country uh so we've seen significant'?"
    Options: A. The man waves his hand B. The man leaves C. A woman enters the scene D. The light behind the man turns on
    Correct Answer: A
    Output:
    {{
    "proposition": ["man in black armor with glasses", "man speaks into the microphone", "man is in front of a golden sky", "subtitle_'country uh so we've seen significant'", "man waves his hand"],
    "specification": "man waves his hand UNTIL (man in black armor with glasses AND man speakes into the microphone AND man is in front of a golden sky) UNTIL (subtitle_'country uh so we've seen significant')"
    }}

    Example 5: "A news anchor with curled hair is wearing a pink blazer over a black base and sitting in front of the camera reading the news. What happened in between the woman entering the scene and the caption 'standards our climate editor Justin rout' appeared?"
    Options: A. The man waves his hand B. The man leaves C. A woman enters the scene D. The light behind the man turns on
    Correct Answer: D
    Output:
    {{
    "proposition": ["news anchor with curled hair", "news anchor has pink blazer", "news anchor sits over a black base", "news anchor sitting in front of the camera reading the news", "subtitle_'standards our climate editor Justin rout'", "woman enters the scene", "light behind the man turns on"],
    "specification": "(news anchor with curled hair AND news anchor has pink blazer AND news anchor sits over a black base AND news anchor sitting in front of the camera reading the news) UNTIL woman enters the scene UNTIL (light behind the man turns on AND subtitle_'standards our climate editor Justin rout')"
    }}


    Example 6: "How did the girl feel before turning on the computer?"
    Options: A. The girl feels happy B. The girl feels sad C. The girl feels excited D. The girl feels bored
    Correct Answer: A
    Output:
    {{
    "proposition": ["girl turns on computer", "girl feels happy"],
    "specification": "girl feels happy UNTIL girl turns on computer"
    }}

    **Now process the following prompt:**
    Input:
    {{
    "prompt": "{prompt}"
    }}

    Expected Output (only output the following JSON structure — nothing else):
    {{
    "proposition": ["<proposition1>", "<proposition2>", ...],
    "specification": "<specification_in_temporal_logic>"
    }}
    
    You must ensure that there is atleast one proposition in the list of propositions and a non empty temporal logic specification.
    """
    return full_prompt