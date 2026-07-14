import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI
from tqdm import tqdm


def parse_json_object(raw: str) -> dict[str, Any]:
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start < 0 or end <= start:
        raise ValueError("Model response did not contain a JSON object.")
    return json.loads(raw[start:end])


def resolve_correct_answer_text(candidates: list[Any], correct_choice: Any) -> str:
    if not candidates:
        return ""

    if isinstance(correct_choice, int):
        idx = correct_choice
    else:
        choice_str = str(correct_choice).strip()
        if choice_str.isdigit():
            idx = int(choice_str)
        elif re.fullmatch(r"[A-Za-z]", choice_str):
            idx = ord(choice_str.upper()) - ord("A")
        else:
            return ""

    if 0 <= idx < len(candidates):
        return str(candidates[idx])
    return ""


def sanitize_question(question: Any) -> str:
    question_text = str(question).strip()
    if "Question:" in question_text:
        question_text = question_text.split("Question:")[-1].strip()
    return question_text


def build_prompt(entry: dict[str, Any]) -> str:
    question = sanitize_question(entry.get("question", ""))
    candidates = entry.get("candidates", [])
    correct_choice = entry.get("correct_choice")
    correct_answer = resolve_correct_answer_text(candidates, correct_choice)

    answer_block = (
        f"Correct Answer: {correct_answer}"
        if correct_answer
        else f"Correct Answer Index (0-based): {correct_choice}"
    )

    return f"""
Extract direct primitives from this video QA sample.

Question:
{question}

Answer options:
{json.dumps(candidates, ensure_ascii=True)}

{answer_block}
""".strip()


def direct_primitive_prompt(user_prompt: str) -> list[dict[str, str]]:
    system_prompt = """
You extract atomic visual/text primitives from video QA samples.

Rules:
- Return primitives only;
- Use the correct answer text as the source of truth for answer content.
- Ignore distractor options that are not the correct answer.
- Extract primitives grounded in the question context and correct answer.
- Keep primitive strings concise, concrete, and human-readable.
- If subtitle/caption content is referenced, use exactly: subtitle_'<exact text>'.

Return only valid JSON with this shape:
{"proposition": ["<primitive1>", "<primitive2>"]}
""".strip()

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def clean_primitives(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []

    primitives: list[str] = []
    seen: set[str] = set()
    for value in values:
        primitive = re.sub(r"\s+", " ", str(value)).strip().strip('"')
        if not primitive:
            continue
        dedupe_key = primitive.casefold()
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        primitives.append(primitive)
    return primitives


def normalize_proposition_text(primitive: str) -> str:
    cleaned = re.sub(r"^[^a-zA-Z]+|[^a-zA-Z]+$", "", primitive)
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = cleaned.replace("'", "").replace("-", "_").lower()
    cleaned = re.sub(r"[^a-zA-Z0-9_]", "", cleaned)
    return cleaned


def build_and_specification(propositions: list[str]) -> str:
    if not propositions:
        return ""
    quoted = [f'"{proposition}"' for proposition in propositions]
    return " & ".join(quoted)


def extract_direct_primitives(client: OpenAI, model: str, entry: dict[str, Any]) -> dict[str, Any]:
    response = client.chat.completions.create(
        model=model,
        messages=direct_primitive_prompt(build_prompt(entry)),
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content
    if raw is None:
        raise ValueError("Model returned an empty response.")

    parsed = parse_json_object(raw)
    primitives = clean_primitives(parsed.get("proposition", []))
    normalized_propositions: list[str] = []
    for primitive in primitives:
        normalized = normalize_proposition_text(primitive)
        if normalized:
            normalized_propositions.append(normalized)
    specification = build_and_specification(normalized_propositions)
    return {"proposition": normalized_propositions, "specification": specification}


def load_entries(path: Path) -> list[dict[str, Any]]:
    with path.open("r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input JSON must contain a list of entries.")
    return data


def process_entry(idx: int, entry: dict[str, Any], model: str) -> tuple[int, dict[str, Any]]:
    updated_entry = dict(entry)
    client = OpenAI()
    try:
        updated_entry["puls"] = extract_direct_primitives(client, model, entry)
    except Exception as exc:
        updated_entry["puls"] = {"proposition": [], "specification": ""}
        updated_entry["puls_direct_error"] = str(exc)
    return idx, updated_entry


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract direct PULS primitives without temporal-logic generation or verification."
    )
    parser.add_argument("--input_json", required=True, type=str, help="Path to input JSON list.")
    parser.add_argument("--output_json", required=True, type=str, help="Path to save updated entries.")
    parser.add_argument("--model", default="gpt-5.2", type=str, help="OpenAI model for primitive extraction.")
    parser.add_argument("--num_workers", default=8, type=int, help="Parallel workers.")
    parser.add_argument("--max_samples", default=None, type=int, help="Optional prefix sample size.")
    args = parser.parse_args()

    input_path = Path(args.input_json)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    entries = load_entries(input_path)
    if args.max_samples is not None:
        entries = entries[: args.max_samples]

    results: list[dict[str, Any] | None] = [None] * len(entries)
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [
            executor.submit(process_entry, idx, entry, args.model)
            for idx, entry in enumerate(entries)
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting direct primitives"):
            idx, updated_entry = future.result()
            results[idx] = updated_entry

    with output_path.open("w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
