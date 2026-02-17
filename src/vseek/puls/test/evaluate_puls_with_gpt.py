import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

from openai import OpenAI
from tqdm import tqdm


SYSTEM_PROMPT = """
You are an expert evaluator of temporal logic extraction from video QA.

You will evaluate a predicted:
1) proposition list
2) temporal logic specification

Return ONLY valid JSON with the required schema.
Do not include markdown or extra text.
""".strip()


def build_user_prompt(
    question: str,
    candidates: list[str],
    correct_choice: Any,
    proposition: list[str],
    specification: str,
) -> str:
    options_block = "\n".join([f"{idx}. {opt}" for idx, opt in enumerate(candidates)])
    return f"""
Evaluate whether this PULS output is correct for the QA sample.

### QA Sample
Question: {question}
Options:
{options_block}
Correct Answer Index: {correct_choice}

### Predicted PULS
proposition: {json.dumps(proposition, ensure_ascii=True)}
specification: {json.dumps(specification, ensure_ascii=True)}

### Evaluation criteria (score each from 0 to 2)
1) Inclusion of all potential subjects and events
   - 2: Includes all key subjects/events needed to answer correctly; no critical missing item.
   - 1: Mostly complete but missing at least one non-trivial subject/event.
   - 0: Misses multiple key subjects/events or is largely irrelevant.

2) Temporal alignment of subjects/events
   - 2: The ordering/temporal relationship among events matches the question and correct answer.
   - 1: Partially aligned; one notable temporal mismatch or ambiguity.
   - 0: Temporal relationships are wrong or unsupported by the QA.

3) Correct temporal operators between subjects and events
   - 2: Operators (AND/OR/NOT/UNTIL or symbolic equivalents) are appropriate and logically consistent.
   - 1: Minor operator issue but mostly acceptable.
   - 0: Operator usage is incorrect, contradictory, or invalid.

Important rules:
- Judge only against the QA sample and correct answer.
- Penalize hallucinated events that are not grounded in question/answer.
- Prefer atomic propositions over merged multi-event clauses.
- If the current PULS is wrong, provide a revised proposition/specification.
- Revised specification can use either textual operators (AND/OR/NOT/UNTIL) or symbolic operators (&/|/!/U), but must be logically coherent.

Return JSON with EXACTLY this schema:
{{
  "inclusion": {{
    "score": 0,
    "reason": "",
    "missing_or_extra": []
  }},
  "temporal_alignment": {{
    "score": 0,
    "reason": ""
  }},
  "operator_correctness": {{
    "score": 0,
    "reason": ""
  }},
  "overall_score": 0,
  "verdict": "correct",
  "needs_revision": false,
  "revised_proposition": [],
  "revised_specification": ""
}}

Constraints:
- Scores must be integers.
- overall_score must equal inclusion.score + temporal_alignment.score + operator_correctness.score.
- verdict must be one of: "correct", "partially_correct", "incorrect".
- needs_revision must be true unless verdict is "correct".
- If verdict is "correct", revised_proposition and revised_specification can be empty.
- If verdict is not "correct", revised_proposition and revised_specification must be non-empty.
""".strip()


def parse_json_object(raw_text: str) -> dict[str, Any]:
    raw_text = raw_text.strip()
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start >= 0 and end > start:
        candidate = raw_text[start : end + 1]
        return json.loads(candidate)
    raise ValueError("Could not parse model output as JSON object.")


def sanitize_entry_question(question: str) -> str:
    if not isinstance(question, str):
        return ""
    return question.split("Question:")[-1].strip()


def load_data(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}, got {type(data)}")
    return data


def sample_indices(total: int, max_samples: int | None, seed: int) -> list[int]:
    indices = list(range(total))
    if max_samples is None or max_samples >= total:
        return indices
    rng = random.Random(seed)
    rng.shuffle(indices)
    return sorted(indices[:max_samples])


def evaluate_single_entry(client: OpenAI, model: str, entry: dict[str, Any]) -> dict[str, Any]:
    question = sanitize_entry_question(entry.get("question", ""))
    candidates = entry.get("candidates", [])
    correct_choice = entry.get("correct_choice")
    puls = entry.get("puls", {})
    proposition = puls.get("proposition", []) if isinstance(puls, dict) else []
    specification = puls.get("specification", "") if isinstance(puls, dict) else ""

    user_prompt = build_user_prompt(
        question=question,
        candidates=candidates if isinstance(candidates, list) else [],
        correct_choice=correct_choice,
        proposition=proposition if isinstance(proposition, list) else [],
        specification=specification if isinstance(specification, str) else "",
    )

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    raw = response.choices[0].message.content
    if raw is None:
        raise ValueError("Empty response content from model.")
    judge = parse_json_object(raw)

    # Minimal post-check and normalization
    for field in ("inclusion", "temporal_alignment", "operator_correctness"):
        if field not in judge or not isinstance(judge[field], dict):
            raise ValueError(f"Missing field: {field}")
        if "score" not in judge[field]:
            raise ValueError(f"Missing score in: {field}")
        judge[field]["score"] = int(judge[field]["score"])

    computed_total = (
        judge["inclusion"]["score"]
        + judge["temporal_alignment"]["score"]
        + judge["operator_correctness"]["score"]
    )
    judge["overall_score"] = int(judge.get("overall_score", computed_total))
    if judge["overall_score"] != computed_total:
        judge["overall_score"] = computed_total

    verdict = str(judge.get("verdict", "")).strip().lower()
    if verdict not in {"correct", "partially_correct", "incorrect"}:
        # Fall back based on score
        if computed_total == 6:
            verdict = "correct"
        elif computed_total >= 3:
            verdict = "partially_correct"
        else:
            verdict = "incorrect"
    judge["verdict"] = verdict

    judge["needs_revision"] = bool(judge.get("needs_revision", verdict != "correct"))
    if verdict == "correct":
        judge["needs_revision"] = False

    revised_prop = judge.get("revised_proposition", [])
    revised_spec = judge.get("revised_specification", "")
    if not isinstance(revised_prop, list):
        revised_prop = []
    if not isinstance(revised_spec, str):
        revised_spec = ""
    judge["revised_proposition"] = revised_prop
    judge["revised_specification"] = revised_spec

    return judge


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "num_evaluated": 0,
            "avg_inclusion": 0.0,
            "avg_temporal_alignment": 0.0,
            "avg_operator_correctness": 0.0,
            "avg_overall_score": 0.0,
            "verdict_counts": {},
            "revision_rate": 0.0,
        }

    n = len(records)
    inc = sum(r["judge"]["inclusion"]["score"] for r in records)
    align = sum(r["judge"]["temporal_alignment"]["score"] for r in records)
    op = sum(r["judge"]["operator_correctness"]["score"] for r in records)
    overall = sum(r["judge"]["overall_score"] for r in records)

    verdict_counts: dict[str, int] = {}
    revisions = 0
    for r in records:
        verdict = r["judge"]["verdict"]
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
        if r["judge"]["needs_revision"]:
            revisions += 1

    return {
        "num_evaluated": n,
        "avg_inclusion": round(inc / n, 4),
        "avg_temporal_alignment": round(align / n, 4),
        "avg_operator_correctness": round(op / n, 4),
        "avg_overall_score": round(overall / n, 4),
        "verdict_counts": verdict_counts,
        "revision_rate": round(revisions / n, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate PULS proposition/specification quality with GPT."
    )
    parser.add_argument("--input_json", required=True, type=str, help="Path to puls.json")
    parser.add_argument(
        "--output_jsonl",
        required=True,
        type=str,
        help="Path to save per-entry GPT judgments (jsonl).",
    )
    parser.add_argument(
        "--summary_json",
        default=None,
        type=str,
        help="Optional path to save aggregate summary JSON.",
    )
    parser.add_argument("--model", default="gpt-4o", type=str, help="OpenAI model name.")
    parser.add_argument(
        "--max_samples",
        default=None,
        type=int,
        help="Optional random sample size from input set.",
    )
    parser.add_argument("--seed", default=42, type=int, help="Seed for random sampling.")
    args = parser.parse_args()

    input_path = Path(args.input_json)
    output_path = Path(args.output_jsonl)
    summary_path = Path(args.summary_json) if args.summary_json else None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)

    entries = load_data(input_path)
    selected_indices = sample_indices(len(entries), args.max_samples, args.seed)
    selected_entries = [(idx, entries[idx]) for idx in selected_indices]

    client = OpenAI()
    records: list[dict[str, Any]] = []

    with output_path.open("w", encoding="utf-8") as fout:
        for idx, entry in tqdm(selected_entries, desc="GPT evaluating PULS"):
            try:
                judge = evaluate_single_entry(client, args.model, entry)
                record = {
                    "index": idx,
                    "question": sanitize_entry_question(entry.get("question", "")),
                    "puls": entry.get("puls", {}),
                    "judge": judge,
                }
            except Exception as exc:
                record = {
                    "index": idx,
                    "question": sanitize_entry_question(entry.get("question", "")),
                    "puls": entry.get("puls", {}),
                    "judge_error": str(exc),
                }
            records.append(record)
            fout.write(json.dumps(record, ensure_ascii=True) + "\n")

    valid_records = [r for r in records if "judge" in r]
    summary = aggregate(valid_records)
    summary["num_total_selected"] = len(selected_entries)
    summary["num_errors"] = len(selected_entries) - len(valid_records)
    summary["model"] = args.model
    summary["input_json"] = str(input_path)
    summary["output_jsonl"] = str(output_path)

    if summary_path is not None:
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=True)

    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
