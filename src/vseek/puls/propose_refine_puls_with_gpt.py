import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI
from tqdm import tqdm

from vseek.puls.llm import LLM
from vseek.puls.primitives import PULS, process_specification, resolve_correct_answer_text
from vseek.puls.test.evaluate_puls_with_gpt import (
    aggregate,
    evaluate_single_entry,
    load_data,
    parse_json_object,
    sample_indices,
    sanitize_entry_question,
)


def build_prompt_for_proposal(entry: dict[str, Any]) -> str:
    question = sanitize_entry_question(entry.get("question", ""))
    candidates = entry.get("candidates", [])
    correct_choice = entry.get("correct_choice")
    correct_answer_text = resolve_correct_answer_text(candidates, correct_choice)
    if correct_answer_text:
        return question + "\n Correct Answer: " + correct_answer_text
    return question + "\n Correct Answer Index (0-based): " + str(correct_choice)


def normalize_puls_output(proposition: list[Any], specification: Any) -> dict[str, Any]:
    proposition_as_str = [str(p) for p in proposition if str(p).strip()]
    specification_as_str = str(specification).strip()
    if not proposition_as_str or not specification_as_str:
        return {"proposition": [], "specification": ""}
    cleaned_props, processed_spec = process_specification(specification_as_str, proposition_as_str)
    return {"proposition": cleaned_props, "specification": processed_spec}


def apply_judge_revision(base_puls: dict[str, Any], judge: dict[str, Any]) -> tuple[dict[str, Any], bool, str | None]:
    improved_puls = base_puls
    if not judge.get("needs_revision", False):
        return improved_puls, False, None

    revised_prop = judge.get("revised_proposition", [])
    revised_spec = judge.get("revised_specification", "")
    candidate_refinement = normalize_puls_output(revised_prop, revised_spec)
    if candidate_refinement["proposition"] and candidate_refinement["specification"]:
        return candidate_refinement, True, None
    return improved_puls, False, "Judge requested revision but returned invalid revised fields."


def reason_with_judge_feedback(
    model: str,
    entry: dict[str, Any],
    proposed_puls: dict[str, Any],
    judge: dict[str, Any],
) -> tuple[dict[str, Any], str | None]:
    user_prompt = f"""
You are improving a temporal-logic extraction result.

Task:
- Use the original QA and the first-pass PULS and judge feedback on the first pass and the judge's proposed revision.
- Re-reason and produce the BEST final proposition/specification.
- Prefer judge feedback, but do not blindly copy if inconsistent.

QA:
question: {json.dumps(sanitize_entry_question(entry.get("question", "")), ensure_ascii=True)}
options: {json.dumps(entry.get("candidates", []), ensure_ascii=True)}
correct_choice: {json.dumps(entry.get("correct_choice"), ensure_ascii=True)}

First-pass PULS:
proposition: {json.dumps(proposed_puls.get("proposition", []), ensure_ascii=True)}
specification: {json.dumps(proposed_puls.get("specification", ""), ensure_ascii=True)}

Judge feedback:
inclusion: {json.dumps(judge.get("inclusion", {}), ensure_ascii=True)}
temporal_alignment: {json.dumps(judge.get("temporal_alignment", {}), ensure_ascii=True)}
operator_correctness: {json.dumps(judge.get("operator_correctness", {}), ensure_ascii=True)}
verdict: {json.dumps(judge.get("verdict", ""), ensure_ascii=True)}
needs_revision: {json.dumps(judge.get("needs_revision", False), ensure_ascii=True)}
judge_revised_proposition: {json.dumps(judge.get("revised_proposition", []), ensure_ascii=True)}
judge_revised_specification: {json.dumps(judge.get("revised_specification", ""), ensure_ascii=True)}

Requirements:
- Return only JSON with keys "proposition" and "specification".
- proposition must be a non-empty list of atomic strings.
- specification must be non-empty and only use these operators conceptually: AND, OR, NOT, UNTIL (or symbols & | ! U).
- Keep it grounded in the question and correct answer, avoid distractors.
""".strip()

    response = OpenAI().chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": "You refine PULS outputs into better final temporal logic. Ensure that the ouputs are in json format."},
            {"role": "user", "content": user_prompt},
        ],
    )
    raw = response.choices[0].message.content
    if raw is None:
        return {"proposition": [], "specification": ""}, "Empty response from feedback reasoner."
    parsed = parse_json_object(raw)
    normalized = normalize_puls_output(parsed.get("proposition", []), parsed.get("specification", ""))
    if not normalized["proposition"] or not normalized["specification"]:
        return normalized, "Feedback reasoner returned invalid proposition/specification."
    return normalized, None


def evaluate_pipeline_pass(
    llm: LLM,
    reasoner_model: str,
    evaluator_model: str,
    entry: dict[str, Any],
    propose_prompt: str,
) -> dict[str, Any]:
    proposed = PULS(llm, propose_prompt)
    if not isinstance(proposed, dict):
        raise ValueError("PULS proposer did not return a dict.")
    if "proposition" not in proposed or "specification" not in proposed:
        raise ValueError("PULS proposer output is missing proposition/specification.")

    proposed_puls = normalize_puls_output(proposed.get("proposition", []), proposed.get("specification", ""))
    eval_entry = dict(entry)
    eval_entry["puls"] = proposed_puls
    judge = evaluate_single_entry(OpenAI(), evaluator_model, eval_entry)
    improved_puls, used_revision, revision_error = apply_judge_revision(proposed_puls, judge)
    reasoned_puls = improved_puls
    reasoning_error = None
    used_feedback_reasoning = False
    if judge.get("needs_revision", False):
        reasoned_candidate, reasoning_error = reason_with_judge_feedback(
            model=reasoner_model,
            entry=entry,
            proposed_puls=proposed_puls,
            judge=judge,
        )
        if reasoned_candidate["proposition"] and reasoned_candidate["specification"]:
            reasoned_puls = reasoned_candidate
            used_feedback_reasoning = True

    return {
        "proposed_puls": proposed_puls,
        "judge": judge,
        "improved_puls": reasoned_puls,
        "used_revision": used_revision,
        "revision_error": revision_error,
        "used_feedback_reasoning": used_feedback_reasoning,
        "feedback_reasoning_error": reasoning_error,
    }


def process_single_entry(
    idx: int,
    entry: dict[str, Any],
    proposer_model: str,
    evaluator_model: str,
    verification_pass: bool,
) -> tuple[int, dict[str, Any], dict[str, Any]]:
    updated_entry = dict(entry)
    debug_record: dict[str, Any] = {
        "index": idx,
        "question": sanitize_entry_question(entry.get("question", "")),
        "options": entry.get("candidates", []),
        "propose_model": proposer_model,
        "evaluate_model": evaluator_model,
    }

    llm = LLM(model=proposer_model)
    propose_prompt = build_prompt_for_proposal(updated_entry)
    first_pass = evaluate_pipeline_pass(llm, proposer_model, evaluator_model, updated_entry, propose_prompt)
    debug_record["first_pass"] = first_pass

    selected_pass = first_pass
    if verification_pass:
        second_pass = evaluate_pipeline_pass(llm, proposer_model, evaluator_model, updated_entry, propose_prompt)
        debug_record["second_pass"] = second_pass
        first_score = int(first_pass["judge"].get("overall_score", 0))
        second_score = int(second_pass["judge"].get("overall_score", 0))
        if second_score > first_score:
            selected_pass = second_pass
            debug_record["selected_pass"] = "second"
        else:
            debug_record["selected_pass"] = "first"
    else:
        debug_record["selected_pass"] = "first"

    updated_entry["puls"] = selected_pass["improved_puls"]
    # updated_entry["puls_feedback"] = selected_pass["judge"]
    debug_record["judge"] = selected_pass["judge"]
    debug_record["used_revision"] = selected_pass["used_revision"]
    if selected_pass["revision_error"]:
        debug_record["revision_error"] = selected_pass["revision_error"]
    debug_record["used_feedback_reasoning"] = selected_pass.get("used_feedback_reasoning", False)
    if selected_pass.get("feedback_reasoning_error"):
        debug_record["feedback_reasoning_error"] = selected_pass["feedback_reasoning_error"]
    debug_record["final_puls"] = selected_pass["improved_puls"]
    return idx, updated_entry, debug_record


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Propose PULS with primitives, evaluate using GPT, and refine from feedback."
    )
    parser.add_argument("--input_json", required=True, type=str, help="Path to input JSON list.")
    parser.add_argument("--output_json", required=True, type=str, help="Path to save updated entries.")
    parser.add_argument(
        "--output_jsonl",
        required=True,
        type=str,
        help="Path to save per-entry debug/judge records (jsonl).",
    )
    parser.add_argument(
        "--summary_json",
        default=None,
        type=str,
        help="Optional path to save aggregate summary JSON.",
    )
    parser.add_argument(
        "--proposer_model",
        default="gpt-5.2",
        type=str,
        help="Model used by existing PULS primitive generation.",
    )
    parser.add_argument(
        "--evaluator_model",
        default="gpt-5.2",
        type=str,
        help="Model used for PULS evaluation and revision feedback.",
    )
    parser.add_argument("--max_samples", default=None, type=int, help="Optional random sample size.")
    parser.add_argument("--seed", default=42, type=int, help="Seed for random sampling.")
    parser.add_argument("--num_workers", default=8, type=int, help="Parallel workers.")
    parser.set_defaults(verification_pass=True)
    parser.add_argument(
        "--verification_pass",
        dest="verification_pass",
        action="store_true",
        help="Run a second propose+judge+refine pass and select the better result (default: enabled).",
    )
    parser.add_argument(
        "--no_verification_pass",
        dest="verification_pass",
        action="store_false",
        help="Disable the second verification pass.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_json)
    output_path = Path(args.output_json)
    output_jsonl_path = Path(args.output_jsonl)
    summary_path = Path(args.summary_json) if args.summary_json else None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)

    entries = load_data(input_path)
    selected_indices = sample_indices(len(entries), args.max_samples, args.seed)
    selected_set = set(selected_indices)

    num_workers = max(1, int(args.num_workers))
    updated_by_index: dict[int, dict[str, Any]] = {}
    debug_records: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_index = {
            executor.submit(
                process_single_entry,
                idx,
                entries[idx],
                args.proposer_model,
                args.evaluator_model,
                args.verification_pass,
            ): idx
            for idx in selected_indices
        }
        for future in tqdm(as_completed(future_to_index), total=len(future_to_index), desc="Propose + evaluate + refine"):
            idx = future_to_index[future]
            try:
                result_idx, updated_entry, debug_record = future.result()
                updated_by_index[result_idx] = updated_entry
                debug_records.append(debug_record)
            except Exception as exc:
                debug_records.append(
                    {
                        "index": idx,
                        "question": sanitize_entry_question(entries[idx].get("question", "")),
                        "options": entries[idx].get("candidates", []),
                        "correct_choice": entries[idx].get("correct_choice"),
                        "judge_error": str(exc),
                    }
                )

    final_entries: list[dict[str, Any]] = []
    for idx, entry in enumerate(entries):
        if idx in selected_set and idx in updated_by_index:
            final_entries.append(updated_by_index[idx])
        else:
            final_entries.append(entry)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(final_entries, f, indent=2, ensure_ascii=True)

    debug_records.sort(key=lambda x: x["index"])
    with output_jsonl_path.open("w", encoding="utf-8") as fout:
        for record in debug_records:
            fout.write(json.dumps(record, ensure_ascii=True) + "\n")

    valid_records = [r for r in debug_records if "judge" in r]
    wrapped_records = [{"judge": r["judge"]} for r in valid_records]
    summary = aggregate(wrapped_records)
    summary["num_total_entries"] = len(entries)
    summary["num_selected"] = len(selected_indices)
    summary["num_errors"] = len(debug_records) - len(valid_records)
    summary["proposer_model"] = args.proposer_model
    summary["evaluator_model"] = args.evaluator_model
    summary["input_json"] = str(input_path)
    summary["output_json"] = str(output_path)
    summary["output_jsonl"] = str(output_jsonl_path)

    if summary_path is not None:
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=True)

    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
