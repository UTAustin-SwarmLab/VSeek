#!/usr/bin/env python3
"""
Run VideoTree QA on long videos using the realtime implementation.

Usage:
  # Single video with question (GPT-5)
  python run_realtime_videotree.py --video /path/to/video.mp4 --question "What happens in this video?"

  # With MCQ options and ground truth
  python run_realtime_videotree.py \\
    --video /path/to/video.mp4 \\
    --question "What action is performed?" \\
    --choices "Option A" "Option B" "Option C" "Option D" "Option E" \\
    --answer "A"

  # Using local Qwen3-VL (OpenAI-compatible server at localhost:8000)
  python run_realtime_videotree.py --video /path/to/video.mp4 --question "..." --model-variant qwen4b_instruct

  # From a JSON manifest (list of entries)
  python run_realtime_videotree.py --manifest entries.json --output results.json

Environment:
  OPENAI_API_KEY  Required when using GPT models (gpt5 variant).
"""
import argparse
import json
import sys
from pathlib import Path

# Add parent to path so we can import from baselines/VideoTree
sys.path.insert(0, str(Path(__file__).resolve().parent))
from collections import Counter

from realtime_videotree_qa import RuntimeVideoTreeConfig, RuntimeVideoTreeQA


def main():
    parser = argparse.ArgumentParser(description="Run VideoTree QA on long videos (realtime implementation)")
    parser.add_argument("--video", type=str, help="Path to video file")
    parser.add_argument("--question", type=str, help="Question to answer")
    parser.add_argument(
        "--choices",
        type=str,
        nargs="+",
        help="MCQ options (e.g. --choices 'A text' 'B text' 'C text')",
    )
    parser.add_argument("--answer", type=str, help="Ground truth answer (optional)")
    parser.add_argument(
        "--manifest",
        type=str,
        help="JSON file with list of entries: [{question, video/video_path, choices?, answer?}]",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="-",
        help="Output JSON file (default: stdout)",
    )
    parser.add_argument(
        "--model-variant",
        type=str,
        default="gpt5",
        choices=["gpt5", "qwen4b_instruct"],
        help="Model variant: gpt5 (OpenAI) or qwen4b_instruct (local)",
    )
    parser.add_argument(
        "--n-passes",
        type=int,
        default=4,
        help="Number of QA passes for majority voting (default: 4)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=512,
        help="Max frames to read from video at 1 FPS (default: 512)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key (or set OPENAI_API_KEY)",
    )
    args = parser.parse_args()

    # Build entries
    entries = []
    if args.manifest:
        with open(args.manifest) as f:
            data = json.load(f)
            entries = data if isinstance(data, list) else data.get("entries", data.get("data", []))
        if not entries:
            print("No entries found in manifest.", file=sys.stderr)
            sys.exit(1)
    elif args.video and args.question:
        entry = {
            "video": args.video,
            "question": args.question,
            "choices": args.choices,
            "answer": args.answer,
        }
        entries = [entry]
    else:
        parser.error("Provide either (--video and --question) or --manifest")

    # Config
    config = RuntimeVideoTreeConfig(
        model_variant=args.model_variant,
        api_key=args.api_key,
        n_passes=args.n_passes,
        max_frames_to_read=args.max_frames,
    )
    agent = RuntimeVideoTreeQA(config)

    # Run
    results = []
    for i, entry in enumerate(entries):
        print(
            f"[{i+1}/{len(entries)}] {entry.get('video', entry.get('video_path', '?'))}: {entry.get('question', '')[:60]}...",
            file=sys.stderr,
        )
        try:
            out = agent.answer_entry(entry)

            # Majority vote over parsed answers
            parsed = [p for p in out["parsed_answers"] if p]
            majority = Counter(parsed).most_common(1)[0][0] if parsed else ""

            result = {
                "question": out["question"],
                "video_path": out["video_path"],
                "ground_truth": out.get("ground_truth_answer"),
                "parsed_answers": out["parsed_answers"],
                "majority_vote": majority,
                "num_keyframes": out["num_keyframes"],
                "n_passes": out["n_passes"],
            }
            if out.get("choices"):
                result["choices"] = out["choices"]
            results.append(result)

            correct = result["ground_truth"] and str(majority).upper() == str(result["ground_truth"]).upper()
            status = "✓" if correct else "✗"
            print(f"  -> majority={majority} gt={result['ground_truth']} {status}", file=sys.stderr)
        except Exception as e:
            print(f"  -> ERROR: {e}", file=sys.stderr)
            results.append(
                {
                    "question": entry.get("question"),
                    "video_path": entry.get("video") or entry.get("video_path"),
                    "error": str(e),
                }
            )

    # Write output
    output = {"results": results}
    if args.output == "-":
        print(json.dumps(output, indent=2))
    else:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Wrote {args.output}", file=sys.stderr)

    # Summary
    valid = [r for r in results if "error" not in r and r.get("ground_truth")]
    if valid:
        correct = sum(1 for r in valid if str(r["majority_vote"]).upper() == str(r["ground_truth"]).upper())
        print(f"\nAccuracy: {correct}/{len(valid)} = {100*correct/len(valid):.1f}%", file=sys.stderr)


if __name__ == "__main__":
    main()
