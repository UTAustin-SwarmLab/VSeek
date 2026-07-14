import argparse
import json
import re
import sys
from pathlib import Path

import cv2
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.lvb import LongVideoBench  # noqa: E402
from data.lvbench import LVBench  # noqa: E402
from data.mlvu import MLVU  # noqa: E402
from data.videomme import VideoMME  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser("Prepare VideoTree JSON files from VSeek dataset loaders.")
    parser.add_argument("--dataset_name", required=True, choices=["lvb", "lvbench", "videomme", "mlvu"])
    parser.add_argument("--dataset_path", required=True, type=str)
    parser.add_argument("--burned_path", default="", type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument(
        "--narration_source",
        default="subtitle",
        choices=["subtitle", "caption", "question_only"],
        help="subtitle: subtitle json; caption: external caption json; question_only: placeholder text.",
    )
    parser.add_argument(
        "--captions_json",
        default="",
        type=str,
        help="Optional caption JSON (video_id -> list[str] or str). Legacy uid keys are also supported.",
    )
    parser.add_argument("--max_examples", default=-1, type=int)
    return parser.parse_args()


def build_cfg(args):
    burned = args.burned_path if args.burned_path else args.dataset_path
    return OmegaConf.create(
        {
            "dataset": {
                args.dataset_name: {
                    "dataset_path": args.dataset_path,
                    "burned_path": burned,
                }
            },
            "retriever": {
                "window_size": 8,
                "index_path": "",
                "gpu_number": 0,
                "retrieval_model_path": "",
            },
        }
    )


def get_manager(dataset_name, cfg):
    if dataset_name == "lvb":
        return LongVideoBench(cfg)
    if dataset_name == "lvbench":
        return LVBench(cfg)
    if dataset_name == "videomme":
        return VideoMME(cfg)
    if dataset_name == "mlvu":
        return MLVU(cfg)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def strip_question_prefix(question):
    if not isinstance(question, str):
        return ""
    clean = question.strip()
    clean = clean.replace(
        "This is a multiple choice question. You must choose the correct answer as a number or letter of the option.",
        "",
    )
    clean = clean.replace(
        "This is a multiple choice question. You must choose the correct answer from the options with the number or letter of the option.",
        "",
    )
    clean = re.sub(r"^\s*Question:\s*", "", clean, flags=re.IGNORECASE)
    return clean.strip()


def normalize_candidates(candidates):
    out = [str(c).strip() for c in (candidates or [])]
    while len(out) < 5:
        out.append("")
    return out[:5]


def normalize_truth(correct_choice, candidates):
    if correct_choice is None:
        return -1
    if isinstance(correct_choice, int):
        return correct_choice if 0 <= correct_choice <= 4 else -1

    raw = str(correct_choice).strip()
    if raw.isdigit():
        val = int(raw)
        return val if 0 <= val <= 4 else -1

    upper = raw.upper()
    if upper in {"A", "B", "C", "D", "E"}:
        return "ABCDE".index(upper)

    # Last fallback: match candidate text exactly.
    norm_candidates = [str(c).strip() for c in candidates]
    if raw in norm_candidates:
        return norm_candidates.index(raw)
    return -1


def video_duration_seconds(video_path):
    if not video_path:
        return 0
    p = Path(video_path)
    if not p.exists():
        return 0
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    if fps <= 0:
        return 0
    return int(frame_count / fps)


def subtitles_to_narration(subtitle_path):
    if not subtitle_path:
        return ""
    p = Path(subtitle_path)
    if not p.exists():
        return ""
    try:
        data = json.loads(p.read_text())
    except Exception:
        return ""

    if isinstance(data, dict):
        data = data.get("subtitles", [])
    if not isinstance(data, list):
        return ""

    lines = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        text = item.get("line") or item.get("text") or ""
        if not text:
            continue
        # Keep VideoTree-like style with #C marker to help loc_pred slicing.
        lines.append(f"#C {idx}: {str(text).strip()}")
    return "\n".join(lines)


def captions_to_narration(captions_data, uid, video_id):
    if not captions_data:
        return ""
    value = captions_data.get(uid)
    if value is None and video_id:
        value = captions_data.get(video_id)
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        lines = []
        for i, cap in enumerate(value):
            text = str(cap).strip()
            if not text:
                continue
            if text.startswith("#C ") or text.startswith("#O "):
                lines.append(text)
            else:
                lines.append(f"#C {i}: {text}")
        return "\n".join(lines)
    return ""


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_cfg(args)
    manager = get_manager(args.dataset_name, cfg)
    entries = manager.load_data()
    if args.max_examples > 0:
        entries = entries[: args.max_examples]

    captions_data = {}
    if args.narration_source == "caption":
        if not args.captions_json:
            raise ValueError("--captions_json is required when --narration_source=caption")
        captions_path = Path(args.captions_json)
        if not captions_path.exists():
            raise FileNotFoundError(f"Caption JSON not found: {captions_path}")
        captions_data = json.loads(captions_path.read_text())

    data_json = {}
    anno_json = {}
    duration_json = {}

    for entry in entries:
        meta = entry.get("metadata", {})
        uid = str(meta.get("id", ""))
        video_id = str(meta.get("video_id", ""))
        if not uid:
            continue

        candidates = normalize_candidates(entry.get("candidates", []))
        truth = normalize_truth(entry.get("correct_choice"), candidates)
        question = strip_question_prefix(entry.get("question", ""))

        subtitle_path = entry.get("paths", {}).get("subtitle_path")
        if args.narration_source == "subtitle":
            narration = subtitles_to_narration(subtitle_path)
        elif args.narration_source == "caption":
            narration = captions_to_narration(captions_data, uid, video_id)
        else:
            narration = ""

        if not narration:
            narration = "No subtitle narration available for this sample."

        video_path = entry.get("paths", {}).get("video_path") or entry.get("paths", {}).get("raw_video_path")
        duration = video_duration_seconds(video_path)

        data_json[uid] = narration
        anno_json[uid] = {
            "question": question,
            "option 0": candidates[0],
            "option 1": candidates[1],
            "option 2": candidates[2],
            "option 3": candidates[3],
            "option 4": candidates[4],
            "truth": truth,
        }
        duration_json[uid] = duration

    (output_dir / "data.json").write_text(json.dumps(data_json, indent=2))
    (output_dir / "anno.json").write_text(json.dumps(anno_json, indent=2))
    (output_dir / "duration.json").write_text(json.dumps(duration_json, indent=2))
    print(f"Wrote {len(data_json)} entries to {output_dir}")


if __name__ == "__main__":
    main()
