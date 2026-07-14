import asyncio
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from tqdm import tqdm
import hydra
from omegaconf import DictConfig

PROJECT_ROOT = "/home/hg22723/projects/VSeek-R1"
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from data.cgbench import CGBench
from data.lvb import LongVideoBench
from data.lvbench import LVBench
from data.mlvu import MLVU
from data.videomme import VideoMME
from scripts.baselines.VideoTree.realtime_videotree_qa import (
    RuntimeVideoTreeConfig,
    RuntimeVideoTreeQA,
)


def _build_dataset(cfg: DictConfig):
    if cfg.dataset.name == "lvb":
        return LongVideoBench(cfg)
    if cfg.dataset.name == "lvbench":
        return LVBench(cfg)
    if cfg.dataset.name == "videomme":
        return VideoMME(cfg)
    if cfg.dataset.name == "mlvu":
        return MLVU(cfg)
    if cfg.dataset.name == "cgbench":
        return CGBench(cfg)
    raise ValueError(f"Unsupported dataset: {cfg.dataset.name}")


def _entry_to_runtime_input(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "question": entry.get("question", ""),
        "answer": str(entry.get("correct_choice", "")),
        "video": entry.get("paths", {}).get("video_path", ""),
        "choices": entry.get("candidates", []),
    }


def _pass_at_k(n: int, c: int, k: int) -> float:
    import math

    if n < k:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - (math.comb(n - c, k) / math.comb(n, k))


def _write_results(results: list[dict[str, Any]], file_path: str) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


async def _process_entry(
    entry: dict[str, Any],
    qa: RuntimeVideoTreeQA,
    passes: int,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any] | None:
    async with semaphore:
        video_id = str(entry["metadata"]["video_id"])
        question_id = str(entry["metadata"]["id"])
        gt = str(entry.get("correct_choice", ""))
        runtime_input = _entry_to_runtime_input(entry)

        if not runtime_input["video"] or not os.path.exists(runtime_input["video"]):
            print(f"[skip] Missing video path: {runtime_input['video']}")
            return None

        try:
            out = await asyncio.to_thread(qa.answer_entry, runtime_input)
        except Exception as e:
            print(f"Failed processing video_id={video_id}, qid={question_id}: {e}")
            return None

        parsed = out.get("parsed_answers", [])
        counts = Counter([p for p in parsed if p])
        majority_vote = counts.most_common(1)[0][0] if counts else ""
        result = {
            "video_id": video_id,
            "question_id": question_id,
            "gt": gt,
            "pred": out.get("answers", []),
            "parsed_pred": parsed,
            "all_preds": out.get("answers", []),
            "all_parsed_preds": parsed,
            "majority_vote": majority_vote,
            "is_majority_correct": majority_vote == gt,
            "correct_count": sum(1 for p in parsed if p == gt),
            "total_passes": passes,
            "keyframe_indices": out.get("keyframe_indices", []),
            "num_keyframes": out.get("num_keyframes", 0),
            "width_tree_node": out.get("width_tree_node", []),
            "frame_relevance": out.get("frame_relevance", []),
        }
        return result


async def run_async(cfg: DictConfig) -> None:
    dataset = _build_dataset(cfg)
    entries = await asyncio.to_thread(dataset.load_data)

    output_dir = cfg.inference.output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "agent_results.json")

    results: list[dict[str, Any]] = []
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            results = json.load(f)

    done_qids = {str(r["question_id"]) for r in results}
    remaining = [e for e in entries if str(e["metadata"]["id"]) not in done_qids]

    passes = int(getattr(cfg.inference, "passes", 4))
    batch_size = int(getattr(cfg.inference, "batch_size", 4))
    semaphore = asyncio.Semaphore(batch_size)

    vt_cfg = RuntimeVideoTreeConfig(
        model_variant=str(getattr(cfg.inference, "videotree_model_variant", "gpt5")),
        model_name=getattr(cfg.inference, "videotree_model_name", None),
        api_key=getattr(cfg.inference, "videotree_api_key", None),
        base_url=getattr(cfg.inference, "videotree_base_url", None),
        n_passes=passes,
        temperature=float(getattr(cfg.inference, "temperature", 0.0)),
        max_output_tokens=int(getattr(cfg.inference, "max_output_tokens", 512)),
        extraction_fps=1.0 / max(1e-6, float(getattr(cfg.dataset, "desired_interval_in_sec", 1))),
        max_keyframes=int(getattr(cfg.inference, "max_images_per_turn", 64)),
        max_images_per_call=int(getattr(cfg.inference, "max_images_per_turn", 64)),
    )
    qa = RuntimeVideoTreeQA(vt_cfg)

    print(
        f"Running VideoTree runtime eval | dataset={cfg.dataset.name} "
        f"| remaining={len(remaining)} | passes={passes} | batch_size={batch_size}"
    )

    tasks = [_process_entry(e, qa, passes, semaphore) for e in remaining]
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Evaluating VideoTree"):
        out = await coro
        if out is None:
            continue
        results.append(out)
        _write_results(results, output_file)

    total = len(results)
    if total == 0:
        print("No results found.")
        return

    majority_correct = sum(1 for r in results if r.get("is_majority_correct", False))
    print(f"Majority vote accuracy: {majority_correct / total:.3f} ({majority_correct}/{total})")

    ks = [1, 2, 4, 5, 8, 10, 16]
    ks = sorted({k for k in ks if k <= passes} | ({passes} if passes > 1 else set()))
    for k in ks:
        avg = sum(_pass_at_k(passes, r.get("correct_count", 0), k) for r in results) / total
        print(f"Pass@{k}: {avg:.3f}")


@hydra.main(version_base=None, config_path="../src/vseek/config/retriever", config_name="config")
def main(cfg: DictConfig):
    asyncio.run(run_async(cfg))


if __name__ == "__main__":
    main()
