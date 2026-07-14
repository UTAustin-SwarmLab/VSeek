import argparse
import asyncio
import base64
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import requests
from openai import AsyncOpenAI
from tqdm import tqdm
from omegaconf import OmegaConf


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
from data.prompts.tagbasedsummary import system_prompt as tagbased_summary_system_prompt
from data.videomme import VideoMME
from vseek.data.frame import VideoFrames


MODEL_PRESETS = {
    "gpt52": {
        "model": "gpt-5.2",
        "api_key_env": "OPENAI_API_KEY",
        "base_url": None,
    },
    "gemini": {
        "model": "gemini-3.1-pro-preview",
        "api_key_env": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
    },
}

DATASET_DEFAULTS = {
    "lvb": {
        "dataset_path": "/nas/mars/dataset/longvideobench/LongVideoBench/",
        "burned_path": "/nas/mars/dataset/longvideobench/",
    },
    "lvbench": {
        "dataset_path": "/nas/mars/dataset/LVBench",
        "burned_path": "/nas/mars/dataset/LVBench/videos",
    },
    "videomme": {
        "dataset_path": "/nas/mars/dataset/Video-MME",
        "burned_path": "/nas/mars/dataset/Video-MME/burn-subtitles",
    },
    "mlvu": {
        "dataset_path": "/nas/mars/dataset/MLVU/MLVU",
        "burned_path": "/nas/mars/dataset/MLVU/MLVU",
    },
    "cgbench": {
        "dataset_path": "/nas/mars/dataset/CGBench",
        "burned_path": "/nas/mars/dataset/CGBench/burn-subtitles",
    },
}


def build_options_string(candidates: list[str], dataset_name: str) -> str:
    if dataset_name in {"lvb", "mlvu", "cgbench"}:
        lines = [f"{idx}. {text}" for idx, text in enumerate(candidates)]
    elif dataset_name in {"lvbench", "videomme"}:
        lines = [f"{text}" for text in candidates]
    else:
        lines = [f"{idx}. {text}" for idx, text in enumerate(candidates)]
    return "\n".join(lines)


def parse_answer(answer: str | None) -> str:
    if answer is None:
        return ""
    answer_tag = re.findall(
        r"<\s*answer\s*>([\s\S]*?)<\s*/\s*answer\s*>",
        answer,
        flags=re.IGNORECASE,
    )
    target = answer_tag[-1].strip() if answer_tag else answer
    numbers = re.findall(r"\d+", target)
    letters = re.findall(r"[a-zA-Z]+", target)
    if numbers:
        return numbers[0]
    if letters:
        return letters[0]
    return ""


def _encode_frame(frame: Any, max_width: int, max_height: int, quality: int) -> str:
    height, width = frame.shape[:2]
    scale = min(max_width / width, max_height / height)
    if scale < 1.0:
        frame = cv2.resize(
            frame,
            (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
    ret, buffer = cv2.imencode(".jpg", frame, encode_params)
    if not ret:
        raise ValueError("Could not encode frame")
    return base64.b64encode(buffer).decode("utf-8")


def _build_dataset(dataset_name: str, args: argparse.Namespace):
    ds_cfg = {
        "name": dataset_name,
        dataset_name: {
            "dataset_path": args.dataset_path or DATASET_DEFAULTS[dataset_name]["dataset_path"],
            "burned_path": args.burned_path or DATASET_DEFAULTS[dataset_name]["burned_path"],
        },
    }
    cfg = OmegaConf.create(
        {
            "dataset": ds_cfg,
            "retriever": {
                "window_size": args.window_size,
                "index_path": args.index_root,
            },
        }
    )
    if dataset_name == "lvb":
        return LongVideoBench(cfg)
    if dataset_name == "lvbench":
        return LVBench(cfg)
    if dataset_name == "videomme":
        return VideoMME(cfg)
    if dataset_name == "mlvu":
        return MLVU(cfg)
    if dataset_name == "cgbench":
        return CGBench(cfg)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def _resolve_model_config(args: argparse.Namespace) -> tuple[str, str, str | None]:
    preset = MODEL_PRESETS[args.model_type]
    model_name = args.model_name or preset["model"]
    base_url = args.base_url if args.base_url is not None else preset["base_url"]
    api_key = args.api_key or os.getenv(preset["api_key_env"])
    if not api_key:
        raise ValueError(
            f"Missing API key for model_type={args.model_type}. "
            f"Set {preset['api_key_env']} or pass --api_key."
        )
    return model_name, api_key, base_url


def _extract_tool_action(content: str) -> dict[str, str] | None:
    search = re.findall(r"<search>([\s\S]*?)</search>", content, flags=re.IGNORECASE)
    if search:
        query = search[-1].strip()
        if query:
            return {"action": "search", "query": query}

    subtitle = re.findall(
        r"<search_subtitle>([\s\S]*?)</search_subtitle>",
        content,
        flags=re.IGNORECASE,
    )
    if subtitle:
        query = subtitle[-1].strip()
        if query:
            return {"action": "search_subtitle", "query": query}

    summary = re.findall(
        r"<search_summary>\s*</search_summary>",
        content,
        flags=re.IGNORECASE,
    )
    if summary:
        return {"action": "search_summary", "query": ""}

    return None


def _call_retriever(
    retriever_server_url: str,
    dataset_name: str,
    video_id: str | None,
    query: str,
    mode: str,
    topk: int,
) -> list[int]:
    endpoint = "search_subtitle" if mode == "subtitle" else "search"
    params: dict[str, Any] = {
        "query": query,
        "dataset_name": dataset_name,
        "topk": int(topk),
    }
    if video_id:
        params["video_id"] = video_id
    try:
        resp = requests.get(
            f"{retriever_server_url.rstrip('/')}/{endpoint}",
            params=params,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        idxs = data.get("subtitle_indices") or data.get("frame_indices") or []
        return [int(i) for i in idxs]
    except Exception:
        return []


def _uniform_system_prompt() -> str:
    return (
        "You are a helpful assistant. "
        "Look at the provided images sampled uniformly from a video and choose the correct option. "
        "Return only the option id (number or letter), with no extra text."
    )


async def _chat(
    client: AsyncOpenAI,
    model_name: str,
    messages: list[dict[str, Any]],
    temperature: float,
    max_tokens: int,
) -> str:
    resp = await client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


async def _run_uniform_once(
    client: AsyncOpenAI,
    model_name: str,
    entry: dict[str, Any],
    video_frames: VideoFrames,
    dataset_name: str,
    args: argparse.Namespace,
) -> tuple[str, list[dict[str, Any]]]:
    candidates = entry["candidates"]
    options_str = build_options_string(candidates, dataset_name)
    frames = video_frames.all_frames
    if not frames:
        return ""
    k = max(1, min(int(args.max_images_per_turn), len(frames)))
    # Keep dependency-free selection.
    indices = [round(i * (len(frames) - 1) / max(k - 1, 1)) for i in range(k)]
    selected = [frames[i] for i in indices]
    user_content: list[dict[str, Any]] = []
    for frame in selected:
        enc = _encode_frame(
            frame,
            max_width=args.max_image_width,
            max_height=args.max_image_height,
            quality=args.image_quality,
        )
        user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{enc}"}})
    user_content.append(
        {
            "type": "text",
            "text": f"Question: {entry['question']}\nOptions:\n{options_str}",
        }
    )
    messages = [
        {"role": "system", "content": _uniform_system_prompt()},
        {"role": "user", "content": user_content},
    ]
    pred = await _chat(
        client=client,
        model_name=model_name,
        messages=messages,
        temperature=args.temperature,
        max_tokens=args.max_output_tokens,
    )
    conversation = messages + [{"role": "assistant", "content": pred}]
    conversation = _compact_conversation_for_log(conversation)
    return pred, conversation


async def _run_agentic_once(
    client: AsyncOpenAI,
    model_name: str,
    entry: dict[str, Any],
    video_frames: VideoFrames,
    dataset_name: str,
    args: argparse.Namespace,
) -> tuple[str, list[dict[str, Any]]]:
    options_str = build_options_string(entry["candidates"], dataset_name)
    conversation: list[dict[str, Any]] = [
        {"role": "system", "content": tagbased_summary_system_prompt.strip()},
        {
            "role": "user",
            "content": f"Question: {entry['question']}\nOptions:\n{options_str}\n"
            "No frames are provided yet. Use one of <search>, <search_subtitle>, "
            "or <search_summary></search_summary> if you need evidence.",
        },
    ]
    video_id = entry.get("metadata", {}).get("video_id")
    used_search_summary = False

    for _ in range(args.max_turns):
        assistant = await _chat(
            client=client,
            model_name=model_name,
            messages=conversation,
            temperature=args.temperature,
            max_tokens=args.max_output_tokens,
        )
        conversation.append({"role": "assistant", "content": assistant})

        answer_hit = re.findall(
            r"<\s*answer\s*>([\s\S]*?)<\s*/\s*answer\s*>",
            assistant,
            flags=re.IGNORECASE,
        )
        if answer_hit:
            compact_conversation = _compact_conversation_for_log(conversation)
            return answer_hit[-1].strip(), conversation

        tc = _extract_tool_action(assistant)
        if not tc:
            # If the model does not follow tags, treat the raw output as final.
            
            compact_conversation = _compact_conversation_for_log(conversation)
            return assistant, conversation

        action_name = tc["action"]
        query = tc["query"]

        if action_name == "search_summary":
            if used_search_summary:
                conversation.append(
                    {
                        "role": "user",
                        "content": "Invalid action: <search_summary> can only be used once. "
                        "Use <search> or <search_subtitle> instead.",
                    }
                )
                continue
            used_search_summary = True
            all_frames = video_frames.all_frames
            if not all_frames:
                retrieved_frames = []
            else:
                k = max(1, min(int(args.max_images_per_turn), len(all_frames)))
                indices = [round(i * (len(all_frames) - 1) / max(k - 1, 1)) for i in range(k)]
                retrieved_frames = [all_frames[i] for i in indices]
            window_indices: list[int] = []
            tool_note = (
                "External tool result from retriever server.\n"
                "tool=search_summary (uniform whole-video sampling)\n"
                "window_indices=[]"
            )
        elif action_name in {"search", "search_subtitle"}:
            mode = "subtitle" if action_name == "search_subtitle" else "base"
            if not query:
                conversation.append(
                    {
                        "role": "user",
                        "content": "Invalid action: search query is empty. "
                        "Use <search>query</search> or <search_subtitle>query</search_subtitle>.",
                    }
                )
                continue
            window_indices = _call_retriever(
                retriever_server_url=args.retriever_server_url,
                dataset_name=dataset_name,
                video_id=video_id,
                query=query,
                mode=mode,
                topk=args.topk,
            )
            retrieved_frames = []
            for idx in sorted(window_indices[: args.topk]):
                retrieved_frames.extend(video_frames.get_frame_chunk(int(idx)))

            if len(retrieved_frames) > args.max_images_per_turn:
                step = len(retrieved_frames) / args.max_images_per_turn
                keep = [retrieved_frames[int(i * step)] for i in range(args.max_images_per_turn)]
                retrieved_frames = keep
            tool_note = (
                "External tool result from retriever server.\n"
                f"tool={action_name} query={query}\n"
                f"window_indices={window_indices}"
            )
        else:
            conversation.append(
                {
                    "role": "user",
                    "content": "Invalid action. Use exactly one of "
                    "<search>, <search_subtitle>, <search_summary>, or <answer>.",
                }
            )
            continue

        tool_payload: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": tool_note,
            }
        ]
        for frame in retrieved_frames:
            enc = _encode_frame(
                frame,
                max_width=args.max_image_width,
                max_height=args.max_image_height,
                quality=args.image_quality,
            )
            tool_payload.append(
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{enc}"}}
            )
        conversation.append({"role": "user", "content": tool_payload})
    conversation = _compact_conversation_for_log(conversation)
    return "", conversation

def _compact_conversation_for_log(conversation: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for j,msg in enumerate(conversation):
        compact_msg = _compact_message_for_log(msg)
        conversation[j] = compact_msg
    return conversation

def _compact_message_for_log(message: dict[str, Any]) -> dict[str, Any]:
    role = message.get("role")
    content = message.get("content")
    if isinstance(content, str):
        return {"role": role, "content": content}
    if isinstance(content, list):
        text_parts: list[str] = []
        image_count = 0
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                txt = item.get("text")
                if isinstance(txt, str):
                    text_parts.append(txt)
            elif item.get("type") == "image_url":
                image_count += 1
        return {
            "role": role,
            "content_text": "\n".join(text_parts).strip(),
            "image_count": image_count,
        }
    return {"role": role, "content": ""}


def _extract_thinks_from_conversation(conversation: list[dict[str, Any]]) -> list[str]:
    thoughts: list[str] = []
    for msg in conversation:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if not isinstance(content, str):
            continue
        think_hits = re.findall(
            r"<\s*think\s*>([\s\S]*?)<\s*/\s*think\s*>",
            content,
            flags=re.IGNORECASE,
        )
        for t in think_hits:
            cleaned = t.strip()
            if cleaned:
                thoughts.append(cleaned)
    return thoughts


def _build_pass_trace(
    pass_idx: int,
    conversation: list[dict[str, Any]],
    trace_mode: str,
) -> dict[str, Any]:
    if trace_mode == "full":
        # Keep full raw message payloads, including tool-call tags and multimodal content.
        convo_to_store = conversation
    else:
        convo_to_store = [_compact_message_for_log(m) for m in conversation]
    return {
        "pass_index": pass_idx,
        "conversation": convo_to_store,
        "thinking": _extract_thinks_from_conversation(conversation),
    }


def write_results(results: list[dict[str, Any]], file_path: str) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def _pass_at_k(n: int, c: int, k: int) -> float:
    if n < k:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - (math.comb(n - c, k) / math.comb(n, k))


async def process_entry(
    entry: dict[str, Any],
    client: AsyncOpenAI,
    model_name: str,
    args: argparse.Namespace,
    data_root: Path,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any] | None:
    async with semaphore:
        video_id = entry["metadata"]["video_id"]
        pkl_path = data_root.joinpath(f"{video_id}")
        if not pkl_path.exists():
            print(f"[skip] Missing preprocessed frames: {pkl_path}")
            return None
        video_frames = await asyncio.to_thread(VideoFrames.load, str(pkl_path))

        all_preds: list[Any] = []
        all_parsed: list[str] = []
        pass_traces: list[dict[str, Any]] = []
        for pass_idx in range(args.passes):
            try:
                if args.agent_type == "uniform":
                    final_answer, conversation = await _run_uniform_once(
                        client=client,
                        model_name=model_name,
                        entry=entry,
                        video_frames=video_frames,
                        dataset_name=args.dataset,
                        args=args,
                    )
                else:
                    final_answer, conversation = await _run_agentic_once(
                        client=client,
                        model_name=model_name,
                        entry=entry,
                        video_frames=video_frames,
                        dataset_name=args.dataset,
                        args=args,
                    )
            except Exception as e:
                print(f"Failed one pass for video {video_id}: {e}")
                final_answer = ""
                conversation = []

            # Log full conversation as prediction trace for consistency with other methods.
            all_preds.append(conversation)
            all_parsed.append(parse_answer(final_answer))

            if args.save_traces:
                pass_traces.append(
                    _build_pass_trace(
                        pass_idx=pass_idx,
                        conversation=conversation,
                        trace_mode=args.trace_mode,
                    )
                )

        gt = str(entry["correct_choice"])
        counts = Counter([p for p in all_parsed if p])
        majority_vote = counts.most_common(1)[0][0] if counts else ""
        return {
            "video_id": video_id,
            "question_id": entry["metadata"]["id"],
            "gt": gt,
            "pred": all_preds,
            "parsed_pred": all_parsed,
            "all_preds": all_preds,
            "all_parsed_preds": all_parsed,
            "majority_vote": majority_vote,
            "is_majority_correct": majority_vote == gt,
            "correct_count": sum(1 for p in all_parsed if p == gt),
            "total_passes": args.passes,
            "pass_traces": pass_traces if args.save_traces else [],
        }


async def run_async(args: argparse.Namespace) -> None:
    model_name, api_key, base_url = _resolve_model_config(args)
    client = AsyncOpenAI(api_key=api_key, base_url=base_url) if base_url else AsyncOpenAI(api_key=api_key)
    dataset = _build_dataset(args.dataset, args)
    entries = await asyncio.to_thread(dataset.load_data)

    data_root = Path(args.index_root).joinpath(f"{args.dataset}_window_{args.window_size}")
    os.makedirs(args.output_dir, exist_ok=True)
    results_file = os.path.join(args.output_dir, args.output_file)
    results: list[dict[str, Any]] = []
    if os.path.exists(results_file):
        with open(results_file, "r", encoding="utf-8") as f:
            results = json.load(f)
    done_qids = {r["question_id"] for r in results}
    remaining = [e for e in entries if e["metadata"]["id"] not in done_qids]

    print(
        f"Running {args.agent_type} eval | model_type={args.model_type} model={model_name} "
        f"| dataset={args.dataset} | remaining={len(remaining)}"
    )
    semaphore = asyncio.Semaphore(args.batch_size)
    tasks = [
        process_entry(
            entry=e,
            client=client,
            model_name=model_name,
            args=args,
            data_root=data_root,
            semaphore=semaphore,
        )
        for e in remaining
    ]

    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Evaluating"):
        out = await coro
        if out is None:
            continue
        results.append(out)
        write_results(results, results_file)

    total = len(results)
    if total == 0:
        print("No results found.")
        return

    majority_correct = sum(1 for r in results if r["is_majority_correct"])
    print(f"Majority vote accuracy: {majority_correct / total:.3f} ({majority_correct}/{total})")

    ks = [1, 2, 4, 5, 8, 10, 16]
    ks = sorted({k for k in ks if k <= args.passes} | ({args.passes} if args.passes > 1 else set()))
    for k in ks:
        avg = sum(_pass_at_k(args.passes, r.get("correct_count", 0), k) for r in results) / total
        print(f"Pass@{k}: {avg:.3f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified API eval runner for uniform and agentic VSeek with GPT-5.2 or Gemini."
    )
    parser.add_argument("--agent_type", choices=["uniform", "agentic"], required=True)
    parser.add_argument("--model_type", choices=["gpt52", "gemini"], required=True)
    parser.add_argument(
        "--dataset",
        choices=["lvb", "lvbench", "videomme", "mlvu", "cgbench"],
        required=True,
    )
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--burned_path", type=str, default=None)
    parser.add_argument("--index_root", type=str, default="/home/hg22723/vseek/dataset")
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="output/api_eval")
    parser.add_argument("--output_file", type=str, default="agent_results.json")
    parser.add_argument("--retriever_server_url", type=str, default="http://127.0.0.1:9000")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--passes", type=int, default=4)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--max_turns", type=int, default=4)
    parser.add_argument("--max_images_per_turn", type=int, default=16)
    parser.add_argument("--max_image_width", type=int, default=256)
    parser.add_argument("--max_image_height", type=int, default=256)
    parser.add_argument("--image_quality", type=int, default=85)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max_output_tokens", type=int, default=512)
    parser.add_argument(
        "--save_traces",
        action="store_true",
        help="Save compact per-pass conversation and <think> traces in output JSON.",
    )
    parser.add_argument(
        "--trace_mode",
        choices=["full", "compact"],
        default="full",
        help="Trace payload mode when --save_traces is enabled. "
        "full stores raw conversation (including image payloads).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(run_async(args))


if __name__ == "__main__":
    main()
