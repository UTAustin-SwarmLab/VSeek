import argparse
import base64
import io
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import openai
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.lvb import LongVideoBench  # noqa: E402
from data.lvbench import LVBench  # noqa: E402
from data.mlvu import MLVU  # noqa: E402
from data.videomme import VideoMME  # noqa: E402

def parse_args():
    parser = argparse.ArgumentParser("Caption sampled frames via OpenAI-compatible vLLM API.")
    parser.add_argument("--dataset_name", required=True, choices=["lvb", "lvbench", "videomme", "mlvu"])
    parser.add_argument("--dataset_path", required=True, type=str)
    parser.add_argument("--burned_path", default="", type=str)
    parser.add_argument(
        "--prepared_dir",
        default="",
        type=str,
        help="Optional prepared dataset dir with anno.json to limit target UIDs.",
    )
    parser.add_argument("--output_json", required=True, type=str)

    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--base_url", default="http://127.0.0.1:8005/v1", type=str)
    parser.add_argument(
        "--base_urls",
        default="",
        type=str,
        help="Optional comma-separated OpenAI-compatible endpoints. If set, overrides --base_url.",
    )
    parser.add_argument("--api_key", default="EMPTY", type=str)
    parser.add_argument("--request_timeout", default=30.0, type=float)
    parser.add_argument("--max_retries", default=2, type=int)
    parser.add_argument("--max_tokens", default=256, type=int)
    parser.add_argument(
        "--num_workers",
        default=0,
        type=int,
        help="Video-level worker count. 0 chooses automatically from number of endpoints.",
    )

    parser.add_argument("--sample_fps", default=1.0, type=float)
    parser.add_argument("--max_frames_per_video", default=64, type=int)
    parser.add_argument("--caption_prompt", default="Describe this frame succinctly.", type=str)
    parser.add_argument("--max_examples", default=-1, type=int)
    parser.add_argument("--overwrite", action="store_true")
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


def build_uid_video_maps(entries):
    uid_to_video_id = {}
    video_to_path = {}
    for entry in entries:
        meta = entry.get("metadata", {})
        uid = str(meta.get("id", ""))
        if not uid:
            continue
        video_id = str(meta.get("video_id", ""))
        if not video_id:
            video_id = uid
        video_path = entry.get("paths", {}).get("video_path") or entry.get("paths", {}).get("raw_video_path")
        if video_path:
            uid_to_video_id[uid] = video_id
            # Keep first seen path for this video id.
            if video_id not in video_to_path:
                video_to_path[video_id] = str(video_path)
    return uid_to_video_id, video_to_path


def normalize_existing_captions(existing, uid_to_video_id):
    """
    Convert mixed key styles (uid and/or video_id) into video_id-keyed captions.
    """
    if not isinstance(existing, dict):
        return {}

    out = {}
    for key, value in existing.items():
        key_str = str(key)
        video_id = uid_to_video_id.get(key_str, key_str)
        if video_id not in out:
            out[video_id] = value
    return out


def sample_video_frames(video_path, sample_fps, max_frames):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0
    step = max(int(round(fps / sample_fps)), 1)

    frames = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % step == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
            if max_frames > 0 and len(frames) >= max_frames:
                break
        frame_idx += 1
    cap.release()
    return frames


def pil_to_data_url(image):
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def caption_frame(client, model, prompt, image, timeout, max_retries, max_tokens):
    image_url = pil_to_data_url(image)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }
    ]

    for attempt in range(max_retries + 1):
        try:
            res = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                timeout=timeout,
                max_tokens=max_tokens,
            )
            text = res.choices[0].message.content
            return text.strip() if isinstance(text, str) else ""
        except Exception as e:
            if attempt >= max_retries:
                return f"Error: {e}"
            time.sleep(2)
    return "Error: unknown"


def parse_base_urls(args):
    if args.base_urls:
        urls = [u.strip() for u in args.base_urls.split(",") if u.strip()]
        if urls:
            return urls
    return [args.base_url]


def repeat_caption_texts_to_target_fps(caption_texts, sample_fps, target_fps=1.0):
    """
    Expand sampled captions to a denser FPS by repeating each caption when needed.
    """
    if sample_fps <= 0 or target_fps <= 0:
        return caption_texts
    if sample_fps >= target_fps:
        return caption_texts

    ratio = target_fps / sample_fps
    repeat_count = max(int(round(ratio)), 1)
    return [text for text in caption_texts for _ in range(repeat_count)]


def caption_video(video_id, video_path, client, args):
    if not video_path or not Path(video_path).exists():
        return video_id, [], False

    frames = sample_video_frames(
        video_path=video_path,
        sample_fps=args.sample_fps,
        max_frames=args.max_frames_per_video,
    )
    if not frames:
        return video_id, [], False

    caption_texts = []
    for i, frame in enumerate(frames):
        text = caption_frame(
            client=client,
            model=args.model,
            prompt=args.caption_prompt,
            image=frame,
            timeout=args.request_timeout,
            max_retries=args.max_retries,
            max_tokens=args.max_tokens,
        )
        caption_texts.append(text)

    caption_texts = repeat_caption_texts_to_target_fps(
        caption_texts=caption_texts,
        sample_fps=args.sample_fps,
        target_fps=1.0,
    )
    captions = [f"#C {i}: {text}" for i, text in enumerate(caption_texts)]
    return video_id, captions, True


def main():
    args = parse_args()
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing = {}
    if output_path.exists() and not args.overwrite:
        try:
            existing = json.loads(output_path.read_text())
        except Exception:
            existing = {}

    cfg = build_cfg(args)
    manager = get_manager(args.dataset_name, cfg)
    entries = manager.load_data()
    uid_to_video_id, video_to_path = build_uid_video_maps(entries)

    target_video_ids = list(video_to_path.keys())
    if args.prepared_dir:
        anno_path = Path(args.prepared_dir) / "anno.json"
        if anno_path.exists():
            anno = json.loads(anno_path.read_text())
            target_video_ids = []
            seen = set()
            for uid in anno.keys():
                video_id = uid_to_video_id.get(str(uid))
                if video_id and video_id not in seen:
                    seen.add(video_id)
                    target_video_ids.append(video_id)
    if args.max_examples > 0:
        target_video_ids = target_video_ids[: args.max_examples]

    base_urls = parse_base_urls(args)
    clients = [openai.OpenAI(api_key=args.api_key, base_url=url) for url in base_urls]
    print(f"Using {len(base_urls)} endpoint(s): {base_urls}")

    out = normalize_existing_captions(existing, uid_to_video_id)
    pending_video_ids = []
    skipped = 0
    for video_id in target_video_ids:
        if video_id in out and not args.overwrite:
            skipped += 1
        else:
            pending_video_ids.append(video_id)

    if args.num_workers > 0:
        num_workers = args.num_workers
    else:
        num_workers = max(1, len(clients) * 2)
    num_workers = max(1, min(num_workers, len(pending_video_ids) if pending_video_ids else 1))
    print(
        f"Target videos: {len(target_video_ids)} | pending: {len(pending_video_ids)} | "
        f"skipped(existing): {skipped} | workers: {num_workers}"
    )

    saved = 0
    failed = 0
    progress = tqdm(total=len(target_video_ids), desc=f"Captioning ({args.dataset_name})")
    if skipped:
        progress.update(skipped)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        for idx, video_id in enumerate(pending_video_ids):
            endpoint_idx = idx % len(clients)
            client = clients[endpoint_idx]
            video_path = video_to_path.get(video_id)
            future = executor.submit(caption_video, video_id, video_path, client, args)
            futures[future] = video_id

        completed_since_flush = 0
        for future in as_completed(futures):
            video_id = futures[future]
            try:
                video_id, captions, ok = future.result()
            except Exception as e:
                out[video_id] = [f"Error: worker exception: {e}"]
                failed += 1
                progress.update(1)
                completed_since_flush += 1
                if completed_since_flush >= 20:
                    output_path.write_text(json.dumps(out, indent=2))
                    completed_since_flush = 0
                continue

            out[video_id] = captions
            if ok:
                saved += 1
            else:
                failed += 1
            progress.update(1)
            completed_since_flush += 1
            if completed_since_flush >= 20:
                output_path.write_text(json.dumps(out, indent=2))
                completed_since_flush = 0

    progress.close()

    output_path.write_text(json.dumps(out, indent=2))
    print(f"Saved captions: {saved}")
    print(f"Skipped existing: {skipped}")
    print(f"Failed: {failed}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
