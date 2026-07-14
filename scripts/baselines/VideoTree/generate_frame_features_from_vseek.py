import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, CLIPImageProcessor
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data.lvb import LongVideoBench  # noqa: E402
from data.lvbench import LVBench  # noqa: E402
from data.videomme import VideoMME  # noqa: E402
from data.mlvu import MLVU  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser("Generate VideoTree frame features from VSeek datasets.")
    parser.add_argument("--dataset_name", required=True, choices=["lvb", "lvbench", "videomme", "mlvu"])
    parser.add_argument("--dataset_path", required=True, type=str)
    parser.add_argument("--burned_path", default="", type=str)
    parser.add_argument("--prepared_dir", required=True, type=str, help="Directory containing anno.json")
    parser.add_argument("--output_dir", required=True, type=str, help="Where <uid>.pt features will be stored")

    parser.add_argument("--model_name_or_path", default="BAAI/EVA-CLIP-8B", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--sample_fps", default=1.0, type=float, help="Frame sampling rate per second")
    parser.add_argument("--batch_size", default=16, type=int)
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


def build_uid_to_video_path(entries):
    uid_to_video = {}
    for entry in entries:
        uid = str(entry.get("metadata", {}).get("id", ""))
        if not uid:
            continue
        video_path = entry.get("paths", {}).get("video_path") or entry.get("paths", {}).get("raw_video_path")
        if video_path:
            uid_to_video[uid] = video_path
    return uid_to_video


def video_key(video_path: str) -> str:
    # Stable key for cache files from absolute video path
    return hashlib.md5(video_path.encode("utf-8")).hexdigest()


def sample_video_frames(video_path, sample_fps):
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
        frame_idx += 1
    cap.release()
    return frames


def encode_frames(frames, processor, model, device, batch_size):
    feature_chunks = []
    for i in range(0, len(frames), batch_size):
        batch = frames[i : i + batch_size]
        inputs = processor(images=batch, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(device)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.startswith("cuda"))):
            if hasattr(model, "encode_image"):
                feats = model.encode_image(pixel_values)
            elif hasattr(model, "get_image_features"):
                feats = model.get_image_features(pixel_values=pixel_values)
            else:
                raise AttributeError(
                    "Loaded model supports neither encode_image nor get_image_features."
                )
        feature_chunks.append(feats.detach().cpu())
    if not feature_chunks:
        return torch.empty((0, 0))
    return torch.cat(feature_chunks, dim=0)


def main():
    args = parse_args()
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available. Set --device cpu or enable CUDA.")

    prepared_dir = Path(args.prepared_dir)
    anno_path = prepared_dir / "anno.json"
    if not anno_path.exists():
        raise FileNotFoundError(f"Missing anno file: {anno_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model_name_or_path}")
    processor = CLIPImageProcessor.from_pretrained(args.model_name_or_path)
    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
        trust_remote_code=True,
    ).to(device).eval()

    with open(anno_path, "r") as f:
        anno = json.load(f)
    target_uids = list(anno.keys())
    if args.max_examples > 0:
        target_uids = target_uids[: args.max_examples]

    cfg = build_cfg(args)
    manager = get_manager(args.dataset_name, cfg)
    entries = manager.load_data()
    uid_to_video = build_uid_to_video_path(entries)

    total = len(target_uids)
    done = 0
    done_video_cache = 0
    missing_video = 0
    skipped_existing = 0
    failed = 0

    cache_dir = output_dir / "_video_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Build dedup groups: one extraction per unique video path.
    video_to_uids = {}
    for uid in target_uids:
        video_path = uid_to_video.get(uid, "")
        if not video_path:
            missing_video += 1
            continue
        video_to_uids.setdefault(video_path, []).append(uid)

    mapped_uid_total = sum(len(uids) for uids in video_to_uids.values())
    unique_video_total = len(video_to_uids)
    print(f"Target UIDs: {len(target_uids)}")
    print(f"Unique videos to process: {unique_video_total}")

    for video_path, uids in tqdm(video_to_uids.items(), desc=f"Extracting features ({args.dataset_name})"):
        valid_video = bool(video_path) and os.path.exists(video_path)
        if not valid_video:
            missing_video += len(uids)
            continue

        cache_path = cache_dir / f"{video_key(video_path)}.pt"

        if cache_path.exists() and not args.overwrite:
            done_video_cache += 1
        else:
            try:
                frames = sample_video_frames(video_path, sample_fps=args.sample_fps)
                if not frames:
                    failed += len(uids)
                    continue
                feats = encode_frames(
                    frames=frames,
                    processor=processor,
                    model=model,
                    device=device,
                    batch_size=args.batch_size,
                )
                if feats.numel() == 0:
                    failed += len(uids)
                    continue
                torch.save(feats, cache_path)
                done_video_cache += 1
            except Exception as e:
                print(f"[WARN] failed for video={video_path}: {e}")
                failed += len(uids)
                continue

        # Materialize uid-level outputs expected by downstream scripts.
        for uid in uids:
            out_path = output_dir / f"{uid}.pt"
            if out_path.exists() and not args.overwrite:
                skipped_existing += 1
                continue
            try:
                if out_path.exists() and args.overwrite:
                    out_path.unlink()
                shutil.copy2(cache_path, out_path)
                done += 1
            except Exception as e:
                print(f"[WARN] failed to materialize uid={uid}: {e}")
                failed += 1

    print("")
    print(f"Dataset: {args.dataset_name}")
    print(f"Target UIDs requested: {total}")
    print(f"Saved uid features: {done}")
    print(f"Processed unique videos: {done_video_cache}")
    print(f"Skipped existing: {skipped_existing}")
    print(f"Missing videos: {missing_video}")
    print(f"Failed: {failed}")
    if mapped_uid_total > 0 and unique_video_total > 0:
        reuse_ratio = mapped_uid_total / unique_video_total
        savings_pct = (1.0 - (unique_video_total / mapped_uid_total)) * 100.0
        print(
            f"Dedup summary: {mapped_uid_total} mapped UIDs / {unique_video_total} unique videos "
            f"({reuse_ratio:.2f}x UID-per-video, ~{savings_pct:.1f}% fewer video encodes)"
        )
    print(f"Output dir: {output_dir}")


if __name__ == "__main__":
    main()
