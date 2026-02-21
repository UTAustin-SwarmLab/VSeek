"""
CGBench dataset manager for loading and normalizing MCQ entries.

This module keeps `MVBench` as a backward-compatible alias so existing imports
continue to work.
"""

import json
import os
import re
from collections import defaultdict
from pathlib import Path

from data.manager import Manager
from omegaconf import DictConfig
from data.videomme import parse_srt_to_json
import traceback
from vseek.video_embedding.video_clip import ViClip
from vseek.data.frame import VideoFrames
from vseek.video.read_video import read_video
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from threading import Lock



def _normalize_answer_to_index(answer, candidates: list[str]) -> str | None:
    if answer is None:
        return None

    answer_str = str(answer).strip()
    if not answer_str:
        return None

    letter_match = re.fullmatch(r"[\(\[]?([A-Za-z])[\)\]]?", answer_str)
    if letter_match:
        idx = ord(letter_match.group(1).upper()) - ord("A")
        if 0 <= idx < len(candidates):
            return str(idx)

    if answer_str.isdigit():
        numeric_idx = int(answer_str)
        if 0 <= numeric_idx < len(candidates):
            return str(numeric_idx)
        if 1 <= numeric_idx <= len(candidates):
            return str(numeric_idx - 1)

    lowered = answer_str.lower()
    for idx, candidate in enumerate(candidates):
        if lowered == str(candidate).strip().lower():
            return str(idx)

    return None


class CGBench(Manager):
    def __init__(self, cfg: DictConfig | None = None):
        self.cfg = cfg
        dataset_cfg = None
        if cfg is not None:
            # Prefer explicit CGBench config, then fall back for backward compatibility.
            dataset_cfg = (
                cfg.get("dataset", {}).get("cgbench")
            )
        dataset_path = dataset_cfg.get("dataset_path") if dataset_cfg else None
        self._burned_path = dataset_cfg.get("burned_path") if dataset_cfg else None
        if not dataset_path:
            raise ValueError(
                "Missing dataset path. Expected `dataset.cgbench.dataset_path` "
            )
        self._dataset_path = str(dataset_path)
        self._video_index = None

    def _load_annotations(self) -> list[dict]:
        explicit_candidates = [
            os.path.join(self._dataset_path, "puls.json"),
            os.path.join(self._dataset_path, "cgbench_mini.json"),

        ]
        for candidate in explicit_candidates:
            if os.path.exists(candidate):
                with open(candidate, "r", encoding="utf-8") as f:
                    data = json.load(f)
                print(f"Loading CGBench dataset from {candidate}")
                if isinstance(data, list):
                    return data
                if isinstance(data, dict):
                    if "data" in data and isinstance(data["data"], list):
                        return data["data"]
                    if "questions" in data and isinstance(data["questions"], list):
                        return data["questions"]
                    return [data]
                raise ValueError(f"Unsupported annotation format in {candidate}")


    def _build_video_index(self):
        if self._video_index is not None:
            return

        self._video_index = {}
        roots = [
            os.path.join(self._dataset_path, "video"),
            os.path.join(self._dataset_path, "videos"),
            self._dataset_path,
        ]

        for root in roots:
            if not os.path.exists(root):
                continue
            for path in Path(root).rglob("*"):
                if not path.is_file():
                    continue
                ext = path.suffix.lower()
                if ext not in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
                    continue
                rel = str(path.relative_to(self._dataset_path))
                self._video_index[rel] = str(path)
                self._video_index[path.stem] = str(path)
                self._video_index[path.name] = str(path)

        print(f"Indexed {len(self._video_index)} CGBench video keys.")

        return None

    def load_data(self):
        raw_data = self._load_annotations()
        if isinstance(raw_data, list) and raw_data and "prompt" in raw_data[0]:
            # Already normalized puls format
            return raw_data

        category_buckets = defaultdict(list)
        subtitle_dir = os.path.join(self._dataset_path, "subtitles_json")
        os.makedirs(subtitle_dir, exist_ok=True)

        for idx, item in enumerate(raw_data):
            try:
                raw_video_path = os.path.join(self._dataset_path, "videos", f"{item.get('video_uid')}.mp4")
                video_path = os.path.join(self._burned_path, f"{item.get('video_uid')}.mp4")
                
                if not os.path.exists(raw_video_path) or not os.path.exists(video_path):
                    print(f"Warning: video not found for item {idx} (video={item.get('video_uid')})")
                    continue

                video_id = str(item.get("video_uid"))
                question = str(item.get("question", "")).strip()
                candidates = item.get("choices", item.get("options", item.get("choices", [])))
                if not isinstance(candidates, list):
                    candidates = []
                candidates = [str(c).strip() for c in candidates]

                answer = item.get("right_answer", item.get("gt", item.get("correct_choice")))
                normalized_answer = ord(answer) - ord("A")

                subtitle_path = os.path.join(self._dataset_path, "cg_subtitles", f"{video_id}.srt")

                category = str(item.get("sub_category", item.get("data_type", "general")))
                question_text = (
                    "\n This is a multiple choice question. "
                    "You must choose the correct answer as a number or letter of the option. \n"
                )
                question_text += f"Question: {question} \n"

                entry = {
                    "question": question_text,
                    "candidates": candidates,
                    "correct_choice": normalized_answer,
                    "paths": {
                        "raw_video_path": raw_video_path,
                        "subtitle_path": subtitle_path,
                        "video_path": video_path,
                    },
                    "metadata": {
                        "video_id": video_id,
                        "id": str(item.get("qid")),
                        "original_data": json.dumps(item),
                    },
                }
                category_buckets[category].append(entry)
            except Exception as exc:
                print(f"Error processing CGBench item {idx}: {exc}")
                continue

        all_entries = [entry for entries in category_buckets.values() for entry in entries]
        print(
            f"Successfully processed {len(all_entries)} entries across "
            f"{len(category_buckets)} categories."
        )
        return all_entries

    def save_it_as_vseek_data(self, desired_interval_in_sec: int = 1):
        """
        Index videos by window size and save embeddings.
        
        Args:
            desired_interval_in_sec: Interval in seconds for frame extraction
        """
        try:
            data = self.load_data()
            window_size = self.cfg.retriever.window_size
            
            # Allow overriding workers via env
            default_workers = min(4, (os.cpu_count() or 4))
            max_workers = int(os.getenv("VSEEK_WORKERS", default_workers))
            
            # Deduplicate entries by video_id
            unique_entries: list[dict] = []
            seen_ids: set[str] = set()
            
            for e in data:
                vid = e.get("metadata", {}).get("video_id")
                if not vid or vid in seen_ids:
                    continue
                seen_ids.add(vid)
                unique_entries.append(e)
            
            print(f"Processing {len(unique_entries)} unique videos...")
            
            # In-process guard against accidental duplicate scheduling
            processed_ids: set[str] = set()
            processed_lock = Lock()
            
            def _process_single_entry(entry: dict):
                try:
                    unique_id = entry["metadata"]["video_id"]
                    
                    # Fast skip if already processed on disk
                    # if self.is_file_exists(window_size, unique_id):
                    #     print(f"Skipping already processed video: {unique_id}")
                    #     return
                    
                    # Prevent duplicate work in this process
                    with processed_lock:
                        if unique_id in processed_ids:
                            return
                        processed_ids.add(unique_id)
                    
                    print(f"Processing video: {unique_id}")
                    
                    # Load subtitles if available
                    subtitles = []
                    if entry["paths"]["subtitle_path"] and os.path.exists(entry["paths"]["subtitle_path"]):
                        with open(entry["paths"]["subtitle_path"], "r") as f:
                            subtitles = json.load(f)
                    
                    # Read video
                    video = read_video(video_path=entry["paths"]["video_path"])
                    
                    # Initialize retriever model from Hydra cfg
                    vclip = ViClip(
                        pretrained_model_path=self.cfg.retriever.retrieval_model_path,
                        gpu_number=self.cfg.retriever.gpu_number,
                    )
                    
                    video_frames = VideoFrames(window_size=window_size)
                    
                    # Get all frames at desired interval
                    all_frames = video.get_all_frames_of_video(
                        desired_interval_in_sec=desired_interval_in_sec
                    )
                    video_frames.add_all_frames(all_frames)
                    video_frames.partition_frames()
                    
                    print(f"Added {len(all_frames)} frames for video {unique_id}")
                    
                    # Process subtitles if available
                    if subtitles:
                        from data.lvb import process_subtitles
                        
                        original_fps = video.video_info.original_fps
                        original_frame_count = video.video_info.original_frame_count
                        original_duration = video.video_info.original_duration
                        desired_frame_count = video.video_info.processed_frame_count
                        frame_step = video.get_frame_step(
                            desired_interval_in_sec=desired_interval_in_sec
                        )
                        
                        all_subtitles = process_subtitles(
                            subtitles,
                            0,
                            original_fps,
                            original_frame_count,
                            original_duration,
                            desired_frame_count,
                            frame_step,
                        )
                        
                        video_frames.add_all_subtitles(all_subtitles)
                        video_frames.partition_subtitles()
                        print(f"Added {len(all_subtitles)} subtitle entries")
                        
                        # Add subtitle embeddings
                        all_unique_subtitles = list(video_frames.window_by_subtitle.keys())
                        for subtitle in all_unique_subtitles:
                            video_frames.add_subtitle_embedding(
                                subtitle, vclip.get_text_embedding(subtitle)
                            )
                    
                    # Add frame embeddings for each window
                    for window_idx in video_frames.frames_by_window.keys():
                        frame_chunk = video_frames.get_frame_chunk(window_idx)
                        video_frames.add_embedding(
                            window_idx, vclip.get_feature(frame_chunk)
                        )
                    
                    # Save indexed data
                    dataset_name = "cgbench"
                    dir_name = f"{dataset_name}_window_{window_size}"
                    file_name = f"{unique_id}.pkl"
                    output_dir = self.cfg.retriever.index_path
                    output_path = Path(output_dir).joinpath(dir_name, file_name)
                    video_frames.save(str(output_path))
                    
                    print(f"Saved indexed video: {unique_id}")
                    
                except Exception as e:
                    print(f"Error processing entry {entry.get('metadata', {}).get('video_id', 'unknown')}: {e}")
                    print(traceback.format_exc())
            
            # Process all entries in parallel
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                list(tqdm(
                    executor.map(_process_single_entry, unique_entries),
                    total=len(unique_entries),
                    desc="Indexing videos"
                ))
                
        except Exception as e:
            print(f"Error saving data: {e}")
            print(traceback.format_exc())

    def postprocess_data(self):
        pass


# Backward compatibility for existing imports.
class MVBench(CGBench):
    pass
