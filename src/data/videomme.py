"""
Video-MME dataset manager for loading, indexing, and processing the Video-MME dataset.
Adapted from lvbench.py for Video-MME specific format.
"""

import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from tqdm import tqdm
import traceback
import torch
import base64

from vseek.data.frame import SingleFrame, VideoFrames
from data.manager import Manager
from omegaconf import DictConfig
from vseek.video.read_video import read_video
from vseek.video_embedding.video_clip import ViClip


def srt_time_to_seconds(time_str):
    """
    Convert SRT timestamp to seconds.
    
    Args:
        time_str: Time string in format "HH:MM:SS,mmm" or "HH:MM:SS.mmm"
    
    Returns:
        Float seconds
    """
    time_str = time_str.strip().replace(',', '.')
    parts = time_str.split(':')
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds


def parse_srt_to_json(srt_path):
    """
    Parse SRT subtitle file and convert to JSON format compatible with LVB processing.
    
    Args:
        srt_path: Path to SRT file
    
    Returns:
        List of dictionaries with keys: timestamp (tuple of start, end in seconds), text
    """
    subtitles = []
    
    try:
        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by double newlines to get individual subtitle blocks
        blocks = content.strip().split('\n\n')
        
        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) < 3:
                continue
            
            # First line is the sequence number (skip it)
            # Second line contains timestamps
            # Remaining lines are the text
            
            try:
                # Parse timestamp line: "00:00:01,000 --> 00:00:04,000"
                timestamp_line = lines[1]
                if '-->' not in timestamp_line:
                    continue
                
                start_str, end_str = timestamp_line.split('-->')
                start = srt_time_to_seconds(start_str)
                end = srt_time_to_seconds(end_str)
                
                # Join all remaining lines as text
                text = '\n'.join(lines[2:]).strip()
                
                # Format compatible with LVB's process_subtitles function
                # Using "timestamp" format: [start, end] in seconds
                subtitles.append({
                    'timestamp': [start, end],
                    'text': text
                })
            except (ValueError, IndexError) as e:
                print(f"Warning: Failed to parse subtitle block: {e}")
                continue
    
    except Exception as e:
        print(f"Error reading SRT file {srt_path}: {e}")
        return []
    
    return subtitles


class VideoMME(Manager):
    """
    Manager for Video-MME dataset.
    Handles loading, video indexing, and data preparation.
    """
    
    def __init__(self, cfg: DictConfig | None = None):
        """
        Initialize Video-MME manager.
        
        Args:
            cfg: Hydra DictConfig with dataset and retriever configuration
        """
        self.cfg = cfg
        self._dataset_path = cfg.dataset.videomme.dataset_path
        self._burned_path = cfg.dataset.videomme.burned_path
        
    def load_data(self):
        """
        Load Video-MME dataset.
        
        Returns:
            List of processed dataset entries
        """
        category_buckets = defaultdict(list)
        
        print(f"Loading Video-MME dataset from {self._dataset_path}...")
        
        # Try loading from JSON file
        json_file = os.path.join(self._dataset_path, "videomme", "videomme_val.json")
        if not os.path.exists(json_file):
            # Try alternative paths
            json_file = os.path.join(self._dataset_path, "videomme_val.json")
        
        if not os.path.exists(json_file):
            raise FileNotFoundError(f"Video-MME JSON file not found at {json_file}")
        
        with open(json_file, "r") as f:
            videomme_dataset = json.load(f)
        
        print(f"Loaded {len(videomme_dataset)} samples from Video-MME")
        
        # Process each item in the dataset
        for idx, item in enumerate(tqdm(videomme_dataset, desc="Processing Video-MME")):
            try:
                # Extract video information
                video_id = item.get("videoID", item.get("videoID", item.get("id", f"video_{idx}")))
                question = item.get("question", "")
                
                # Handle different answer formats
                answer = None
                if "answer" in item:
                    answer = item["answer"]
                elif "gt" in item:
                    answer = item["gt"]
                elif "correct_choice" in item:
                    answer = item["correct_choice"]
                
                # Handle candidates/options
                candidates = []
                if "options" in item:
                    candidates = item["options"]
                elif "candidates" in item:
                    candidates = item["candidates"]
                elif "choices" in item:
                    candidates = item["choices"]
                
                # Determine paths for video storage
                video_filename = f"{video_id}.mp4"
                
                # Handle different video path formats
                if not video_filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    video_filename = f"{video_filename}.mp4"
                
                video_save_path = os.path.join(self._dataset_path, "videos/data", video_filename)
                video_path = os.path.join(self._burned_path, video_filename)
                # Check if video exists
                if not os.path.exists(video_save_path) and not os.path.exists(video_path):
                    print(f"Warning: Video not found for {video_id} at {video_save_path}, skipping...")
                    continue
                
                # Handle subtitles if present (SRT format)
                subtitle_filename = video_filename.replace('.mp4', '.srt')
                subtitle_path = os.path.join(self._dataset_path, "subtitles/subtitle", subtitle_filename)
                
                subtitle_json_path = None
                if os.path.exists(subtitle_path):
                    # Convert SRT to JSON format and save
                    subtitle_data = parse_srt_to_json(subtitle_path)
                    
                    # Save as JSON for future use
                    subtitle_json_path = os.path.join(
                        self._dataset_path, "subtitles_json", f"{video_id}.json"
                    )
                    os.makedirs(os.path.dirname(subtitle_json_path), exist_ok=True)
                    with open(subtitle_json_path, "w") as f:
                        json.dump(subtitle_data, f, indent=2)
                else:
                    subtitle_json_path = None
                
                # Determine category
                category = item.get("category", item.get("task_type", item.get("domain", "general")))
                
                # Build entry in LVB-compatible format
                question_text = "\n This is a multiple choice question. You must choose the correct answer from the options with the number or letter of the option. \n"
                question_text += f"Question: {question} \n"
                
                entry = {
                    "question": question_text,
                    "candidates": candidates,
                    "correct_choice": answer,
                    "paths": {
                        "raw_video_path": video_save_path,
                        "subtitle_path": subtitle_json_path,
                        "video_path": video_path,
                    },
                    "metadata": {
                        "video_id": video_id,
                        "id": item.get("question_id", video_id),
                        "question_category": category,
                        "level": item.get("level", item.get("difficulty", "medium")),
                        "duration": item.get("duration", 0),
                        "starting_timestamp_for_subtitles": item.get(
                            "starting_timestamp_for_subtitles", 0
                        ),
                        "original_data": item,
                    },
                }
                
                category_buckets[category].append(entry)
                
            except Exception as e:
                print(f"Error processing item {idx}: {e}")
                print(traceback.format_exc())
                continue
        
        # Flatten list of all entries from each category
        all_entries = [entry for entries in category_buckets.values() for entry in entries]
        print(f"Successfully processed {len(all_entries)} entries across {len(category_buckets)} categories")
        
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
                    if self.is_file_exists(window_size, unique_id):
                        print(f"Skipping already processed video: {unique_id}")
                        return
                    
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
                            entry["metadata"]["starting_timestamp_for_subtitles"],
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
                    dataset_name = "videomme"
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
    
    def is_file_exists(self, window_size: str, unique_id: str):
        """
        Check if indexed video file already exists.
        
        Args:
            window_size: Window size used for indexing
            unique_id: Video unique identifier
            
        Returns:
            True if file exists and is valid, False otherwise
        """
        dataset_name = "videomme"
        dir_name = f"{dataset_name}_window_{window_size}"
        output_dir = self.cfg.retriever.index_path
        data_path = Path(output_dir).joinpath(dir_name)
        
        if not data_path.exists():
            return False
        
        files = [dir_path.stem for dir_path in data_path.iterdir()]
        if unique_id in files:
            metadata_path = data_path.joinpath(unique_id, "metadata.json")
            if metadata_path.exists():
                return True
            else:
                # Invalid data, clean up
                invalid_data_path = data_path.joinpath(unique_id)
                if invalid_data_path.exists() and invalid_data_path.is_dir():
                    print(f"Cleaning up invalid data: {invalid_data_path}")
                    shutil.rmtree(invalid_data_path)
        
        return False
    
    def postprocess_data(self):
        """Postprocess data if needed."""
        pass

