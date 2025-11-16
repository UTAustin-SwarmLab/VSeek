"""
MLVU dataset manager for loading, indexing, and processing the MLVU dataset.
Adapted from lvbench.py for MLVU specific format.
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


class MLVU(Manager):
    """
    Manager for MLVU dataset.
    Handles loading, video indexing, and data preparation.
    """
    
    def __init__(self, cfg: DictConfig | None = None):
        """
        Initialize MLVU manager.
        
        Args:
            cfg: Hydra DictConfig with dataset and retriever configuration
        """
        self.cfg = cfg
        self._dataset_path = cfg.dataset.dataset_path
        self._categories = [
            "1_plotQA",
            "2_needle",
            "3_ego",
            "4_count",
            "5_order",
            "6_anomaly_reco",
            "7_topic_reasoning",
        ]

    def _load_and_merge_data_from_json(self, json_file):
        with open(json_file, "r") as f:
            data = json.load(f)
        return data
    
    def merge_category_files(self, output_filename: str = "mlvu_val.json"):
        """
        Merge MLVU category JSON files into a single JSON file.
        
        Args:
            output_filename: Name of the output merged JSON file
            
        Returns:
            Path to the merged JSON file
        """
        merged_data = []
        category_stats = {}
        
        print(f"Searching for MLVU category JSON files in {self._dataset_path}")
        
        # Try different possible directory structures
        possible_data_dirs = [
            os.path.join(self._dataset_path, "data"),
            os.path.join(self._dataset_path, "annotations"),
            os.path.join(self._dataset_path, "json"),
            self._dataset_path,  # Check root directory as well
        ]
        
        data_dir = None
        for possible_dir in possible_data_dirs:
            if os.path.exists(possible_dir):
                # Check if any category JSON exists in this directory
                test_files = [
                    os.path.join(possible_dir, f"{cat}.json") for cat in self._categories
                ]
                if any(os.path.exists(f) for f in test_files):
                    data_dir = possible_dir
                    break
        
        if not data_dir:
            raise FileNotFoundError(
                f"Could not find MLVU category JSON files in any of: {possible_data_dirs}"
            )
        
        print(f"Found MLVU data directory: {data_dir}")
        
        # Load and merge each category
        for category in tqdm(self._categories, desc="Merging categories"):
            category_file = os.path.join(data_dir, f"{category}.json")
            
            if not os.path.exists(category_file):
                print(f"Warning: {category}.json not found, skipping...")
                continue
            
            try:
                with open(category_file, "r", encoding="utf-8") as f:
                    category_data = json.load(f)
                
                # Handle different JSON formats
                if isinstance(category_data, list):
                    items = category_data
                elif isinstance(category_data, dict):
                    # Try common keys for the data list
                    items = category_data.get("data", category_data.get("questions", category_data.get("items", [])))
                    if not isinstance(items, list):
                        items = [category_data]  # Single item wrapped in dict
                else:
                    print(f"Warning: Unexpected format in {category_file}, skipping...")
                    continue
                
                # Add category information to each item if not present
                for item in items:
                    if "category" not in item:
                        item["category"] = category
                    if "task_type" not in item:
                        item["task_type"] = category
                
                merged_data.extend(items)
                category_stats[category] = len(items)
                print(f"  Loaded {len(items)} items from {category}")
                
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from {category_file}: {e}")
                continue
            except Exception as e:
                print(f"Error loading {category_file}: {e}")
                print(traceback.format_exc())
                continue
        
        # Save merged data
        output_path = os.path.join(self._dataset_path, output_filename)
        
        print(f"\nSaving merged data to {output_path}")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(merged_data, f, indent=2, ensure_ascii=False)
        
        # Print statistics
        print("\n" + "="*60)
        print("Merge Statistics:")
        print("="*60)
        for category, count in category_stats.items():
            print(f"  {category:25s}: {count:6d} items")
        print("-"*60)
        print(f"  {'Total':25s}: {len(merged_data):6d} items")
        print("="*60)
        print(f"\nSuccessfully merged {len(merged_data)} items into {output_path}")
        
        return output_path
    
    def load_data(self):
        """
        Load MLVU dataset.
        
        Returns:
            List of processed dataset entries
        """
        category_buckets = defaultdict(list)
        
        print(f"Loading MLVU dataset from {self._dataset_path}...")
        
        # Try loading from JSON file
        json_file = os.path.join(self._dataset_path, "data", "mlvu_val.json")
        if not os.path.exists(json_file):
            # Try alternative paths
            json_file = os.path.join(self._dataset_path, "mlvu_val.json")
        
        if not os.path.exists(json_file):
            # If merged file doesn't exist, try to create it from category files
            print(f"MLVU merged JSON file not found. Attempting to merge category files...")
            try:
                json_file = self.merge_category_files()
            except Exception as e:
                raise FileNotFoundError(
                    f"MLVU JSON file not found at {json_file} and failed to merge category files: {e}"
                )
        
        with open(json_file, "r") as f:
            mlvu_dataset = json.load(f)
        
        print(f"Loaded {len(mlvu_dataset)} samples from MLVU")
        
        # Process each item in the dataset
        for idx, item in enumerate(tqdm(mlvu_dataset, desc="Processing MLVU")):
            try:
                # Extract video information
                video_id = item.get("video_id", item.get("id", f"video_{idx}"))
                question = item.get("question", "")
                
                # Handle different answer formats
                answer = None
                if "answer" in item:
                    answer = item["answer"]
                elif "gt" in item:
                    answer = item["gt"]
                elif "correct_choice" in item:
                    answer = item["correct_choice"]
                
                # Handle candidates/options - MLVU might have them embedded in question
                candidates = []
                if "options" in item:
                    candidates = item["options"]
                elif "candidates" in item:
                    candidates = item["candidates"]
                elif "choices" in item:
                    candidates = item["choices"]
                else:
                    # Try to extract from question if formatted like "A) option1 B) option2"
                    # This is a common format in MLVU
                    if any(marker in question for marker in ['A)', 'B)', 'C)', 'D)']):
                        # Split by option markers
                        import re
                        parts = re.split(r'[A-Z]\)', question)
                        if len(parts) > 1:
                            # First part is the actual question
                            question = parts[0].strip()
                            # Rest are candidates
                            candidates = [part.strip() for part in parts[1:] if part.strip()]
                
                # Determine paths for video storage
                video_filename = item.get("video", item.get("video_path", item.get("video_name", f"{video_id}.mp4")))
                
                # Handle different video path formats
                if not video_filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    video_filename = f"{video_filename}.mp4"
                
                video_save_path = os.path.join(self._dataset_path, "videos", video_filename)
                
                # Check if video exists
                if not os.path.exists(video_save_path):
                    print(f"Warning: Video not found for {video_id} at {video_save_path}, skipping...")
                    continue
                
                # Handle subtitles if present
                subtitle_path = None
                if "subtitles" in item or "subtitle" in item:
                    subtitle_data = item.get("subtitles", item.get("subtitle", []))
                    subtitle_save_path = os.path.join(
                        self._dataset_path, "subtitles", f"{video_id}.json"
                    )
                    os.makedirs(os.path.dirname(subtitle_save_path), exist_ok=True)
                    with open(subtitle_save_path, "w") as f:
                        json.dump(subtitle_data, f)
                    subtitle_path = subtitle_save_path
                
                # Determine category
                category = item.get("category", item.get("task_type", item.get("type", "general")))
                
                # Build entry in LVB-compatible format
                question_text = "\n This is a multiple choice question. You must choose the correct answer as a number. \n"
                question_text += f"Question: {question} \n"
                
                entry = {
                    "question": question_text,
                    "candidates": candidates,
                    "correct_choice": answer,
                    "paths": {
                        "raw_video_path": video_save_path,
                        "subtitle_path": subtitle_path,
                        "video_path": video_save_path,
                    },
                    "metadata": {
                        "video_id": video_id,
                        "id": item.get("id", video_id),
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
                    dataset_name = "mlvu"
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
        dataset_name = "mlvu"
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

