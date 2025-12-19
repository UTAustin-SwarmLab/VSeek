"""
LVBench dataset manager for loading, indexing, and processing the LVBench dataset.
Similar to lvb.py but adapted for the LVBench dataset from llmseval.
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


class LVBench(Manager):
    """
    Manager for LVBench dataset from llmseval.
    Handles loading, video indexing, and data preparation.
    """
    
    def __init__(self, cfg: DictConfig | None = None):
        """
        Initialize LVBench manager.
        
        Args:
            cfg: Hydra DictConfig with dataset and retriever configuration
        """
        self.cfg = cfg
        self._dataset_path = cfg.dataset.lvbench.dataset_path
        
    def load_data(self):
        """
        Load LVBench dataset from llmseval.lvbench.
        Downloads and processes the dataset, storing videos to the specified path.
        
        Returns:
            List of processed dataset entries
        """
        # try:
        #     # Try to import datasets library
        #     import datasets
        # except ImportError:
        #     raise ImportError(
        #         "datasets library not found. Please install it: pip install datasets"
        #     )
        
        category_buckets = defaultdict(list)
        
        # Load LVBench dataset from HuggingFace
        print(f"Loading LVBench dataset from HuggingFace...")
        
        # try:
        #     # LVBench is typically available as lmms-lab/LVBench on HuggingFace
        #     lvbench_dataset = datasets.load_dataset("lmms-lab/LVBench", split="test", cache_dir=self._dataset_path)
        # except Exception as e:
        #     print(f"Error loading from HuggingFace: {e}")
        #     print("Trying alternative loading method...")
        #     # Alternative: load from local path if already downloaded
        #     try:
        #         lvbench_dataset = datasets.load_dataset(
        #             "json",
        #             data_files=os.path.join(self._dataset_path, "lvbench_val.json")
        #         )["train"]
        #     except Exception as e2:
        #         print(f"Error loading from local path: {e2}")
        #         raise RuntimeError(
        #             "Failed to load LVBench dataset. Please ensure you have:\n"
        #             "1. Internet connection to download from HuggingFace, OR\n"
        #             "2. Local JSON file at {}/lvbench_test.json".format(self._dataset_path)
        #         )
        
        with open(os.path.join(self._dataset_path, "data/lvbench_val.json"), "r") as f:
            lvbench_dataset = json.load(f)
        
        print(f"Loaded {len(lvbench_dataset)} samples from LVBench")
        
        # Process each item in the dataset
        for idx, item in enumerate(tqdm(lvbench_dataset, desc="Processing LVBench")):
            try:
                # Extract video information
                video_id = item.get("key", item.get("id", f"video_{idx}"))
                question = item.get("question", "")
                
                # Handle different answer formats
                if "answer" in item:
                    answer = item["answer"]
                elif "gt" in item:
                    answer = item["gt"]
                elif "correct_choice" in item:
                    answer = item["correct_choice"]
                else:
                    answer = None
                
                # candiadates in the question, we need to split the string

                question_parts = question.split("\n")
                question = question_parts[0]
                options = question_parts[1:]
                candidates = item.get("candidates", item.get("options", options))

                # question, options = question.split("\n")
              
                
                # print(f"question: {question}")
                # print(f"candidates: {candidates}")


                # Handle candidates/options
                # candidates = item.get("candidates", item.get("options", []))
                
                # Determine paths for video storage
                video_filename = item.get("video", item.get("video_path", f"{video_id}.mp4"))
                video_save_path = os.path.join(
                    self._dataset_path, "videos", video_filename
                )
                
                # Download or copy video if needed
                if not os.path.exists(video_save_path):
                    os.makedirs(os.path.dirname(video_save_path), exist_ok=True)
                    # If video URL is provided, download it
                    if "video_url" in item:
                        self._download_video(item["video_url"], video_save_path)
                    # If video data is embedded (base64), decode it
                    elif "video_data" in item:
                        self._decode_video(item["video_data"], video_save_path)
                    else:
                        print(f"Warning: Video not found for {video_id}, skipping...")
                        # continue
                
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
                category = item.get("category", item.get("task_type", "general"))
                
                # Build entry in LVB-compatible format
                question_text = "\n This is a multiple choice question. You must choose the correct answer from the options with the number or letter of the option. \n"
                question_text += f"Question: {question} \n"
                
                # TODO: Add ground truth frames
                
                entry = {
                    "question": question_text,
                    "candidates": candidates,
                    "correct_choice": answer,
                    "ground_truth_frames": item.get("ground_truth_frames", []),
                    "paths": {
                        "raw_video_path": video_save_path,
                        "subtitle_path": subtitle_path,
                        "video_path": video_save_path,  # Same as raw for now
                    },
                    "metadata": {
                        "video_id": video_id,
                        "id": item.get("uid", video_id),
                        # Store original item for reference
                        "original_data": json.dumps(item),
                    },
                }
                
                category_buckets[category].append(entry)
                
            except Exception as e:
                print(f"Error processing item {idx}: {e}")
                print(traceback.format_exc())
                print(f"entry: {item}")
                continue
        
        # Flatten list of all entries from each category
        all_entries = [entry for entries in category_buckets.values() for entry in entries]
        print(f"Successfully processed {len(all_entries)} entries across {len(category_buckets)} categories")
        
        return all_entries
    
    def _download_video(self, url: str, save_path: str):
        """
        Download video from URL.
        
        Args:
            url: Video URL
            save_path: Path to save the video
        """
        try:
            import requests
        except ImportError:
            raise ImportError("requests library not found. Please install it: pip install requests")
        
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            with open(save_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded video to {save_path}")
        except Exception as e:
            print(f"Error downloading video from {url}: {e}")
            raise
    
    def _decode_video(self, video_data: str, save_path: str):
        """
        Decode base64 video data and save.
        
        Args:
            video_data: Base64 encoded video data
            save_path: Path to save the decoded video
        """
        try:
            video_bytes = base64.b64decode(video_data)
            with open(save_path, "wb") as f:
                f.write(video_bytes)
            print(f"Decoded video to {save_path}")
        except Exception as e:
            print(f"Error decoding video data: {e}")
            raise
    
    def save_it_as_vseek_data(self, desired_interval_in_sec: int = 1):
        """
        Index videos by window size and save embeddings.
        Similar to LongVideoBench's save_it_as_vseek_data.
        
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
                    dataset_name = "lvbench"
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
        dataset_name = "lvbench"
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
