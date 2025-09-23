import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from tqdm import tqdm

from vseek.data.frame import SingleFrame, VideoFrames
from data.manager import Manager
from vseek.setting import DataSetting, ViClipSetting
from vseek.video.read_video import read_video
from vseek.video_embedding.video_clip import ViClip
import traceback
VICLIP_SETTING = ViClipSetting()

DATA_SETTING = DataSetting()

import re

def convert_time_to_frame(time: str):
    time = re.sub(r"\s+", "", time)
    hrs, mins, secs = time.split(":")
    hrs = int(hrs)
    mins = int(mins)
    secs, ms = secs.split(".")
    
    secs = int(secs)
    ms = int(ms)
    return hrs * 3600 + mins * 60 + secs

def obtain_timestamp_from_subtitle(subtitle, starting_timestamp_for_subtitles, duration):
    if "timestamp" in subtitle:
        start, end = subtitle["timestamp"]

        if not isinstance(end, float):
            end = duration

        start -= starting_timestamp_for_subtitles
        end -= starting_timestamp_for_subtitles

        subtitle_timestamp = (start + end) / 2

    else:
        start, end = subtitle["start"], subtitle["end"]
        start = convert_time_to_frame(start)
        end = convert_time_to_frame(end)
        start -= starting_timestamp_for_subtitles
        end -= starting_timestamp_for_subtitles

        subtitle_timestamp = (start + end) / 2

    return start, end

def process_subtitles(subtitles: list[dict], 
                      starting_timestamp: int, 
                      original_fps: int, 
                      original_frame_count: int, 
                      original_duration: int, 
                      desired_frame_count: int, 
                      frame_step: int):
    # Returns a list of per frame subtitles

    all_subtitles = [[] for _ in range(desired_frame_count)]
    
    subtitle_index = 0
    frames_by_subtitle = [[] for _ in range(desired_frame_count)]
    for sub_idx, subtitle in enumerate(subtitles):
        start_idx, end_idx = obtain_timestamp_from_subtitle(subtitle, starting_timestamp, original_duration)
        subtitle_start_idx = start_idx * original_fps
        subtitle_end_idx = end_idx * original_fps
        if subtitle_end_idx < 0:
            continue
        if subtitle_start_idx > desired_frame_count*frame_step:
            break
        subtitle_start_idx = int(subtitle_start_idx/frame_step)
        subtitle_end_idx = int(subtitle_end_idx/frame_step)
        subtitle_start_idx = max(subtitle_start_idx, 0)
        subtitle_end_idx = min(subtitle_end_idx, desired_frame_count)

        for idx_ in range(subtitle_start_idx, subtitle_end_idx+1):
            idx_ = min(idx_, desired_frame_count-1)
            text = subtitle['line'] if 'line' in subtitle else subtitle['text']
            all_subtitles[idx_].append(text)
    # for frame_idx in range(desired_frame_count):
    #         # next_start_idx, next_end_idx = next_subtitle["start"], next_subtitle["end"]
    #         annotated = False
    #         while not annotated:
    #             subtitle = subtitles[subtitle_index]
    #             # next_subtitle = subtitles[subtitle_index + 1]
    #             start_idx, end_idx = subtitle["start"], subtitle["end"]
    #             subtitle_start_idx = int(convert_time_to_frame(start_idx, original_fps))
    #             subtitle_end_idx = int(convert_time_to_frame(end_idx, original_fps))

    #             original_frame_idx = (frame_idx - starting_timestamp) * frame_step
    #             if original_frame_idx < 0:
    #                 annotated = True
    #                 continue

    #             if original_frame_idx < subtitle_start_idx:
    #                 annotated = True
    #                 continue
    #             elif original_frame_idx > subtitle_end_idx:
    #                 subtitle_index += 1
                
    #             elif original_frame_idx >= subtitle_start_idx and original_frame_idx <= subtitle_end_idx:
    #                 annotated = True
    #                 all_subtitles[original_frame_idx].append(subtitle['line'])
        

    return all_subtitles
    
class LongVideoBench(Manager):
    def __init__(self):
        self._dataset_path = "/nas/mars/dataset/longvideobench/LongVideoBench/"
        self._burned_path = "/nas/mars/dataset/longvideobench/"
        self._nsvs_path = (
            "/nas/mars/experiment_result/nsvqa/5_full_output/longvideobench_output.json"
        )
        # self._output_path_position = "/nas/mars/experiment_result/nsvqa/6_formatted_output/longvideobench_position"
        # self._categories = [
        #     "S2E",
        #     "S2O",
        #     "S2A",
        #     "E2O",
        #     "T2A",
        #     "T2E",
        #     "T2O",
        #     "O2E",
        #     "SSS",
        #     "TAA",
        #     "E3E",
        #     "SAA",
        #     "T3O",
        #     "TOS",
        #     "O3O",
        #     "SOS",
        #     "T3E",
        # ]
        self.read_number = 44

    def load_data(self):
        category_buckets = defaultdict(list)

        with open(
            os.path.join(self._dataset_path, "lvb_val.json"), "r", encoding="utf-8"
        ) as f:
            dataset = json.load(f)
            for item in dataset:
                cat = item["question_category"]
                # if (
                #     cat in self._categories
                #     and len(category_buckets[cat]) < self.read_number
                # ):
                video_path = os.path.join(
                    self._burned_path, "burn-subtitles", f"{item['video_id']}.mp4"
                )
                if not os.path.exists(video_path):
                    print(f"Burnt Video Does Not Exist: {video_path}")
                    continue

                raw_video_path = os.path.join(
                    self._dataset_path, "videos", item["video_path"]
                )
                subtitle_path = os.path.join(
                    self._dataset_path, "subtitles", item["subtitle_path"]
                )
                question = f"{item['question']} here are the candidates: "
                for choice_idx, candidate in enumerate(item["candidates"]):
                    question += f"\n{choice_idx}. {candidate}"
                question += "\n It's a multiple choice question. You must choose the correct answer as a number."

                entry = {
                    "question": question,
                    "candidates": item["candidates"],
                    "correct_choice": item["correct_choice"],
                    "paths": {
                        "raw_video_path": raw_video_path,
                        "subtitle_path": subtitle_path,
                        "video_path": video_path,
                    },
                    "metadata": {
                        "video_id": item["video_id"],
                        "id": item["id"],
                        "position": item["position"],
                        "question_wo_referring_query": item[
                            "question_wo_referring_query"
                        ],
                        "topic_category": item["topic_category"],
                        "question_category": cat,
                        "level": item["level"],
                        "duration_group": item["duration_group"],
                        "starting_timestamp_for_subtitles": item[
                            "starting_timestamp_for_subtitles"
                        ],
                        "duration": item["duration"],
                        "view_count": item["view_count"],
                    },
                }
                category_buckets[cat].append(entry)

        # Flatten list of all selected entries from each category
        return [entry for entries in category_buckets.values() for entry in entries]

    


    def save_it_as_vseek_data(self, desired_interval_in_sec: int = 1):
        try:
            data = self.load_data()
            window_size = DATA_SETTING.window_size

            # Allow overriding workers via env; keep conservative default for GPU workloads
            default_workers = min(4, (os.cpu_count() or 4))
            max_workers = int(os.getenv("VSEEK_WORKERS", default_workers))

            # Deduplicate entries by video_id (same video can appear across categories)
            unique_entries: list[dict] = []
            seen_ids: set[str] = set()
            
            for e in data:
                vid = e.get("metadata", {}).get("video_id")
                if not vid or vid in seen_ids:
                    continue
                seen_ids.add(vid)
                unique_entries.append(e)

            # In-process guard against accidental duplicate scheduling
            processed_ids: set[str] = set()
            processed_lock = Lock()

            def _process_single_entry(entry: dict):
                try:
                    unique_id = entry["metadata"]["video_id"]

                    # Fast skip if already processed on disk
                    if self.is_file_exists(window_size, unique_id):
                        return

                    # Prevent duplicate work in this process
                    with processed_lock:
                        if unique_id in processed_ids:
                            return
                        processed_ids.add(unique_id)
                    subtitles = json.load(open(entry["paths"]["subtitle_path"], "r"))
                    video = read_video(video_path=entry["paths"]["video_path"])
                    vclip = ViClip(
                        pretrained_model_path=VICLIP_SETTING.vclip_model_path,
                        gpu_number=VICLIP_SETTING.gpu_number,
                    )
                    frame_index = 0
                    window_index = 0
                    subtitle_index = 0
                    video_frames = VideoFrames(window_size=DATA_SETTING.window_size)

                    all_frames = video.get_all_frames_of_video(desired_interval_in_sec=desired_interval_in_sec)
                    video_frames.add_all_frames(all_frames)
                    video_frames.partition_frames()

                    print(f"Added {len(all_frames)} frames")
                    print(f"Processed video frames {video.video_info.processed_frame_count}")

                    original_fps = video.video_info.original_fps
                    original_frame_count = video.video_info.original_frame_count
                    original_duration = video.video_info.original_duration
                    desired_frame_count = video.video_info.processed_frame_count
                    frame_step = video.get_frame_step(desired_interval_in_sec=desired_interval_in_sec)

                    # print(f"Subtitles {subtitles}")
                    all_subtitles = process_subtitles(subtitles,
                                                      entry["metadata"]["starting_timestamp_for_subtitles"],
                                                      original_fps,
                                                      original_frame_count,
                                                      original_duration,
                                                      desired_frame_count,
                                                      frame_step)

                    print(f"Processed subtitles {all_subtitles}")
                    video_frames.add_all_subtitles(all_subtitles)
                    video_frames.partition_subtitles()
                    print(f"Added {len(all_subtitles)} subtitles")

                    all_unique_subtitles = video_frames.window_by_subtitle.keys()

                    for window_idx in video_frames.frames_by_window.keys():
                        frame_chunk = video_frames.get_frame_chunk(window_idx)
                        video_frames.add_embedding(vclip.get_feature(frame_chunk))

                    for subtitle in all_unique_subtitles:
                        video_frames.add_subtitle_embedding(vclip.get_text_embedding(subtitle))

                    # Saving data
                    dataset_name = "lvb"
                    dir_name = f"{dataset_name}_window_{window_size}"
                    file_name = f"{unique_id}.pkl"
                    output_path = Path(DATA_SETTING.output_dir).joinpath(
                        dir_name, file_name
                    )
                    video_frames.save(str(output_path))
                except Exception as e:
                    print(f"Error processing entry {entry.get('metadata', {}).get('video_id', 'unknown')}: {e}")
                    print(traceback.format_exc())

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                list(tqdm(executor.map(_process_single_entry, unique_entries), total=len(unique_entries)))
        except Exception as e:
            print(f"Error saving data: {e}")
            print(traceback.format_exc())

    def is_file_exists(self, window_size: str, unique_id: str):
        dataset_name = "lvb"
        dir_name = f"{dataset_name}_window_{window_size}"
        data_path = Path(DATA_SETTING.output_dir).joinpath(dir_name)
        if not data_path.exists():
            return False
        files = [dir_path.stem for dir_path in data_path.iterdir()]
        if unique_id in files:
            metadata_path = data_path.joinpath(unique_id, "metadata.json")
            if metadata_path.exists():
                return True
            else:
                invalid_data_path = data_path.joinpath(unique_id)
                if invalid_data_path.exists() and invalid_data_path.is_dir():
                    shutil.rmtree(invalid_data_path)
        return False

    def postprocess_data(self): ...
