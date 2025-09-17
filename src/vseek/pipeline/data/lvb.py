import copy
import hashlib
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from vseek.data.frame import SingleFrame, VideoFrames
from vseek.pipeline.data.manager import Manager
from vseek.setting import DataSetting, ViClipSetting
from vseek.video.read_video import read_video
from vseek.video_embedding.video_clip import ViClip

VICLIP_SETTING = ViClipSetting()

DATA_SETTING = DataSetting()


class LongVideoBench(Manager):
    def __init__(self):
        self._dataset_path = "/nas/mars/dataset/longvideobench/LongVideoBench/"
        self._burned_path = "/nas/mars/dataset/longvideobench/"
        self._nsvs_path = (
            "/nas/mars/experiment_result/nsvqa/5_full_output/longvideobench_output.json"
        )
        self._output_path_nsvqa = "/nas/mars/experiment_result/nsvqa/6_formatted_output/longvideobench_nsvqa_all_categories"
        self._output_path_full = "/nas/mars/experiment_result/nsvqa/6_formatted_output/longvideobench_full_all_categories"
        # self._output_path_position = "/nas/mars/experiment_result/nsvqa/6_formatted_output/longvideobench_position"
        self._categories = [
            "S2E",
            "S2O",
            "S2A",
            "E2O",
            "T2A",
            "T2E",
            "T2O",
            "O2E",
            "SSS",
            "TAA",
            "E3E",
            "SAA",
            "T3O",
            "TOS",
            "O3O",
            "SOS",
            "T3E",
        ]
        self.read_number = 44

    def load_data(self):
        category_buckets = defaultdict(list)

        with open(
            os.path.join(self._dataset_path, "lvb_val.json"), "r", encoding="utf-8"
        ) as f:
            dataset = json.load(f)
            for item in dataset:
                cat = item["question_category"]
                if (
                    cat in self._categories
                    and len(category_buckets[cat]) < self.read_number
                ):
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
                        choice_idx += 1
                        question += f"\n{choice_idx + 1}. {candidate}"
                    question += "It's a multiple choice question. You must choose the correct answer as a number."

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
        data = self.load_data()
        window_size = DATA_SETTING.window_size
        for entry in tqdm(data):
            if self.is_file_exists(window_size, entry["metadata"]["video_id"]):
                continue
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
            while True:
                try:
                    current_subtitle = subtitles[subtitle_index]
                except IndexError:
                    current_subtitle = subtitles[-1]
                subtitle_start_timestamp, subtitle_end_timestamp = (
                    current_subtitle["start"],
                    current_subtitle["end"],
                )
                subtitle_text = current_subtitle["line"]

                frame = video.get_next_frame(
                    desired_interval_in_sec=desired_interval_in_sec
                )
                if frame is None:
                    break

                video_frames.add_frame(
                    frame=SingleFrame(
                        frame_idx=frame_index,
                        real_video_index=video.current_frame_index,
                        image=frame,
                    )
                )
                # Add subtitle
                frame_start_timestamp, frame_end_timestamp = video.current_timestamp
                if frame_start_timestamp >= subtitle_end_timestamp:
                    subtitle_index += 1

                video_frames.add_subtitle(subtitle_text)
                if frame_index % window_size == 0 and frame_index != 0:
                    frame_chunk = video_frames.get_frame_chunk(window_index)
                    # Adding video embedding
                    video_frames.add_embedding(vclip.get_feature(frame_chunk))
                    window_index += 1
                frame_index += 1

            # Saving data
            dataset_name = "lvb"
            unique_id = entry["metadata"]["video_id"]
            dir_name = f"{dataset_name}_window_{window_size}"
            file_name = f"{unique_id}.pkl"
            output_path = Path(DATA_SETTING.output_dir).joinpath(dir_name, file_name)
            video_frames.save(str(output_path))

    def is_file_exists(self, window_size: str, unique_id: str):
        dataset_name = "lvb"
        dir_name = f"{dataset_name}_window_{window_size}"
        data_path = Path(DATA_SETTING.output_dir).joinpath(dir_name)
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

    def postprocess_data(self):
        with open(os.path.join(self._dataset_path, "lvb_val.json"), "r") as f:
            lvb_data = json.load(f)
        with open(self._nsvs_path, "r") as f:
            nsvs_data = json.load(f)

        output_nsvqa = []  # nsvqa cropped video
        output_full = []  # entire video
        for entry_nsvs in tqdm(nsvs_data):
            found = False
            for entry in lvb_data:
                if (
                    entry["question"] == entry_nsvs["question"]
                    and entry["id"] == entry_nsvs["metadata"]["id"]
                ):
                    found = True

                    candidates = entry["candidates"]
                    for i in range(5):
                        if i < len(candidates):
                            entry[f"option{i}"] = candidates[i]
                        else:
                            entry[f"option{i}"] = "N/A"

                    entry_full = copy.deepcopy(entry)

                    code = entry["question"] + entry["id"]
                    id = hashlib.sha256(code.encode()).hexdigest()
                    entry["id"] = id + "_0"
                    entry["video_id"] = id
                    entry["video_path"] = id + ".mp4"

                    self.crop_video(
                        entry_nsvs,
                        save_path=os.path.join(
                            self._output_path_nsvqa, "videos", entry["video_path"]
                        ),
                        hardset=False,
                    )

                    if os.path.exists(
                        os.path.join(
                            self._output_path_nsvqa, "videos", entry["video_path"]
                        )
                    ):  # if crop is successful
                        # self.crop_video(
                        #     entry_nsvs,
                        #     save_path=os.path.join(self._output_path_position, "videos", entry["video_path"]),
                        #     hardset=True
                        # )

                        output_nsvqa.append(entry)
                        output_full.append(entry_full)

            if found == False:
                print(f"Entry not found for question: {entry_nsvs['question']}")

        with open(os.path.join(self._output_path_nsvqa, "lvb_val.json"), "w") as f:
            json.dump(output_nsvqa, f, indent=4)
        # with open(os.path.join(self._output_path_position, "lvb_val.json"), "w") as f:
        #     json.dump(output_nsvqa, f, indent=4)
        with open(os.path.join(self._output_path_full, "lvb_val.json"), "w") as f:
            json.dump(output_full, f, indent=4)
            json.dump(output_full, f, indent=4)
