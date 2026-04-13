import pdb
from pprint import pprint

import pandas as pd
from torch.utils.data import Dataset
from util import load_json, load_pkl, makedir, parse_args, save_json, save_pkl


class BaseDataset(Dataset):
    def __init__(self, args, quids_to_exclude=None, num_examples_to_run=-1):
        """
        num_examples_to_run < 0: run all
        """
        self.args = args
        self.narrations = self.get_descriptions()  # uid --> list of str  or  uid --> str
        self.anno = self.get_anno()
        self.durations = load_json(args.duration_path)  # uid --> float
        data = self.build()
        data = self.filter(data, quids_to_exclude, num_examples_to_run)
        self.data = data

    def set_ukey(self, name):
        self.ukey = name

    def filter(self, data, quids_to_exclude, num_examples_to_run):
        if quids_to_exclude is not None:
            data = [el for el in data if el[self.ukey] not in quids_to_exclude]
        if num_examples_to_run >= 0:
            data = data[:num_examples_to_run]
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class EgoSchemaDataset(BaseDataset):
    def __init__(self, args, quids_to_exclude=None, num_examples_to_run=-1):
        self.set_ukey("uid")
        super().__init__(args, quids_to_exclude=quids_to_exclude, num_examples_to_run=num_examples_to_run)

    def get_descriptions(self):
        narrations = load_json(self.args.data_path)
        return narrations

    def format_narration(self, narr):
        if isinstance(narr, list):
            narr = ". ".join(narr)
        return narr

    def get_anno(self):
        anno = load_json(
            self.args.anno_path
        )  # uid --> {question, option 0, option 1, option 2, option 3, option 4, truth (optional)}
        return anno

    def build(self):
        data = []
        for uid, item in self.anno.items():
            if uid not in self.narrations:
                continue
            narration = self.format_narration(self.narrations[uid])

            question = item["question"]

            choices = [item["option 0"], item["option 1"], item["option 2"], item["option 3"], item["option 4"]]
            truth = item["truth"] if "truth" in item else -1
            duration = int(self.durations[uid])
            data.append(
                {
                    "uid": uid,
                    "narration": narration,
                    "question": question,
                    "optionA": choices[0],
                    "optionB": choices[1],
                    "optionC": choices[2],
                    "optionD": choices[3],
                    "optionE": choices[4],
                    "truth": truth,
                    "duration": duration,
                }
            )
        return data


class NextDataset(BaseDataset):
    def __init__(self, args, quids_to_exclude=None, num_examples_to_run=-1):
        self.set_ukey("quid")
        super().__init__(args, quids_to_exclude=quids_to_exclude, num_examples_to_run=num_examples_to_run)

    def get_descriptions(self):
        narrations = load_json(self.args.data_path)
        return narrations

    def format_narration(self, narr):
        if isinstance(narr, list):
            caption_every = int(1 / self.args.fps)
            narr = ".\n".join([f"{int(i*caption_every)}: {cap}" for i, cap in enumerate(narr[::caption_every])])
        return narr

    def get_anno(self):
        return pd.read_csv(
            self.args.anno_path
        )  # video,frame_count,width,height,question,answer,qid,type,a0,a1,a2,a3,a4

    def build(self):
        data = []
        for row in self.anno.iterrows():
            if isinstance(row, tuple):
                row = row[-1]  # remove table index
            uid = str(row["video"])
            if uid not in self.narrations:
                continue
            question, truth = row["question"], row["answer"]
            qid, q_type = row["qid"], row["type"]
            choices = [row["a0"], row["a1"], row["a2"], row["a3"], row["a4"]]
            quid = f"{uid}_{qid}"
            narration = self.format_narration(self.narrations[uid])
            duration = int(self.durations[uid])
            data.append(
                {
                    "quid": quid,
                    "uid": uid,
                    "qid": qid,
                    "q_type": q_type,
                    "narration": narration,
                    "question": question,
                    "optionA": choices[0],
                    "optionB": choices[1],
                    "optionC": choices[2],
                    "optionD": choices[3],
                    "optionE": choices[4],
                    "truth": truth,
                    "duration": duration,
                }
            )
        return data


class LVBDataSet(BaseDataset):
    def __init__(self, args, quids_to_exclude=None, num_examples_to_run=-1):
        self.set_ukey("uid")  # Use 'uid' as the unique identifier for LVB
        self.args = args
        self.anno = self.get_anno()
        self.narrations = self.get_descriptions()  # uid --> list of str  or  uid --> str

        data = self.build()
        data = self.filter(data, quids_to_exclude, num_examples_to_run)
        self.data = data

    def get_descriptions(self):
        caption_path = getattr(self.args, "caption_path", None) or getattr(self.args, "data_path", None)
        if not caption_path:
            return {}
        narrations = load_json(caption_path)
        data = {}
        for vid, item in narrations.items():
            list_captions = []
            for _, caption in item.items():
                list_captions.append(caption["caption"])
            data[vid] = list_captions
        anno_keys = set(item["video_id"] for item in self.anno.values())
        intersect = set(data.keys()) & set(anno_keys)
        # there are some keys in data that are a portion of the keys in self.anno
        print(len(data.keys()), len(anno_keys))
        not_in_data = set(data.keys()) - intersect
        for key in not_in_data:
            for key_anno in anno_keys:
                if key in key_anno:
                    modified_key = key_anno
                    data[modified_key] = data[key]
                    del data[key]

        intersect = set(data.keys()) & set(anno_keys)
        assert len(intersect) == len(anno_keys)

        return data

    def get_anno(self):
        # Load the LVB data.json file
        import json

        with open(self.args.lvb_data_path, "r") as f:
            data = json.load(f)
        # If the file is a list, convert to dict by id
        if isinstance(data, list):
            anno = {item["id"]: item for item in data}
        elif isinstance(data, dict):
            # If the file is a dict, assume it's id -> item
            anno = data
        else:
            raise ValueError("Unexpected LVB data format")
        return anno

    def format_narration(self, narr):
        if isinstance(narr, list):
            # Prefix each caption with #C delimiter so that when split in prompts.py,
            # parts[i] + parts[i+1] correctly reconstructs each caption with its delimiter
            caption_every = int(1 / self.args.fps)
            narr = "".join(
                [
                    "#C Frame" + str(i * caption_every) + ": " + caption
                    for i, caption in enumerate(narr[::caption_every])
                ]
            )
        return narr

    def build(self):
        data = []
        for uid, item in self.anno.items():
            # Candidates and correct_choice are required
            candidates = item.get("candidates", [])
            correct_choice = item.get("correct_choice", -1)

            # Convert candidates to option format for compatibility
            options = candidates if len(candidates) >= 5 else candidates + [""] * (5 - len(candidates))

            # Get video_id for frame feature loading
            video_id = item.get("video_id", uid)
            narration = ""
            if video_id in self.narrations:
                narration = self.format_narration(self.narrations[video_id])

            # Compose the data dict with required fields for breath expansion
            entry = {
                "uid": uid,  # Primary identifier for breath expansion
                "video_id": video_id,  # For frame feature loading
                "question": item.get("question", ""),
                "question_wo_referring_query": item.get("question_wo_referring_query", ""),
                "optionA": options[0] if len(options) > 0 else "",
                "optionB": options[1] if len(options) > 1 else "",
                "optionC": options[2] if len(options) > 2 else "",
                "optionD": options[3] if len(options) > 3 else "",
                "optionE": options[4] if len(options) > 4 else "",
                "candidates": candidates,
                "correct_choice": correct_choice,
                "truth": correct_choice,  # For compatibility with evaluation
                "position": item.get("position", []),
                "topic_category": item.get("topic_category", ""),
                "question_category": item.get("question_category", ""),
                "level": item.get("level", ""),
                "video_path": item.get("video_path", ""),
                "subtitle_path": item.get("subtitle_path", ""),
                "duration_group": item.get("duration_group", None),
                "starting_timestamp_for_subtitles": item.get("starting_timestamp_for_subtitles", None),
                "duration": item.get("duration", None),
                "view_count": item.get("view_count", None),
                # Add empty narration for compatibility (LVB doesn't use narrations)
                "narration": narration,
            }
            data.append(entry)
        return data


def get_dataset(args, quids_to_exclude=None, num_examples_to_run=-1):
    if args.dataset == "egoschema":
        return EgoSchemaDataset(args, quids_to_exclude=quids_to_exclude, num_examples_to_run=num_examples_to_run)
    elif args.dataset == "lvb":
        return LVBDataSet(args, quids_to_exclude=quids_to_exclude, num_examples_to_run=num_examples_to_run)
    else:
        return NextDataset(args, quids_to_exclude=quids_to_exclude, num_examples_to_run=num_examples_to_run)


if __name__ == "__main__":
    args = parse_args()
    dataset = get_dataset(args, num_examples_to_run=args.num_examples_to_run)
    print(len(dataset))
    # for data in dataset:
    #     pprint(data)
