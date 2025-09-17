from pathlib import Path

from vseek.agent.video_agent import VSeekAgent
from vseek.data.frame import VideoFrames

DATA_PATH = "/nas/mars/dataset/vseek/dataset/lvb_window_10/"
FILE_NAME = "86CxyhFV9MI"
if __name__ == "__main__":
    data_path = Path(DATA_PATH).joinpath(FILE_NAME)
    video_frames = VideoFrames.load(str(data_path))

    video_agent = VSeekAgent()

    # Flatten the dictionary to a list in the same order
    subtitle_list = [
        video_frames.unique_subtitles_by_window[i]
        for i in sorted(video_frames.unique_subtitles_by_window.keys())
    ]

    search_subtitle = video_frames.unique_subtitles_by_window[11]
    idx = video_agent.search_subtitle(subtitle_list, search_subtitle)
    print(f"Index of the most similar subtitle: {idx}")
