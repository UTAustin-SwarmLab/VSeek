from pathlib import Path

from vseek.data.frame import VideoFrames

DATA_PATH = "/nas/mars/dataset/vseek/dataset/lvb_window_10/"

if __name__ == "__main__":
    data_path = Path(DATA_PATH)
    for dir_path in data_path.iterdir():
        video_frames = VideoFrames.load(dir_path)
        # TODO: you can load the data from the file
        print(len(video_frames.embeddings))
