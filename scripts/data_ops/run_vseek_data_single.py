from pathlib import Path

from vseek.data.frame import VideoFrames

DATA_PATH = "/nas/mars/dataset/vseek/dataset/lvb_window_10/"
FILE_NAME = "86CxyhFV9MI"
if __name__ == "__main__":
    data_path = Path(DATA_PATH).joinpath(FILE_NAME)
    video_frames = VideoFrames.load(str(data_path))
    # TODO: you can load the data from the file
    print(len(video_frames.embeddings))
