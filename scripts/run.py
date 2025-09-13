from vseek.data.exp_io import DataInput
from vseek.data.frame import VideoFrames
from vseek.pipeline.video_index import video_indexing_pipeline

VIDEOS = [
    {
        "path": "/nas/mars/dataset/LongVideoBench/burn-subtitles/zVudr8cxHRE.mp4",
        "query": "a unicorn is prancing until a pizza eats a strawberry",
    }
]
DEVICE = 7  # GPU device index
OPENAI_SAVE_PATH = "/nas/mars/experiment_result/nsvs/openai_conversation_history/"
OUTPUT_DIR = "output"


if __name__ == "__main__":
    video_frames: VideoFrames = video_indexing_pipeline(
        video_path="/nas/mars/dataset/LongVideoBench/burn-subtitles/zVudr8cxHRE.mp4",
        desired_interval_in_sec=1,
    )

    data_input = DataInput(
        video=video_frames,
        question="What is the video about?",  # Q
        answer="The video is about a unicorn prancing until a pizza eats a strawberry.",  # A
    )

    # VLM pipeline
    # VLM(V, Q, A)
