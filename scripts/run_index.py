    
from vseek.data.exp_io import DataInput
from vseek.data.frame import VideoFrames
from vseek.setting import DataSetting, VLLMSetting
from vseek.pipeline.video_index import video_indexing_pipeline
VLLM_SETTING = VLLMSetting()
DATA_SETTING = DataSetting()

VIDEOS = [
    {
        "path": "/nas/mars/dataset/LongVideoBench/burn-subtitles/zVudr8cxHRE.mp4",
        "query": "a unicorn is prancing until a pizza eats a strawberry",
    }
]
VCLIP_DEVICE = 7  # GPU device index
OPENAI_SAVE_PATH = "/nas/mars/experiment_result/nsvs/openai_conversation_history/"
OUTPUT_DIR = "output"

if __name__ =="__main__":    
    video_frames: VideoFrames = video_indexing_pipeline(
        video_path="/nas/mars/dataset/LongVideoBench/burn-subtitles/zVudr8cxHRE.mp4",
        desired_interval_in_sec=1,
        gpu_number=VCLIP_DEVICE,
        window_size=DATA_SETTING.window_size,
    )
    video_frames.save(f"{OUTPUT_DIR}/video_frames.pkl")
