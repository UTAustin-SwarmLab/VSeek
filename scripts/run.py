from vseek.data.exp_io import DataInput
from vseek.data.frame import VideoFrames
from vseek.setting import DataSetting, VLLMSetting
from vseek.vlm.vseek_vllm import VSeekAgent

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


if __name__ == "__main__":
    # video_frames: VideoFrames = video_indexing_pipeline(
    #     video_path="/nas/mars/dataset/LongVideoBench/burn-subtitles/zVudr8cxHRE.mp4",
    #     desired_interval_in_sec=1,
    #     gpu_number=VCLIP_DEVICE,
    #     window_size=DATA_SETTING.window_size,
    # )
    # video_frames.save(f"{OUTPUT_DIR}/video_frames.pkl")
    video_frames = VideoFrames.load(f"{OUTPUT_DIR}/video_frames.pkl")

    data_input = DataInput(
        video=video_frames,
        question="Is there a unicorn prancing until a pizza eats a strawberry? Please answer in yes or no.",  # Q
        answer="yes",  # A
    )

    # VLM pipeline
    vlm_client = VSeekAgent(
        api_key=VLLM_SETTING.openai_api_key,
        api_base=VLLM_SETTING.api_base,
        model=VLLM_SETTING.model,
    )
    # VLM(V, Q, A)
    trajectory = vlm_client.run(data_input)
    print(trajectory)
