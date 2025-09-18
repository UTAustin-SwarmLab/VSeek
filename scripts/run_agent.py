from vseek.data.exp_io import DataInput
from vseek.data.frame import VideoFrames
from vseek.setting import DataSetting, VLLMSetting
from vseek.agent.video_agent import VSeekAgent
from vseek.pipeline.video_index import video_indexing_pipeline

VLLM_SETTING = VLLMSetting()
DATA_SETTING = DataSetting()

VIDEOS = [
    {
        "path": "/nas/mars/dataset/LongVideoBench/burn-subtitles/zVudr8cxHRE.mp4",
        "query": "A man is in front of red bus until a mobile phone appear. What happens after the mobile phone disappers?",
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
        question="A man is in front of blue bus until a mobile phone appears. What happens after the mobile phone disappers?",
        options="Options: A. A woman with blue bag appears, B. The man starts dancing, C. A screen with 20 minutes later shows",# Q
        answer="yes",  # A
    )

    # VLM pipeline with reduced image dimensions for easier parsing
    answer = []
    for i in range(10):
        vlm_client = VSeekAgent(
            api_key=VLLM_SETTING.openai_api_key,
            api_base=VLLM_SETTING.api_base,
            model=VLLM_SETTING.model,
            max_image_width=384,  # Reduced from default 512
            max_image_height=384,  # Reduced from default 512
            image_quality=90,  # Higher quality for better parsing
        )
        # VLM(V, Q, A)
        trajectory = vlm_client.run(data_input)
        answer.append(trajectory.answer)
        
    print(answer)
