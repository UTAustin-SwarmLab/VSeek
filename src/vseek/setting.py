from typing import Optional

from pydantic_settings import BaseSettings

ENV_FILE_PATH = "/home/mc767728/repo/Swarmlab/VSeek-R1/.env"


class DataSetting(BaseSettings):
    dataset_name: str = "LongVideoBench"
    output_dir: str = "/nas/mars/dataset/vseek/dataset"
    window_size: int = 10

    class Config:
        env_file = ENV_FILE_PATH
        env_file_encoding = "utf-8"
        extra = "ignore"  # Ignore extra environment variables


class ViClipSetting(BaseSettings):
    vclip_model_path: str = (
        "/nas/mars/model_weights/viclip/ViClip-InternVid-10M-FLT.pth"
    )
    text_encoder_model_path: str = (
        "/nas/mars/model_weights/viclip/bpe_simple_vocab_16e6.txt.gz"
    )
    gpu_number: int = 4

    class Config:
        env_file = ENV_FILE_PATH
        env_file_encoding = "utf-8"
        extra = "ignore"  # Ignore extra environment variables


class VLLMSetting(BaseSettings):
    openai_api_key: Optional[str] = "EMPTY"
    api_base: str = "http://localhost:8002/v1"
    model: str = "Qwen/Qwen2.5-VL-7B-Instruct"

    class Config:
        env_file = ENV_FILE_PATH
        env_file_encoding = "utf-8"
        extra = "ignore"  # Ignore extra environment variables
