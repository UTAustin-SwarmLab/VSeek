from pydantic_settings import BaseSettings


class ViClipSetting(BaseSettings):
    vclip_model_path: str = (
        "/nas/mars/model_weights/viclip/ViClip-InternVid-10M-FLT.pth"
    )
    text_encoder_model_path: str = (
        "/nas/mars/model_weights/viclip/bpe_simple_vocab_16e6.txt.gz"
    )

    class Config:
        env_file = "/home/mc76728/repo/Swarmlab/VSeek-R1/.env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Ignore extra environment variables
