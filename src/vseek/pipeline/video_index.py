from vseek.data.frame import SingleFrame, VideoFrames
from vseek.setting import ViClipSetting
from vseek.video.read_video import read_video
from vseek.video_embedding.video_clip import ViClip

VICLIP_SETTING = ViClipSetting()


def video_indexing_pipeline(
    video_path: str,
    desired_interval_in_sec: int = 1,
    window_size: int = 10,
    gpu_number: int = 0,
):
    video = read_video(video_path=video_path)
    vclip = ViClip(
        pretrained_model_path=VICLIP_SETTING.vclip_model_path, gpu_number=gpu_number
    )
    frame_index = 0
    window_index = 0
    video_frames = VideoFrames(window_size=window_size)
    while True:
        frame = video.get_next_frame(desired_interval_in_sec=desired_interval_in_sec)
        if frame is None:
            break

        video_frames.add_frame(
            frame=SingleFrame(
                frame_idx=frame_index,
                real_video_index=video.current_frame_index,
                image=frame,
            )
        )
        if frame_index % window_size == 0 and frame_index != 0:
            frame_chunk = video_frames.get_frame_chunk(window_index)
            # Vclip
            video_frames.add_embedding(vclip.get_feature(frame_chunk))
            window_index += 1
        frame_index += 1
    return video_frames
