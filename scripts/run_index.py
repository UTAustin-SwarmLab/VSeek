    
if __name__ =="__main__":    
    video_frames: VideoFrames = video_indexing_pipeline(
        video_path="/nas/mars/dataset/LongVideoBench/burn-subtitles/zVudr8cxHRE.mp4",
        desired_interval_in_sec=1,
        gpu_number=VCLIP_DEVICE,
        window_size=DATA_SETTING.window_size,
    )
    video_frames.save(f"{OUTPUT_DIR}/video_frames.pkl")
    