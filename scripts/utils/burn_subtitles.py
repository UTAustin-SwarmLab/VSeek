import json
import os
import subprocess
from datetime import datetime
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import shutil

def time_to_seconds(time_str):
    time_str = time_str.replace(" ", "")
    t = datetime.strptime(time_str, "%H:%M:%S.%f")
    return t.hour * 3600 + t.minute * 60 + t.second


def seconds_to_srt_format(seconds):
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000.0))
    hours, rem = divmod(total_ms, 3600 * 1000)
    minutes, rem = divmod(rem, 60 * 1000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"

def obtain_timestamp_from_subtitle(subtitle, starting_timestamp_for_subtitles, duration):
    if "timestamp" in subtitle:
        start, end = subtitle["timestamp"]

        if not isinstance(end, float):
            end = duration

        start -= starting_timestamp_for_subtitles
        end -= starting_timestamp_for_subtitles

        subtitle_timestamp = (start + end) / 2

    else:
        start, end = subtitle["start"], subtitle["end"]
        start = time_to_seconds(start)
        end = time_to_seconds(end)
        start -= starting_timestamp_for_subtitles
        end -= starting_timestamp_for_subtitles

        subtitle_timestamp = (start + end) / 2

    return start, end
       
def burn_subtitles_on_video(video_path, subtitles_json_path, starting_timestamp, save_path, video_duration,
                             font_size=24, font="Arial-Bold", color="white"):
    with open(subtitles_json_path, "r") as f:
        subtitles = json.load(f)

    bad = False
    prevstart = -1000
    minstartdelta = 100000
    
    modified_subtitles = []
    
    for subtitle in subtitles:  
        start, end = obtain_timestamp_from_subtitle(subtitle, starting_timestamp, video_duration)

        subtitle_timestamp = (start + end) / 2
        if end - start < 1:
            end = subtitle_timestamp + 0.5
            start = subtitle_timestamp - 0.5
        if end < 0:
            continue
        if start > video_duration:
            break
        start = max(start, 0)
        end = min(end, video_duration)
        text = subtitle["line"] if "line" in subtitle else subtitle["text"]
        modified_subtitles.append({"start": start, "end": end, "line": text})

    temp_path = save_path.replace(".mp4", "")
    temp_srt_path = f"{temp_path}_temp.srt"
    

    with open(temp_srt_path, "w", encoding="utf-8") as out:
        for i, entry in enumerate(modified_subtitles, start=1):
            start_str = seconds_to_srt_format(entry["start"])
            end_str = seconds_to_srt_format(entry["end"])
            out.write(f"{i}\n{start_str} --> {end_str}\n{entry['line']}\n\n")

    # Detect if input has audio; avoid failing when copying non-existent audio
    def _has_audio_stream(path: str) -> bool:
        try:
            probe = subprocess.run(
                [
                    "ffprobe", "-v", "quiet", "-select_streams", "a:0",
                    "-show_entries", "stream=index", "-of", "csv=p=0", path,
                ],
                capture_output=True, text=True, check=True,
            )
            return probe.stdout.strip() != ""
        except Exception:
            return False

    #audio_args = ["-c:a", "copy"] if _has_audio_stream(video_path) else ["-an"]
    audio_args = ["-c:a", "copy"] 
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", video_path,
        "-vf", f"subtitles={temp_srt_path}:force_style='PrimaryColour=&HFFFFFF&,BackColour=&H000000&,BorderStyle=3'",
        "-c:v", "libx264",
        *audio_args,
        "-y", save_path,
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error for {video_path}: {e}")
        with open("err.txt", "a") as error_log:
            error_log.write(f"FFmpeg error for {video_path}: {e}\n")
            # copy the original video to the save path
            shutil.copy(video_path, save_path)
            print(f"Copied original video to {save_path}")
            if e.stderr:
                try:
                    error_log.write(e.stderr.decode("utf-8", errors="ignore") + "\n")
                except AttributeError:
                    error_log.write(str(e.stderr) + "\n")
    finally:
        try:
            os.remove(temp_srt_path)
        except FileNotFoundError:
            pass


def process_entry(entry):
    video_id = entry["video_id"]
    video_path = f"/nas/mars/dataset/longvideobench/LongVideoBench/videos/{entry['video_path']}"
    subtitles_json_path = f"/nas/mars/dataset/longvideobench/LongVideoBench/subtitles/{entry['subtitle_path']}"
    save_path = f"/nas/mars/dataset/longvideobench/burn-subtitles/{video_id}.mp4"

    # Skip if already processed
    if os.path.exists(save_path):
        return

    if not os.path.exists(video_path) or not os.path.exists(subtitles_json_path):
        with open("err.txt", "a") as error_log:
            if not os.path.exists(video_path):
                error_log.write(f"Video file not found: {video_path}\n")
            if not os.path.exists(subtitles_json_path):
                error_log.write(f"Subtitles file not found: {subtitles_json_path}\n")
        return

    try:
        starting_timestamp = entry["starting_timestamp_for_subtitles"]
        duration = entry["duration"]
    except KeyError as e:
        with open("err.txt", "a") as error_log:
            error_log.write(f"KeyError: {e} for video_id: {video_id}\n")
        return

    burn_subtitles_on_video(video_path, subtitles_json_path, starting_timestamp, save_path, duration)


def main():
    with open("/nas/mars/dataset/longvideobench/LongVideoBench/lvb_val.json", "r") as f:
        data = json.load(f)

    # Build a separate data stream with unique (video_id, subtitle_path)
    seen_pairs = set()
    unique_data = []
    for item in data:
        try:
            pair = (item["video_id"], item["subtitle_path"])
        except KeyError:
            continue
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        unique_data.append(item)

    print(f"Deduplicated entries: {len(data)} -> {len(unique_data)} unique video_id/subtitle_path pairs")

    os.makedirs("/nas/mars/dataset/longvideobench/burn-subtitles", exist_ok=True)

    num_workers = min(10, cpu_count())
    print(f"Using {num_workers} parallel workers.")
    with Pool(processes=num_workers) as pool:
        list(tqdm(pool.imap_unordered(process_entry, unique_data), total=len(unique_data)))


if __name__ == "__main__":
    main()

