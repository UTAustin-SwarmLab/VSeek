import json
import os
import subprocess
import argparse
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
       
def burn_subtitles_on_video(video_path, subtitles_path, starting_timestamp, save_path, video_duration,
                             font_size=24, font="Arial-Bold", color="white"):
    
    
    if 'json' in subtitles_path:
        mode = 'json'
    else:
        mode = 'srt'
    
    if mode == 'json':
        with open(subtitles_path, "r") as f:
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

    elif mode == 'srt':
        temp_srt_path = subtitles_path

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


def process_entry(entry, video_base_dir, subtitle_base_dir, output_dir):
    video_id = entry["video_id"]
    video_path = os.path.join(video_base_dir, entry['video_path'])
    subtitles_json_path = os.path.join(subtitle_base_dir, entry['subtitle_path'])
    save_path = os.path.join(output_dir, f"{video_id}.mp4")
    
    # Skip if already processed
    if os.path.exists(save_path):
        return

    if not os.path.exists(video_path) or not os.path.exists(subtitles_json_path):
        with open("err.txt", "a") as error_log:
            if not os.path.exists(video_path):
                print(f"Video file not found: {video_path}")
                error_log.write(f"Video file not found: {video_path}\n")
            if not os.path.exists(subtitles_json_path):
                print(f"Subtitles file not found: {subtitles_json_path}")
                error_log.write(f"Subtitles file not found: {subtitles_json_path}\n")
        shutil.copy(video_path, save_path)
        print(f"Copied original video to {save_path}")
        return

    try:
        starting_timestamp = entry.get("starting_timestamp_for_subtitles",0)
        duration = entry.get("duration", 0)
    except KeyError as e:
        with open("err.txt", "a") as error_log:
            error_log.write(f"KeyError: {e} for video_id: {video_id}\n")
        print(f"KeyError: {e} for video_id: {video_id}")
        return

    burn_subtitles_on_video(video_path, subtitles_json_path, starting_timestamp, save_path, duration)


def main():
    parser = argparse.ArgumentParser(description="Burn subtitles onto videos")
    
    # Mode selection
    parser.add_argument("--mode", type=str, choices=["single", "batch"], default="batch",
                        help="Processing mode: 'single' for one video or 'batch' for JSON file")
    
    # Single video mode arguments
    parser.add_argument("--video", type=str, help="Path to input video file (single mode)")
    parser.add_argument("--subtitle", type=str, help="Path to subtitle JSON file (single mode)")
    parser.add_argument("--output", type=str, help="Path to output video file (single mode)")
    parser.add_argument("--starting-timestamp", type=float, default=0.0,
                        help="Starting timestamp for subtitles in seconds (single mode, default: 0.0)")
    parser.add_argument("--duration", type=float, help="Video duration in seconds (single mode, auto-detected if not provided)")
    
    # Batch mode arguments
    parser.add_argument("--data-folder", type=str, 
                        help="Path to data folder containing JSON file, videos/, and subtitles/ subdirectories (batch mode)")
    parser.add_argument("--json-file", type=str, default="lvb_val.json",
                        help="Name of JSON file in data folder (batch mode, default: lvb_val.json)")
    parser.add_argument("--output-dir", type=str, help="Output directory for processed videos (batch mode)")
    
    # Common arguments
    parser.add_argument("--workers", type=int, default=min(10, cpu_count()),
                        help="Number of parallel workers (batch mode only, default: 10 or CPU count)")

    
    args = parser.parse_args()
    
    if args.mode == "single":
        # Single video processing
        if not all([args.video, args.subtitle, args.output]):
            parser.error("Single mode requires --video, --subtitle, and --output arguments")
        
        if not os.path.exists(args.video):
            print(f"Error: Video file not found: {args.video}")
            return
        
        if not os.path.exists(args.subtitle):
            print(f"Error: Subtitle file not found: {args.subtitle}")
            return
        
        # Auto-detect duration if not provided
        duration = args.duration
        if duration is None:
            try:
                probe = subprocess.run(
                    ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                     "-of", "default=noprint_wrappers=1:nokey=1", args.video],
                    capture_output=True, text=True, check=True
                )
                duration = float(probe.stdout.strip())
                print(f"Auto-detected video duration: {duration:.2f} seconds")
            except Exception as e:
                print(f"Error detecting video duration: {e}")
                print("Please provide --duration manually")
                return
        
        print(f"Processing video: {args.video}")
        burn_subtitles_on_video(args.video, args.subtitle, args.starting_timestamp, args.output, duration)
        print(f"Output saved to: {args.output}")
        
    else:  # batch mode
        if not all([args.data_folder, args.output_dir]):
            parser.error("Batch mode requires --data-folder and --output-dir arguments")
        
        # Construct paths relative to data folder
        json_path = os.path.join(args.data_folder, args.json_file)
        video_base_dir = os.path.join(args.data_folder, "videos")
        subtitle_base_dir = os.path.join(args.data_folder, "subtitles")
        
        if not os.path.exists(json_path):
            print(f"Error: JSON file not found: {json_path}")
            return
        
        if not os.path.exists(video_base_dir):
            print(f"Warning: Video directory not found: {video_base_dir}")
        
        if not os.path.exists(subtitle_base_dir):
            print(f"Warning: Subtitle directory not found: {subtitle_base_dir}")
        
        with open(json_path, "r") as f:
            data = json.load(f)
        
        # Build a separate data stream with unique (video_id, subtitle_path)
        seen_pairs = set()
        unique_data = []
        for j,item in enumerate(data):
            try:
                if 'videomme' in json_path:
                    pair = (item["videoID"], item["videoID"])
                    data[j]["video_id"] = item["videoID"]
                    data[j]["video_path"] = "data/"+item["videoID"]+".mp4"
                    data[j]["subtitle_path"] = "subtitle/"+item["videoID"]+".srt"
                elif 'lvb' in json_path:
                    pair = (item["video_id"], item["subtitle_path"])
            except KeyError:
                continue
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            unique_data.append(item)
        
        os.makedirs(args.output_dir, exist_ok=True)
        
        num_workers = min(args.workers, cpu_count())
        print(f"Using {num_workers} parallel workers.")
        print(f"Processing {len(unique_data)} videos...")
        print("")
        
        # Create arguments list with all parameters
        process_args = [(entry, video_base_dir, subtitle_base_dir, args.output_dir) 
                       for entry in unique_data]
        
        with Pool(processes=num_workers) as pool:
            list(tqdm(
                pool.starmap(process_entry, process_args), 
                total=len(unique_data),
                desc="Burning subtitles",
                unit="video",
                ncols=100
            ))
        
        print("")
        print(f"Batch processing complete!")
        print(f"Processed {len(unique_data)} videos")
        print(f"Output saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

