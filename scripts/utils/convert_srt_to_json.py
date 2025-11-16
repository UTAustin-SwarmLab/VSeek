"""
Utility script to convert SRT subtitle files to JSON format.
Compatible with VSeek's subtitle processing pipeline.
"""

import json
import argparse
import os
from pathlib import Path
from tqdm import tqdm


def srt_time_to_seconds(time_str):
    """
    Convert SRT timestamp to seconds.
    
    Args:
        time_str: Time string in format "HH:MM:SS,mmm" or "HH:MM:SS.mmm"
    
    Returns:
        Float seconds
    """
    time_str = time_str.strip().replace(',', '.')
    parts = time_str.split(':')
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds


def parse_srt_to_json(srt_path):
    """
    Parse SRT subtitle file and convert to JSON format.
    
    Args:
        srt_path: Path to SRT file
    
    Returns:
        List of dictionaries with keys: timestamp, text
    """
    subtitles = []
    
    try:
        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by double newlines to get individual subtitle blocks
        blocks = content.strip().split('\n\n')
        
        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) < 3:
                continue
            
            try:
                # Parse timestamp line: "00:00:01,000 --> 00:00:04,000"
                timestamp_line = lines[1]
                if '-->' not in timestamp_line:
                    continue
                
                start_str, end_str = timestamp_line.split('-->')
                start = srt_time_to_seconds(start_str)
                end = srt_time_to_seconds(end_str)
                
                # Join all remaining lines as text
                text = '\n'.join(lines[2:]).strip()
                
                subtitles.append({
                    'timestamp': [start, end],
                    'text': text
                })
            except (ValueError, IndexError) as e:
                print(f"Warning: Failed to parse subtitle block in {srt_path}: {e}")
                continue
    
    except Exception as e:
        print(f"Error reading SRT file {srt_path}: {e}")
        return []
    
    return subtitles


def convert_single_file(srt_path, json_path):
    """Convert a single SRT file to JSON."""
    subtitles = parse_srt_to_json(srt_path)
    
    if not subtitles:
        print(f"Warning: No subtitles found in {srt_path}")
        return False
    
    os.makedirs(os.path.dirname(json_path) or '.', exist_ok=True)
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(subtitles, f, indent=2, ensure_ascii=False)
    
    return True


def convert_directory(input_dir, output_dir):
    """Convert all SRT files in a directory to JSON."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Find all SRT files
    srt_files = list(input_path.rglob('*.srt'))
    
    if not srt_files:
        print(f"No SRT files found in {input_dir}")
        return
    
    print(f"Found {len(srt_files)} SRT files")
    os.makedirs(output_dir, exist_ok=True)
    
    success_count = 0
    for srt_file in tqdm(srt_files, desc="Converting SRT files"):
        # Create output path with same relative structure
        relative_path = srt_file.relative_to(input_path)
        json_file = output_path / relative_path.with_suffix('.json')
        
        if convert_single_file(str(srt_file), str(json_file)):
            success_count += 1
    
    print(f"\nSuccessfully converted {success_count}/{len(srt_files)} files")
    print(f"Output saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert SRT subtitle files to JSON format"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input SRT file or directory containing SRT files"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSON file or directory for JSON files"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single file conversion
        print(f"Converting {args.input} to {args.output}")
        if convert_single_file(args.input, args.output):
            print("Conversion successful!")
            
            # Show sample
            with open(args.output, 'r') as f:
                data = json.load(f)
            print(f"\nConverted {len(data)} subtitle entries")
            if data:
                print(f"\nSample entry:")
                print(json.dumps(data[0], indent=2))
        else:
            print("Conversion failed!")
            
    elif input_path.is_dir():
        # Directory conversion
        convert_directory(args.input, args.output)
    else:
        print(f"Error: {args.input} is not a valid file or directory")


if __name__ == "__main__":
    main()

