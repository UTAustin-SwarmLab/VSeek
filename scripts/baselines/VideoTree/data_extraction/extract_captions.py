## extract captions from the video at 1 FPS and save them in a json file

import json
import os
import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import argparse
from transformers import AutoProcessor, MllamaForConditionalGeneration
import multiprocessing as mp
from multiprocessing import Queue, Process
import time

def load_json(fn):
    with open(fn, 'r') as f:
        data = json.load(f)
    return data

def save_json(data, fn, indent=4):
    with open(fn, 'w') as f:
        json.dump(data, f, indent=indent)

class LlamaVisionCaptioner:
    def __init__(self, model_name="meta-llama/Llama-3.2-11B-Vision-Instruct", device="cuda:0", batch_size=4):
        self.device = device
        self.batch_size = batch_size
        print(f"Loading LLaMA Vision model on {device}: {model_name}")
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        
    def caption_images_batch(self, image_paths, prompt="Describe what you see in this image succinctly."):
        """Generate captions for a batch of images"""
        try:
            batch_size = min(len(image_paths), self.batch_size)
            all_captions = []
            
            # Process images in batches
            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i:i + batch_size]
                batch_images = []
                
                # Load images for this batch
                for image_path in batch_paths:
                    try:
                        image = Image.open(image_path).convert('RGB')
                        batch_images.append(image)
                    except Exception as e:
                        print(f"Error loading image {image_path}: {e}")
                        all_captions.append(f"Error loading image: {str(e)}")
                        continue
                
                if not batch_images:
                    continue
                
                try:
                    # Create individual conversations for each image
                    conversations = []
                    # Create nested image list - each sub-list contains one image
                    nested_images = []
                    
                    for image in batch_images:
                        messages = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image"},
                                    {"type": "text", "text": prompt}
                                ]
                            }
                        ]
                        conversation = self.processor.apply_chat_template(messages, add_generation_prompt=True)
                        conversations.append(conversation)
                        # Each conversation gets its own sub-list with one image
                        nested_images.append([image])
                    
                    # Process batch - each conversation gets paired with its image sub-list
                    inputs = self.processor(
                        text=conversations,
                        images=nested_images,
                        return_tensors="pt",
                        padding=True
                    ).to(self.device)
                    
                    # Generate captions for batch
                    with torch.inference_mode():
                        output_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=256,
                            do_sample=True,
                            temperature=0.2,
                            top_p=0.9,
                            use_cache=True,
                            pad_token_id=self.processor.tokenizer.pad_token_id
                        )
                    
                    # Decode responses for each item in batch
                    for j, output in enumerate(output_ids):
                        if j < len(batch_images):
                            input_token_len = inputs['input_ids'][j].shape[0]
                            response = self.processor.decode(output[input_token_len:], skip_special_tokens=True)
                            
                            # Extract assistant response
                            assistant_start = response.find("<|start_header_id|>assistant<|end_header_id|>")
                            if assistant_start != -1:
                                response_start = assistant_start + len("<|start_header_id|>assistant<|end_header_id|>")
                                response = response[response_start:].strip()
                                
                                # Remove trailing tokens
                                eot_token = "<|eot_id|>"
                                if response.endswith(eot_token):
                                    response = response[:-len(eot_token)].strip()
                            
                            all_captions.append(response)
                
                except Exception as e:
                    print(f"Error in batch processing: {e}")
                    # Fallback to individual processing for this batch
                    for image_path in batch_paths:
                        try:
                            caption = self.caption_image_single(image_path, prompt)
                            all_captions.append(caption)
                        except Exception as e2:
                            print(f"Error in fallback processing for {image_path}: {e2}")
                            all_captions.append(f"Error: {str(e2)}")
            
            return all_captions
            
        except Exception as e:
            print(f"Error in batch captioning: {e}")
            return [f"Error: {str(e)}" for _ in image_paths]
    
    def caption_image_single(self, image_path, prompt="Describe what you see in this image succinctly."):
        """Generate caption for a single image (fallback method)"""
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Create messages for LLaMA Vision
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Process with model
            prompt_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(
                image, 
                prompt_text, 
                add_special_tokens=False, 
                return_tensors="pt"
            ).to(self.model.device)
            
            # Generate caption
            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.2,
                    top_p=0.9,
                    use_cache=True
                )
            
            # Decode response
            response = self.processor.decode(output_ids[0], skip_special_tokens=True)
            
            # Extract assistant response
            assistant_start = response.find("<|start_header_id|>assistant<|end_header_id|>")
            if assistant_start != -1:
                response_start = assistant_start + len("<|start_header_id|>assistant<|end_header_id|>")
                response = response[response_start:].strip()
                
                # Remove trailing tokens
                eot_token = "<|eot_id|>"
                if response.endswith(eot_token):
                    response = response[:-len(eot_token)].strip()
            
            return response
            
        except Exception as e:
            print(f"Error captioning {image_path}: {e}")
            return f"Error: {str(e)}"

def worker_process_batched(gpu_id, task_queue, result_queue, model_name, batch_size=4):
    """Worker process for GPU-based batch caption generation"""
    device = f"cuda:{gpu_id}"
    print(f"Worker {gpu_id} starting on device {device} with batch size {batch_size}")
    
    try:
        # Initialize captioner for this GPU
        captioner = LlamaVisionCaptioner(model_name, device, batch_size)
        print(f"Worker {gpu_id}: Model loaded successfully")
    except Exception as e:
        print(f"Worker {gpu_id}: Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    tasks_processed = 0
    
    while True:
        try:
            # Get batch of tasks from queue
            batch_tasks = []
            try:
                # Try to get a batch of tasks
                for _ in range(batch_size):
                    task = task_queue.get(timeout=10)
                    if task is None:  # Sentinel to stop worker
                        # Put sentinel back for other workers and break
                        task_queue.put(None)
                        break
                    batch_tasks.append(task)
            except:
                # If we can't get a full batch, process what we have
                pass
            
            if not batch_tasks:
                break
            
            # Extract image paths and metadata
            image_paths = [str(task[1]) for task in batch_tasks]
            
            try:
                # Generate captions for batch
                captions = captioner.caption_images_batch(image_paths)
                
                # Create results for each item in batch
                for i, (video_name, frame_file, frame_name) in enumerate(batch_tasks):
                    try:
                        caption = captions[i] if i < len(captions) else "Error: No caption generated"
                        
                        result = {
                            'video_name': video_name,
                            'frame_name': frame_name,
                            'frame_path': str(frame_file),
                            'caption': caption,
                            'timestamp': int(frame_name) if frame_name.isdigit() else frame_name
                        }
                        
                        # Put result in result queue
                        result_queue.put(result)
                        tasks_processed += 1
                        
                    except Exception as e:
                        print(f"Worker {gpu_id}: Error creating result for {frame_file}: {e}")
                        # Still put an error result to maintain count
                        error_result = {
                            'video_name': video_name,
                            'frame_name': frame_name,
                            'frame_path': str(frame_file),
                            'caption': f"Error creating result: {str(e)}",
                            'timestamp': int(frame_name) if frame_name.isdigit() else frame_name
                        }
                        result_queue.put(error_result)
                        tasks_processed += 1
                        
            except Exception as e:
                print(f"Worker {gpu_id}: Error in batch processing: {e}")
                import traceback
                traceback.print_exc()
                
                # Put error results for all tasks in this batch
                for video_name, frame_file, frame_name in batch_tasks:
                    error_result = {
                        'video_name': video_name,
                        'frame_name': frame_name,
                        'frame_path': str(frame_file),
                        'caption': f"",
                        'timestamp': int(frame_name) if frame_name.isdigit() else frame_name
                    }
                    result_queue.put(error_result)
                    tasks_processed += 1
                
        except Exception as e:
            print(f"Worker {gpu_id}: Unexpected error in main loop: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"Worker {gpu_id} finished. Processed {tasks_processed} tasks.")

def extract_captions_from_lvb_frames_multiprocessing(num_gpus=2, lvb_data_path=None, batch_size=4):
    """Extract captions from the existing LVB frames using LLaMA Vision with multiprocessing and batching"""
    
    # Paths
    frames_dir = Path('/nas/mars/experiment_result/nsvqa/8_videotree/lvb_frames')
    output_path = Path('/nas/mars/experiment_result/nsvqa/8_videotree/captions/lvb_captions.json')
    model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    
    # Default LVB data path
    if lvb_data_path is None:
        lvb_data_path = Path(__file__).parent.parent.parent / "data.json"
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not frames_dir.exists():
        print(f"Error: Frames directory {frames_dir} does not exist!")
        return
    
    if not Path(lvb_data_path).exists():
        print(f"Error: LVB data file {lvb_data_path} does not exist!")
        return
    
    # Load LVB data to get the video_ids that need captioning
    print(f"Loading LVB data from: {lvb_data_path}")
    with open(lvb_data_path, 'r') as f:
        lvb_data = json.load(f)
    
    # Extract unique video_ids from LVB data
    required_video_ids = set()
    for item in lvb_data:
        video_id = item.get('video_id')
        if video_id:
            required_video_ids.add(video_id)
    
    required_video_ids = set()      
    # TODO: need to merge this 
    for item in lvb_data:
        video_id = item.get('video_id')
        video_path = item.get('video_path')
        video_path_id = video_path.split('/')[-1].split('.')[0]
        if video_id != video_path_id:
            print(f"video_id: {video_id}, video_path_id: {video_path_id}")
            required_video_ids.add(video_path_id)
        
            
    
    print(f"Found {len(required_video_ids)} unique video_ids in LVB dataset")
    
    # Get all video directories that match LVB video_ids
    all_video_dirs = [d for d in frames_dir.iterdir() if d.is_dir()]
    video_dirs = [d for d in all_video_dirs if d.name in required_video_ids]
    
    print(f"Found {len(all_video_dirs)} total video directories")
    print(f"Found {len(video_dirs)} video directories matching LVB dataset")
    
    if len(video_dirs) == 0:
        print("No matching video directories found! Check that frame extraction was done for LVB videos.")
        return
    
    # Collect all tasks (video_name, frame_file, frame_name) for LVB videos only
    all_tasks = []
    total_frames = 0
    
    
    for video_dir in video_dirs:
        video_name = video_dir.name
        frame_files = sorted([f for f in video_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        
        for frame_file in frame_files:
            frame_name = frame_file.stem
            all_tasks.append((video_name, frame_file, frame_name))

    print(f"Total frames to process for LVB videos: {total_frames}")
    print(f"Using batch size: {batch_size}")
    
    # Create queues
    task_queue = Queue()
    result_queue = Queue()
    
    # Add all tasks to queue
    for task in all_tasks:
        task_queue.put(task)
    
    # Start worker processes with batching
    workers = []
    for gpu_id in range(num_gpus):
        worker = Process(target=worker_process_batched, args=(gpu_id, task_queue, result_queue, model_name, batch_size))
        worker.start()
        workers.append(worker)
    
    print(f"Started {num_gpus} worker processes with batch size {batch_size}")
    
    # Collect results
    all_captions = {}
    processed_count = 0
    
    with tqdm(total=total_frames, desc="Processing LVB frames (batched)") as pbar:
        while processed_count < total_frames:
            try:
                result = result_queue.get(timeout=120)  # Increased timeout for batching
                
                video_name = result['video_name']
                frame_name = result['frame_name']
                
                # Initialize video dict if needed
                if video_name not in all_captions:
                    all_captions[video_name] = {}
                
                # Store result
                all_captions[video_name][frame_name] = {
                    "frame_path": result['frame_path'],
                    "caption": result['caption'],
                    "timestamp": result['timestamp']
                }
                
                processed_count += 1
                pbar.update(1)
                
                # Save intermediate results every 500 frames (adjusted for batching)
                if processed_count % 500 == 0:
                    intermediate_path = str(output_path).replace('.json', f'_intermediate_{processed_count}.json')
                    save_json(all_captions, intermediate_path)
                
            except Exception as e:
                print(f"Error collecting result: {e}")
                print(f"Exception type: {type(e).__name__}")
                import traceback
                traceback.print_exc()
                
                # Check if any workers are still alive
                alive_workers = [w for w in workers if w.is_alive()]
                print(f"Workers still alive: {len(alive_workers)}/{len(workers)}")
                
                # Check queue sizes
                print(f"Task queue size: {task_queue.qsize()}")
                print(f"Result queue size: {result_queue.qsize()}")
                
                # If no workers are alive and no results in queue, break
                if len(alive_workers) == 0 and result_queue.empty():
                    print("No workers alive and no results in queue. Stopping collection.")
                    break
                    
                # If it's a timeout, continue trying
                if "timeout" in str(e).lower() or "empty" in str(e).lower():
                    print("Timeout or empty queue, continuing...")
                    continue
                else:
                    # For other errors, break after a few attempts
                    break
    
    # Stop workers
    for _ in range(num_gpus):
        task_queue.put(None)
    
    # Wait for workers to finish
    for worker in workers:
        worker.join()
    
    # Save final results
    save_json(all_captions, str(output_path))
    print(f"\nAll captions saved to: {output_path}")
    print(f"Processed {processed_count} frames across {len(all_captions)} LVB videos")
    
    return all_captions

def extract_captions_from_lvb_frames(lvb_data_path):
    """Single-threaded version for backward compatibility"""
    
    # Paths
    frames_dir = Path('/nas/mars/experiment_result/nsvqa/8_videotree/lvb_frames')
    output_path = Path('/nas/mars/experiment_result/nsvqa/8_videotree/captions2/lvb_captions.json')
    model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    
    # Default LVB data path
    if lvb_data_path is None:
        lvb_data_path = Path(__file__).parent.parent.parent / "data.json"
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not frames_dir.exists():
        print(f"Error: Frames directory {frames_dir} does not exist!")
        return
    
    if not Path(lvb_data_path).exists():
        print(f"Error: LVB data file {lvb_data_path} does not exist!")
        return
    
    # Load LVB data to get the video_ids that need captioning
    print(f"Loading LVB data from: {lvb_data_path}")
    with open(lvb_data_path, 'r') as f:
        lvb_data = json.load(f)
    
    # Extract unique video_ids from LVB data
    required_video_ids = set()
    for item in lvb_data:
        video_id = item.get('video_id')
        if video_id:
            required_video_ids.add(video_id)
    
    required_video_ids = set()      
    # TODO: need to merge this 
    for item in lvb_data:
        video_id = item.get('video_id')
        video_path = item.get('video_path')
        video_path_id = video_path.split('/')[-1].split('.')[0]
        if video_id != video_path_id:
            print(f"video_id: {video_id}, video_path_id: {video_path_id}")
            required_video_ids.add(video_path_id)
        
            
    
    if not frames_dir.exists():
        print(f"Error: Frames directory {frames_dir} does not exist!")
        return
    
    # Initialize LLaMA Vision captioner
    captioner = LlamaVisionCaptioner()
    
    # Get all video directories
    print(f"Found {len(required_video_ids)} unique video_ids in LVB dataset")
    
    # Get all video directories that match LVB video_ids
    all_video_dirs = [d for d in frames_dir.iterdir() if d.is_dir()]
    video_dirs = [d for d in all_video_dirs if d.name in required_video_ids]
    
    print(f"Found {len(all_video_dirs)} total video directories")
    print(f"Found {len(video_dirs)} video directories")
    
    all_captions = {}
    
    for video_dir in tqdm(video_dirs, desc="Processing videos"):
        video_name = video_dir.name
        print(f"\nProcessing video: {video_name}")
        
        # Get all frame images
        frame_files = sorted([f for f in video_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        
        if not frame_files:
            print(f"No frame files found in {video_dir}")
            continue
        
        video_captions = {}
        
        for frame_file in tqdm(frame_files, desc=f"Processing frames for {video_name}", leave=False):
            frame_name = frame_file.stem
            
            # Generate caption
            caption = captioner.caption_image_single(str(frame_file))
            
            video_captions[frame_name] = {
                "frame_path": str(frame_file),
                "caption": caption,
                "timestamp": int(frame_name) if frame_name.isdigit() else frame_name
            }
        
        all_captions[video_name] = video_captions
        
        # Save intermediate results
        intermediate_path = str(output_path).replace('.json', f'_{video_name}.json')
        save_json(all_captions, intermediate_path)
        print(f"Saved intermediate results for {video_name}")
    
    # Save final results
    save_json(all_captions, str(output_path))
    print(f"\nAll captions saved to: {output_path}")
    
    return all_captions

def extract_captions(video_path):
    """Legacy function - extracts frames from video and then captions them"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = int(fps)
    
    # Extract frames at 1 FPS
    frames = []
    count = 0
    success = True
    
    while success:
        success, image = cap.read()
        if not success:
            break
        if count % frame_interval == 0:
            frames.append(image)
        count += 1
    
    cap.release()
    
    # Initialize captioner
    captioner = LlamaVisionCaptioner()
    captions = []
    
    for i, frame in enumerate(frames):
        # Convert to PIL Image
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Save temporary frame
        temp_path = f"temp_frame_{i}.jpg"
        frame_pil.save(temp_path)
        
        # Generate caption
        caption = captioner.caption_image_single(temp_path)
        captions.append({
            "frame_index": i,
            "timestamp": i / fps,
            "caption": caption
        })
        
        # Clean up
        os.remove(temp_path)
    
    return captions

if __name__ == "__main__":
    # Set multiprocessing start method to 'spawn' for CUDA compatibility
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description="Extract captions from LVB frames using LLaMA Vision")
    parser.add_argument("--output", type=str, default="/nas/mars/experiment_result/nsvqa/8_videotree/captions2/lvb_captions.json", help="Output JSON file")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-11B-Vision-Instruct", help="LLaMA Vision model")
    parser.add_argument("--num_gpus", type=int, default=2, help="Number of GPUs to use")
    parser.add_argument("--single_gpu", action="store_true", help="Use single GPU mode")
    parser.add_argument("--lvb_data_path", type=str, default="/nas/mars/experiment_result/nsvqa/8_videotree/lvb_val.json", help="Path to LVB data.json file (default: ../../../data.json)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for multiprocessing")
    
    args = parser.parse_args()
    
    print("Starting LLaMA Vision caption extraction for LVB frames...")
    
    if args.single_gpu:
        print("Using single GPU mode")
        captions = extract_captions_from_lvb_frames(args.lvb_data_path)
    else:
        print(f"Using multiprocessing mode with {args.num_gpus} GPUs and batch size {args.batch_size}")
        captions = extract_captions_from_lvb_frames_multiprocessing(args.num_gpus, args.lvb_data_path, args.batch_size)
    
    print(f"Completed! Processed {len(captions)} videos.")