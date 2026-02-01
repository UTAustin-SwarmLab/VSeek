"""
VSeek Visualization Application

A Gradio-based interface for interactive video question answering with visual search.
This application demonstrates the agentic video search mechanism by:
1. Loading and indexing videos with ViClip embeddings
2. Processing questions through a VLM that generates search queries
3. Visualizing the retrieved video scenes
4. Displaying the thought trace, search queries, and answers
"""

import sys
import re
import base64
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
import gradio as gr
from PIL import Image
import requests

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vseek.data.frame import VideoFrames
from vseek.video_embedding.video_clip import ViClip


@dataclass
class AgentTrace:
    """Stores the trace of an agent's execution."""
    thinking: List[str] = field(default_factory=list)
    search_queries: List[Dict[str, str]] = field(default_factory=list)
    retrieved_frames: List[Dict[str, Any]] = field(default_factory=list)
    answer: Optional[str] = None
    raw_response: str = ""
    
    def add_thinking(self, text: str):
        self.thinking.append(text)
    
    def add_search_query(self, query: str, mode: str, frame_indices: List[int]):
        self.search_queries.append({
            "query": query,
            "mode": mode,
            "frame_indices": frame_indices
        })
    
    def format_trace(self) -> str:
        """Format the trace for display."""
        lines = []
        
        # Thinking section
        if self.thinking:
            lines.append("=" * 50)
            lines.append("🧠 THINKING TRACE")
            lines.append("=" * 50)
            for i, thought in enumerate(self.thinking, 1):
                lines.append(f"\n[Step {i}]")
                lines.append(thought.strip())
        
        # Search queries section
        if self.search_queries:
            lines.append("\n" + "=" * 50)
            lines.append("🔍 SEARCH QUERIES")
            lines.append("=" * 50)
            for i, sq in enumerate(self.search_queries, 1):
                lines.append(f"\n[Query {i}]")
                lines.append(f"  Mode: {sq['mode']}")
                lines.append(f"  Query: {sq['query']}")
                lines.append(f"  Retrieved Frames: {sq['frame_indices']}")
        
        # Answer section
        if self.answer:
            lines.append("\n" + "=" * 50)
            lines.append("✅ FINAL ANSWER")
            lines.append("=" * 50)
            lines.append(f"\n{self.answer}")
        
        return "\n".join(lines)


class VideoIndexer:
    """Handles video loading and indexing with ViClip embeddings."""
    
    def __init__(
        self, 
        viclip_model_path: str,
        viclip_tokenizer_path: str,
        gpu_number: int = 0,
        window_size: int = 8
    ):
        self.window_size = window_size
        self.gpu_number = gpu_number
        
        # Initialize ViClip model
        print(f"Loading ViClip model from {viclip_model_path}...")
        self.viclip = ViClip(
            pretrained_model_path=viclip_model_path,
            tokenizer_path=viclip_tokenizer_path,
            gpu_number=gpu_number
        )
        print("ViClip model loaded successfully!")
        
        # Cache for indexed videos
        self.video_cache: Dict[str, VideoFrames] = {}
        self.embeddings_cache: Dict[str, Dict[int, torch.Tensor]] = {}
    
    def extract_frames(self, video_path: str, fps: float = 1.0) -> List[np.ndarray]:
        """Extract frames from video at specified FPS."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(video_fps / fps) if fps < video_fps else 1
        
        frames = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_interval == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            
            frame_idx += 1
        
        cap.release()
        print(f"Extracted {len(frames)} frames from video")
        return frames
    
    def compute_embeddings(self, frames: List[np.ndarray]) -> Dict[int, torch.Tensor]:
        """Compute ViClip embeddings for video frames grouped by windows."""
        embeddings = {}
        num_windows = (len(frames) + self.window_size - 1) // self.window_size
        
        print(f"Computing embeddings for {num_windows} windows...")
        
        for window_idx in range(num_windows):
            start_idx = window_idx * self.window_size
            end_idx = min(start_idx + self.window_size, len(frames))
            window_frames = frames[start_idx:end_idx]
            
            # Use the middle frame of the window for embedding
            middle_idx = len(window_frames) // 2
            frame = window_frames[middle_idx]
            
            # Get embedding from ViClip - expects (N, H, W, C) array
            # Stack all window frames and get the feature
            frames_array = np.stack(window_frames, axis=0)  # (N, H, W, C)
            embedding = self.viclip.get_feature(frames_array)
            embeddings[window_idx] = embedding.squeeze().cpu()
        
        print(f"Computed {len(embeddings)} window embeddings")
        return embeddings
    
    def index_video(self, video_path: str) -> Tuple[VideoFrames, str]:
        """Index a video and return VideoFrames object with embeddings."""
        video_id = Path(video_path).stem
        
        # Check cache
        if video_id in self.video_cache:
            print(f"Using cached index for video: {video_id}")
            return self.video_cache[video_id], video_id
        
        print(f"Indexing video: {video_path}")
        
        # Extract frames
        frames = self.extract_frames(video_path, fps=1.0)
        
        if len(frames) == 0:
            raise ValueError("No frames extracted from video")
        
        # Create VideoFrames object
        video_frames = VideoFrames(window_size=self.window_size)
        video_frames.add_all_frames(frames)
        video_frames.partition_frames()
        
        # Compute embeddings
        embeddings = self.compute_embeddings(frames)
        for idx, emb in embeddings.items():
            video_frames.add_embedding(idx, emb)
        
        # Cache
        self.video_cache[video_id] = video_frames
        self.embeddings_cache[video_id] = embeddings
        
        print(f"Video indexed successfully: {len(frames)} frames, {len(embeddings)} windows")
        return video_frames, video_id
    
    def search_frames(
        self, 
        video_id: str, 
        query: str, 
        topk: int = 4,
        mode: str = "base"
    ) -> Tuple[List[np.ndarray], List[int]]:
        """Search for relevant frames using text query."""
        if video_id not in self.video_cache:
            raise ValueError(f"Video {video_id} not indexed")
        
        video_frames = self.video_cache[video_id]
        embeddings = self.embeddings_cache[video_id]
        
        # Get text embedding
        text_embedding = self.viclip.get_text_embedding(query)
        if text_embedding.dim() > 1:
            text_embedding = text_embedding.squeeze()
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
        
        # Compute similarities
        device = text_embedding.device
        similarities = []
        window_indices = list(embeddings.keys())
        
        for idx in window_indices:
            emb = embeddings[idx].to(device)
            if emb.dim() > 1:
                emb = emb.squeeze()
            emb = emb / emb.norm(dim=-1, keepdim=True)
            sim = torch.dot(text_embedding, emb).item()
            similarities.append((idx, sim))
        
        # Sort by similarity and get top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in similarities[:topk]]
        
        # Get frames for top windows
        retrieved_frames = []
        for idx in sorted(top_indices):
            window_frames = video_frames.get_frame_chunk(idx)
            retrieved_frames.extend(window_frames)
        
        return retrieved_frames, sorted(top_indices)


class AgentRunner:
    """Runs the VLM agent to generate search queries and answers."""
    
    def __init__(
        self, 
        llm_server_url: str = "http://localhost:8001/v1",
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        api_key: str = "EMPTY"
    ):
        self.llm_server_url = llm_server_url
        self.model_name = model_name
        self.api_key = api_key
        
        # Tag patterns for parsing search queries
        self.search_patterns = [
            re.compile(r"<search>(.*?)</search>", re.DOTALL),
            re.compile(r"<search_subtitle>(.*?)</search_subtitle>", re.DOTALL),
            re.compile(r"<search_summary>(.*?)</search_summary>", re.DOTALL),
        ]
    
    def encode_frame_base64(self, frame: np.ndarray, max_size: int = 512) -> str:
        """Encode a numpy frame to base64 JPEG."""
        h, w = frame.shape[:2]
        scale = min(max_size / w, max_size / h, 1.0)
        if scale < 1.0:
            new_w, new_h = int(w * scale), int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Convert RGB to BGR for cv2
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buffer).decode('utf-8')
    
    def parse_search_queries(self, text: str) -> List[Dict[str, str]]:
        """Parse search queries from model output."""
        queries = []
        
        # Remove thinking tags for parsing
        clean_text = text.split("</think>")[-1] if "</think>" in text else text
        
        for pattern in self.search_patterns:
            for match in pattern.finditer(clean_text):
                query_text = match.group(1).strip()
                tag_str = pattern.pattern
                
                if "<search_subtitle>" in tag_str:
                    mode = "subtitle"
                elif "<search_summary>" in tag_str:
                    mode = "summary"
                else:
                    mode = "base"
                
                queries.append({"query": query_text, "mode": mode})
        
        return queries
    
    def extract_thinking(self, text: str) -> str:
        """Extract thinking content from model output."""
        think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        if think_match:
            return think_match.group(1).strip()
        return ""
    
    def extract_answer(self, text: str) -> str:
        """Extract the final answer from model output."""
        # Remove thinking and search tags
        clean_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        clean_text = re.sub(r"<search>.*?</search>", "", clean_text, flags=re.DOTALL)
        clean_text = re.sub(r"<search_subtitle>.*?</search_subtitle>", "", clean_text, flags=re.DOTALL)
        clean_text = re.sub(r"<search_summary>.*?</search_summary>", "", clean_text, flags=re.DOTALL)
        return clean_text.strip()
    
    def build_system_prompt(self) -> str:
        """Build the system prompt for the agent."""
        return """You are an intelligent video question answering agent. You can search through video frames to find relevant visual information.

You have access to the following search tools:
- <search>query</search> - Search for video frames matching the natural language query
- <search_subtitle>query</search_subtitle> - Search for frames by matching subtitle text
- <search_summary>summary</search_summary> - Get a summary view of the entire video

When answering questions about a video:
1. First, think about what visual information you need to find
2. Use the search tools to retrieve relevant frames
3. Analyze the retrieved frames to answer the question
4. Provide a clear, concise answer

Use <think>...</think> tags to show your reasoning process.
"""
    
    def check_llm_available(self) -> bool:
        """Check if the LLM server is available."""
        try:
            response = requests.get(
                f"{self.llm_server_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=5
            )
            return response.status_code == 200
        except Exception:
            return False
    
    def call_llm(
        self, 
        messages: List[Dict[str, Any]], 
        images: Optional[List[str]] = None
    ) -> str:
        """Call the LLM API."""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Build content with images if provided
            if images:
                content = []
                for img_b64 in images:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                    })
                # Add text from the last user message
                last_msg = messages[-1]
                if isinstance(last_msg.get("content"), str):
                    content.append({"type": "text", "text": last_msg["content"]})
                messages[-1]["content"] = content
            
            payload = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": 1024,
                "temperature": 0.7,
            }
            
            response = requests.post(
                f"{self.llm_server_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
        
        except requests.exceptions.ConnectionError:
            return "LLM_SERVER_UNAVAILABLE"
        except Exception as e:
            return f"Error calling LLM: {str(e)}"
    
    def run_agent(
        self,
        question: str,
        indexer: VideoIndexer,
        video_id: str,
        max_turns: int = 3
    ) -> Tuple[AgentTrace, List[np.ndarray]]:
        """Run the agent to answer a question about a video."""
        trace = AgentTrace()
        all_retrieved_frames = []
        
        # Check if LLM is available
        if not self.check_llm_available():
            trace.add_thinking("LLM server not available. Falling back to direct search mode.")
            # Perform a direct search using the question
            frames, frame_indices = indexer.search_frames(video_id, question, topk=4)
            trace.add_search_query(question, "base", frame_indices)
            trace.answer = f"[LLM Unavailable] Retrieved {len(frames)} frames matching: '{question}'"
            return trace, frames
        
        # Build initial messages
        messages = [
            {"role": "system", "content": self.build_system_prompt()},
            {"role": "user", "content": question}
        ]
        
        for turn in range(max_turns):
            print(f"Agent turn {turn + 1}/{max_turns}")
            
            # Prepare images for this turn
            images = None
            if all_retrieved_frames:
                # Encode frames for the model
                images = [self.encode_frame_base64(f) for f in all_retrieved_frames[-16:]]
            
            # Call LLM
            response = self.call_llm(messages, images)
            
            # Handle LLM server unavailable
            if response == "LLM_SERVER_UNAVAILABLE":
                trace.add_thinking("LLM server became unavailable during conversation.")
                if not all_retrieved_frames:
                    frames, frame_indices = indexer.search_frames(video_id, question, topk=4)
                    trace.add_search_query(question, "base", frame_indices)
                    all_retrieved_frames.extend(frames)
                trace.answer = f"[LLM Unavailable] Retrieved {len(all_retrieved_frames)} frames."
                return trace, all_retrieved_frames
            
            trace.raw_response = response
            
            # Extract thinking
            thinking = self.extract_thinking(response)
            if thinking:
                trace.add_thinking(thinking)
            
            # Parse search queries
            queries = self.parse_search_queries(response)
            
            if queries:
                # Execute searches
                for q in queries:
                    print(f"Executing search: mode={q['mode']}, query={q['query']}")
                    
                    if q['mode'] == 'summary':
                        # Get uniform sample of frames
                        video_frames = indexer.video_cache[video_id]
                        frames = video_frames.uniformly_sample_frames(8)
                        total_frames = len(video_frames.all_frames)
                        if total_frames > 8:
                            frame_indices = list(range(0, total_frames, total_frames // 8))[:8]
                        else:
                            frame_indices = list(range(total_frames))
                    else:
                        frames, frame_indices = indexer.search_frames(
                            video_id, 
                            q['query'], 
                            topk=4,
                            mode=q['mode']
                        )
                    
                    trace.add_search_query(q['query'], q['mode'], frame_indices)
                    all_retrieved_frames.extend(frames)
                
                # Add tool response to messages
                messages.append({"role": "assistant", "content": response})
                tool_response = f"Retrieved {len(all_retrieved_frames)} frames. Please analyze them and provide your answer."
                messages.append({"role": "user", "content": tool_response})
            else:
                # No more searches, extract final answer
                answer = self.extract_answer(response)
                trace.answer = answer if answer else response
                break
        
        # If we exhausted turns without getting an answer
        if trace.answer is None:
            trace.answer = self.extract_answer(trace.raw_response)
        
        return trace, all_retrieved_frames


class VSeekVisualizationApp:
    """Main Gradio application for VSeek visualization."""
    
    def __init__(
        self,
        viclip_model_path: str,
        viclip_tokenizer_path: str,
        llm_server_url: str = "http://localhost:8001/v1",
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        gpu_number: int = 0,
        window_size: int = 8
    ):
        self.indexer = VideoIndexer(
            viclip_model_path=viclip_model_path,
            viclip_tokenizer_path=viclip_tokenizer_path,
            gpu_number=gpu_number,
            window_size=window_size
        )
        
        self.agent = AgentRunner(
            llm_server_url=llm_server_url,
            model_name=model_name
        )
        
        self.current_video_id: Optional[str] = None
        self.current_video_path: Optional[str] = None
    
    def load_video(self, video_file) -> Tuple[str, str, List[Image.Image]]:
        """Load and index a video file."""
        if video_file is None:
            return "No video uploaded", "", []
        
        try:
            video_path = video_file.name if hasattr(video_file, 'name') else video_file
            
            # Index the video
            video_frames, video_id = self.indexer.index_video(video_path)
            self.current_video_id = video_id
            self.current_video_path = video_path
            
            # Get sample frames for preview
            sample_frames = video_frames.uniformly_sample_frames(8)
            preview_images = [Image.fromarray(f) for f in sample_frames]
            
            status = f"✅ Video loaded and indexed successfully!\n"
            status += f"  - Video ID: {video_id}\n"
            status += f"  - Total frames: {len(video_frames.all_frames)}\n"
            status += f"  - Windows: {len(video_frames.frames_by_window)}\n"
            status += f"  - Window size: {self.indexer.window_size}"
            
            return status, video_path, preview_images
        
        except Exception as e:
            return f"❌ Error loading video: {str(e)}", "", []
    
    def run_query(
        self, 
        question: str, 
        use_llm: bool = True
    ) -> Tuple[str, List[Image.Image], str]:
        """Run a query against the indexed video."""
        if self.current_video_id is None:
            return "❌ Please load a video first", [], ""
        
        if not question.strip():
            return "❌ Please enter a question", [], ""
        
        try:
            if use_llm:
                # Run full agent with LLM
                trace, retrieved_frames = self.agent.run_agent(
                    question=question,
                    indexer=self.indexer,
                    video_id=self.current_video_id,
                    max_turns=3
                )
                
                trace_text = trace.format_trace()
                
            else:
                # Direct search without LLM (for testing)
                trace = AgentTrace()
                retrieved_frames, frame_indices = self.indexer.search_frames(
                    self.current_video_id,
                    question,
                    topk=4
                )
                trace.add_search_query(question, "base", frame_indices)
                trace.answer = f"Retrieved {len(retrieved_frames)} frames for query: {question}"
                trace_text = trace.format_trace()
            
            # Convert frames to PIL images
            result_images = [Image.fromarray(f) for f in retrieved_frames[:16]]
            
            status = f"✅ Query completed\n"
            status += f"  - Retrieved {len(retrieved_frames)} frames\n"
            status += f"  - Search queries: {len(trace.search_queries)}"
            
            return status, result_images, trace_text
        
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            return f"❌ Error: {str(e)}\n\n{error_trace}", [], ""
    
    def manual_search(self, query: str, mode: str, topk: int) -> Tuple[str, List[Image.Image]]:
        """Perform a manual search without the LLM agent."""
        if self.current_video_id is None:
            return "❌ Please load a video first", []
        
        if not query.strip():
            return "❌ Please enter a search query", []
        
        try:
            frames, frame_indices = self.indexer.search_frames(
                self.current_video_id,
                query,
                topk=topk,
                mode=mode
            )
            
            result_images = [Image.fromarray(f) for f in frames]
            
            status = f"✅ Manual search completed\n"
            status += f"  - Query: {query}\n"
            status += f"  - Mode: {mode}\n"
            status += f"  - Retrieved windows: {frame_indices}\n"
            status += f"  - Total frames: {len(frames)}"
            
            return status, result_images
        
        except Exception as e:
            return f"❌ Error: {str(e)}", []
    
    def build_interface(self) -> gr.Blocks:
        """Build the Gradio interface."""
        
        with gr.Blocks(
            title="VSeek Video Search Visualization",
            theme=gr.themes.Soft()
        ) as demo:
            
            gr.Markdown("""
            # 🎬 VSeek Video Search Visualization
            
            Interactive video question answering with visual search. Upload a video, ask questions, 
            and visualize which scenes the model retrieves to answer your questions.
            """)
            
            with gr.Row():
                # Left column: Video input and controls
                with gr.Column(scale=1):
                    gr.Markdown("## 📹 Video Input")
                    
                    video_input = gr.Video(
                        label="Upload Video",
                        sources=["upload"]
                    )
                    
                    load_btn = gr.Button("🔄 Load & Index Video", variant="primary")
                    
                    video_status = gr.Textbox(
                        label="Video Status",
                        lines=5,
                        interactive=False
                    )
                    
                    gr.Markdown("### Video Preview (Uniform Sample)")
                    video_preview = gr.Gallery(
                        label="Video Frames",
                        columns=4,
                        rows=2,
                        height=300
                    )
                
                # Right column: Query and results
                with gr.Column(scale=2):
                    gr.Markdown("## ❓ Question Answering")
                    
                    with gr.Row():
                        question_input = gr.Textbox(
                            label="Enter your question about the video",
                            placeholder="What is happening in the video?",
                            lines=2,
                            scale=4
                        )
                        use_llm = gr.Checkbox(
                            label="Use LLM Agent",
                            value=True,
                            scale=1
                        )
                    
                    run_btn = gr.Button("🚀 Run Query", variant="primary")
                    
                    query_status = gr.Textbox(
                        label="Query Status",
                        lines=3,
                        interactive=False
                    )
                    
                    gr.Markdown("### 🖼️ Retrieved Scenes")
                    result_gallery = gr.Gallery(
                        label="Retrieved Frames",
                        columns=4,
                        rows=4,
                        height=400
                    )
                    
                    gr.Markdown("### 📝 Agent Trace")
                    trace_output = gr.Textbox(
                        label="Thinking, Search Queries & Answer",
                        lines=15,
                        interactive=False,
                        show_copy_button=True
                    )
            
            # Manual search section
            with gr.Accordion("🔧 Manual Search (Advanced)", open=False):
                gr.Markdown("""
                Perform direct searches without the LLM agent. Useful for testing the retrieval system.
                """)
                
                with gr.Row():
                    manual_query = gr.Textbox(
                        label="Search Query",
                        placeholder="person walking",
                        scale=3
                    )
                    search_mode = gr.Dropdown(
                        choices=["base", "subtitle"],
                        value="base",
                        label="Search Mode",
                        scale=1
                    )
                    topk_slider = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=4,
                        step=1,
                        label="Top-K Windows",
                        scale=1
                    )
                
                manual_search_btn = gr.Button("🔍 Manual Search")
                
                manual_status = gr.Textbox(
                    label="Search Status",
                    lines=5,
                    interactive=False
                )
                
                manual_gallery = gr.Gallery(
                    label="Manual Search Results",
                    columns=4,
                    rows=2,
                    height=300
                )
            
            # Event handlers
            load_btn.click(
                fn=self.load_video,
                inputs=[video_input],
                outputs=[video_status, video_input, video_preview]
            )
            
            run_btn.click(
                fn=self.run_query,
                inputs=[question_input, use_llm],
                outputs=[query_status, result_gallery, trace_output]
            )
            
            manual_search_btn.click(
                fn=self.manual_search,
                inputs=[manual_query, search_mode, topk_slider],
                outputs=[manual_status, manual_gallery]
            )
            
            # Example questions
            gr.Markdown("### 💡 Example Questions")
            gr.Examples(
                examples=[
                    ["What is the main activity happening in this video?"],
                    ["Who are the people in the video and what are they doing?"],
                    ["What objects can you see in the video?"],
                    ["Describe the scene at the beginning of the video."],
                    ["What happens at the end of the video?"],
                ],
                inputs=[question_input]
            )
        
        return demo


def main():
    """Main entry point for the application."""
    import argparse
    
    parser = argparse.ArgumentParser(description="VSeek Video Search Visualization")
    parser.add_argument(
        "--viclip-model-path",
        type=str,
        default="/nas/mars/model_weights/viclip/ViClip-InternVid-10M-FLT.pth",
        help="Path to ViClip model weights"
    )
    parser.add_argument(
        "--viclip-tokenizer-path",
        type=str,
        default="/nas/mars/model_weights/viclip/bpe_simple_vocab_16e6.txt.gz",
        help="Path to ViClip tokenizer"
    )
    parser.add_argument(
        "--llm-server-url",
        type=str,
        default="http://localhost:8767/v1",
        help="URL of the LLM server"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Name of the LLM model"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU number to use"
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=8,
        help="Window size for frame grouping"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the Gradio app"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public share link"
    )
    
    args = parser.parse_args()
    
    # Create the application
    app = VSeekVisualizationApp(
        viclip_model_path=args.viclip_model_path,
        viclip_tokenizer_path=args.viclip_tokenizer_path,
        llm_server_url=args.llm_server_url,
        model_name=args.model_name,
        gpu_number=args.gpu,
        window_size=args.window_size
    )
    
    # Build and launch the interface
    demo = app.build_interface()
    demo.launch(
        server_port=args.port,
        share=args.share,
        show_error=True
    )


if __name__ == "__main__":
    main()
