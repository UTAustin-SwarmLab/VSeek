import json
import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field

# Video IO: use OpenCV for writing (encoding), Decord for reading (fast)
import cv2
try:
    from decord import VideoReader as DecordVideoReader
    from decord import cpu as decord_cpu
    _HAS_DECORD = True
except Exception:
    _HAS_DECORD = False


class SingleFrame(BaseModel):
    """Frame class."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    frame_idx: int = Field(..., description="The index of the frame")
    real_video_index: int = Field(..., description="The index of the video")
    image: np.ndarray = Field(..., description="The image as ndarray")


class VideoFrames(BaseModel):
    """Window frames class."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Store frames directly as numpy arrays (RGB)
    all_frames: list[np.ndarray] = Field(
        default_factory=list, description="The list of frames (RGB)"
    )
    embeddings: list[torch.Tensor] = Field(
        default_factory=list, description="The list of embeddings"
    )
    subtitle_embeddings: list[torch.Tensor] = Field(
        default_factory=list, description="The list of subtitle embeddings"
    )
    all_subtitles: list[str] = Field(
        default_factory=list, description="The list of captions"
    )
    # Frame windown operation
    frames_by_window: dict[int, list[np.ndarray]] = Field(default_factory=dict)
    subtitles_by_window: dict[int, list[str]] = Field(default_factory=dict)
    unique_subtitles_by_window: dict[int, str] = Field(default_factory=dict)
    window_size: Optional[int] = Field(None, description="The size of the window")
    window_index_map: dict[int, Tuple[int, int]] = Field(default_factory=dict)
    window_by_subtitle: dict[str, list[int]] = Field(default_factory=dict)
    window_index: int = Field(0, description="The current window index counter")
    
    def add_all_frames(self, frames: list[np.ndarray]) -> None:
        self.all_frames = frames
    
    def partition_frames(self) -> None:
        if self.window_size:
            self.frames_by_window.clear()
            self.window_index_map.clear()
            self.window_index = 0
            total = len(self.all_frames)
            for i in range(0, total, self.window_size):
                start_idx, end_idx = self.get_window_range(i // self.window_size)
                self.window_index_map[i // self.window_size] = (start_idx, min(end_idx, total))
                self.frames_by_window[i // self.window_size] = self.all_frames[start_idx:min(end_idx, total)]
                self.window_index += 1

    def add_subtitle(self, subtitles: str) -> None:
        """Add a subtitle to the VideoFrames.

        You must add frames before adding subtitles.

        Args:
            subtitle: The subtitle to add
        """
        self.all_subtitles = subtitles

    def add_all_subtitles(self, subtitles: list[str]) -> None:
        self.all_subtitles = subtitles
    
    def partition_subtitles(self) -> None:
        
        # For each subtitle, check the start and end timestamp, add it to the corresponding window if its within the window size
        
        if self.window_size:
            for i in range(0, len(self.all_subtitles), self.window_size):
                self.subtitle_by_window[i] = sum(self.all_subtitles[i:i+self.window_size], [])
                self.unique_subtitles_by_window[i] = list(set(self.subtitle_by_window[i]))
                for sub in self.unique_subtitles_by_window[i]:
                    if sub not in self.subtitle_by_frame:
                        self.window_by_subtitle[sub] = [i]
                    else:
                        self.window_by_subtitle[sub].append(i)


    def add_embedding(self, embedding: torch.Tensor) -> None:
        self.embeddings.append(embedding)
    
    def add_subtitle_embedding(self, embedding: torch.Tensor) -> None:
        self.subtitle_embeddings.append(embedding)

    def get_frame_chunk(self, window_idx: int) -> list[np.ndarray]:
        return self.frames_by_window[window_idx]

    def get_window_range(self, window_idx: int) -> tuple[int, int]:
        """Get the frame range for a given window index.

        Args:
            window_idx: The window index

        Returns:
            Tuple of (start_idx, end_idx) for the window (both inclusive)
        """
        start_idx = window_idx * self.window_size
        end_idx = (window_idx + 1) * self.window_size
        return start_idx, end_idx

    def save(self, filepath: str | Path, method: str = "hybrid") -> None:
        """Save VideoFrames to disk using different methods.

        Args:
            filepath: Path to save the data
            method: Serialization method ("hybrid", "pickle", "torch")
                - "hybrid": Save metadata as JSON, arrays separately (recommended)
                - "pickle": Use Python pickle (simple but less portable)
                - "torch": Use PyTorch's save format
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if method == "hybrid":
            self._save_hybrid(filepath)
        elif method == "pickle":
            self._save_pickle(filepath)
        elif method == "torch":
            self._save_torch(filepath)
        else:
            raise ValueError(f"Unknown save method: {method}")

    def _save_hybrid(self, filepath: Path) -> None:
        """Save using hybrid approach: JSON metadata + single MP4 at 1 FPS."""
        # Create directory structure
        base_dir = filepath.with_suffix("")
        base_dir.mkdir(parents=True, exist_ok=True)

        # Prepare frames video
        video_path = base_dir / "frames.mp4"
        fps = 1
        if len(self.all_frames) > 0:
            first = self.all_frames[0]
            height, width = int(first.shape[0]), int(first.shape[1])
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
            for frame in self.all_frames:
                # Ensure uint8 and proper color for OpenCV (expects BGR)
                frame_uint8 = frame if frame.dtype == np.uint8 else frame.astype(np.uint8)
                if frame_uint8.ndim == 2:
                    frame_uint8 = cv2.cvtColor(frame_uint8, cv2.COLOR_GRAY2BGR)
                else:
                    frame_uint8 = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)
                writer.write(frame_uint8)
            writer.release()

        # Save metadata (no per-frame npz)
        metadata = {
            "window_size": self.window_size,
            "window_index": self.window_index,
            "window_index_map": {str(k): list(v) for k, v in self.window_index_map.items()},
            "frames_count": len(self.all_frames),
            "embeddings_count": len(self.embeddings),
            "all_subtitles": self.all_subtitles,
            "subtitles_by_window": {
                k: v for k, v in self.subtitles_by_window.items()
            },
            "unique_subtitles_by_window": {
                k: v for k, v in self.unique_subtitles_by_window.items()
            },
            "window_by_subtitle": {
                k: v for k, v in self.window_by_subtitle.items()
            },
            "video_file": "frames.mp4",
            "fps": fps, 
            
        }

        with open(base_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Save embeddings using PyTorch
        if self.embeddings:
            torch.save(self.embeddings, base_dir / "embeddings.pt")
        if self.subtitle_embeddings:
            torch.save(self.subtitle_embeddings, base_dir / "subtitle_embeddings.pt")

    def _save_pickle(self, filepath: Path) -> None:
        """Save using pickle (simple but less portable)."""
        with open(filepath.with_suffix(".pkl"), "wb") as f:
            pickle.dump(self, f)

    def _save_torch(self, filepath: Path) -> None:
        """Save using PyTorch's save format."""
        torch.save(self.model_dump(), filepath.with_suffix(".pt"))

    @classmethod
    def load(cls, filepath: str | Path, method: str = "hybrid") -> "VideoFrames":
        """Load VideoFrames from disk.

        Args:
            filepath: Path to load the data from
            method: Serialization method used ("hybrid", "pickle", "torch")

        Returns:
            VideoFrames instance
        """
        filepath = Path(filepath)

        if method == "hybrid":
            return cls._load_hybrid(filepath)
        elif method == "pickle":
            return cls._load_pickle(filepath)
        elif method == "torch":
            return cls._load_torch(filepath)
        else:
            raise ValueError(f"Unknown load method: {method}")

    @classmethod
    def _load_hybrid(cls, filepath: Path) -> "VideoFrames":
        """Load from hybrid format (read MP4 with Decord at 1 FPS)."""
        base_dir = filepath.with_suffix("")

        # Load metadata
        with open(base_dir / "metadata.json", "r") as f:
            metadata = json.load(f)

        # Create VideoFrames instance
        video_frames = cls(
            window_size=metadata.get("window_size"),
            window_index=metadata.get("window_index", 0),
            window_index_map={
                int(k): tuple(v) for k, v in metadata.get("window_index_map", {}).items()
            },
            window_by_subtitle={
                str(k): v for k, v in metadata.get("window_by_subtitle", {}).items()
            },
            subtitles_by_window={
                str(k): v for k, v in metadata.get("subtitles_by_window", {}).items()
            },
            unique_subtitles_by_window={
                str(k): v for k, v in metadata.get("unique_subtitles_by_window", {}).items()
            },
            all_subtitles=metadata.get("all_subtitles", []),
        )

        # Read frames video
        video_file = metadata.get("video_file", "frames.mp4")
        video_path = base_dir / video_file
        frames: list[np.ndarray] = []
        if _HAS_DECORD and video_path.exists():
            vr = DecordVideoReader(str(video_path), ctx=decord_cpu(0))
            for idx in range(len(vr)):
                frame_nd = vr[idx]
                frame_img = frame_nd.asnumpy() if hasattr(frame_nd, "asnumpy") else np.asarray(frame_nd)
                frames.append(frame_img)
        else:
            # Fallback to OpenCV if decord unavailable
            cap = cv2.VideoCapture(str(video_path))
            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            cap.release()

        video_frames.all_frames = frames
        video_frames.partition_frames()

        # Load embeddings
        embeddings_file = base_dir / "embeddings.pt"
        if embeddings_file.exists():
            video_frames.embeddings = torch.load(embeddings_file)
        
        subtitle_embeddings_file = base_dir / "subtitle_embeddings.pt"
        if subtitle_embeddings_file.exists():
            video_frames.subtitle_embeddings = torch.load(subtitle_embeddings_file)

        return video_frames

    @classmethod
    def _load_pickle(cls, filepath: Path) -> "VideoFrames":
        """Load from pickle format."""
        with open(filepath.with_suffix(".pkl"), "rb") as f:
            return pickle.load(f)

    @classmethod
    def _load_torch(cls, filepath: Path) -> "VideoFrames":
        """Load from PyTorch format."""
        data = torch.load(filepath.with_suffix(".pt"))
        return cls.model_validate(data)
