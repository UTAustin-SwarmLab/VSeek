import json
import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field


class SingleFrame(BaseModel):
    """Frame class."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    frame_idx: int = Field(..., description="The index of the frame")
    real_video_index: int = Field(..., description="The index of the video")
    image: np.ndarray = Field(..., description="The image as ndarray")


class VideoFrames(BaseModel):
    """Window frames class."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    frames: list[SingleFrame] = Field(
        default_factory=list, description="The list of frames"
    )
    embeddings: list[torch.Tensor] = Field(
        default_factory=list, description="The list of embeddings"
    )
    subtitles: list[str] = Field(
        default_factory=list, description="The list of captions"
    )
    # Frame windown operation
    frames_by_window: dict[int, list[SingleFrame]] = Field(default_factory=dict)
    subtitles_by_window: dict[int, list[str]] = Field(default_factory=dict)
    unique_subtitles_by_window: dict[int, str] = Field(default_factory=dict)
    window_size: Optional[int] = Field(None, description="The size of the window")
    window_index_map: dict[int, Tuple[int, int]] = Field(default_factory=dict)
    window_index: int = 0

    def add_frame(self, frame: SingleFrame) -> None:
        self.frames.append(frame)
        if self.window_size:
            if len(self.frames) == self.window_size:
                self.window_index_map[self.window_index] = self.get_window_range(
                    self.window_index
                )
                self.frames_by_window[self.window_index] = self.frames
                self.window_index += 1
                self.frames = []

    def add_subtitle(self, subtitle: str) -> None:
        """Add a subtitle to the VideoFrames.

        You must add frames before adding subtitles.

        Args:
            subtitle: The subtitle to add
        """
        self.subtitles.append(subtitle)
        if self.window_size:
            if len(self.subtitles) == self.window_size:
                window_index = self.window_index - 1
                start_idx, end_idx = self.get_window_range(window_index)

                self.subtitles_by_window[window_index] = self.subtitles
                unique_subtitles = list(dict.fromkeys(self.subtitles))
                self.unique_subtitles_by_window[window_index] = ".".join(
                    unique_subtitles
                )
                self.subtitles = []

    def add_embedding(self, embedding: torch.Tensor) -> None:
        self.embeddings.append(embedding)

    def get_frame_chunk(self, window_idx: int) -> list[np.ndarray]:
        return [frame.image for frame in self.frames_by_window[window_idx]]

    def get_window_range(self, window_idx: int) -> tuple[int, int]:
        """Get the frame range for a given window index.

        Args:
            window_idx: The window index

        Returns:
            Tuple of (start_idx, end_idx) for the window (both inclusive)
        """
        start_idx = window_idx * self.window_size
        end_idx = (window_idx + 1) * self.window_size - 1
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
        """Save using hybrid approach: JSON metadata + separate binary data."""
        # Create directory structure
        base_dir = filepath.with_suffix("")
        base_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata (everything except large arrays)
        metadata = {
            "window_size": self.window_size,
            "window_index": self.window_index,
            "window_index_map": {
                str(k): list(v) for k, v in self.window_index_map.items()
            },
            "frames_count": len(self.frames),
            "embeddings_count": len(self.embeddings),
            "subtitles": self.subtitles,
            "subtitles_by_window": {
                str(k): v for k, v in self.subtitles_by_window.items()
            },
            "unique_subtitles_by_window": {
                str(k): v for k, v in self.unique_subtitles_by_window.items()
            },
        }

        # Save frame metadata and images separately
        frames_dir = base_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        frames_metadata = []

        for i, frame in enumerate(self.frames):
            frame_meta = {
                "frame_idx": frame.frame_idx,
                "real_video_index": frame.real_video_index,
                "image_file": f"frame_{i}.npz",
            }
            frames_metadata.append(frame_meta)
            # Save image as compressed numpy array
            np.savez_compressed(frames_dir / f"frame_{i}.npz", image=frame.image)

        # Save windowed frames
        frames_by_window_meta = {}
        for window_idx, window_frames in self.frames_by_window.items():
            window_dir = frames_dir / f"window_{window_idx}"
            window_dir.mkdir(exist_ok=True)
            window_frames_meta = []

            for i, frame in enumerate(window_frames):
                frame_meta = {
                    "frame_idx": frame.frame_idx,
                    "real_video_index": frame.real_video_index,
                    "image_file": f"frame_{i}.npz",
                }
                window_frames_meta.append(frame_meta)
                np.savez_compressed(window_dir / f"frame_{i}.npz", image=frame.image)

            frames_by_window_meta[str(window_idx)] = window_frames_meta

        metadata["frames"] = frames_metadata
        metadata["frames_by_window"] = frames_by_window_meta

        # Save metadata as JSON
        with open(base_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Save embeddings using PyTorch
        if self.embeddings:
            torch.save(self.embeddings, base_dir / "embeddings.pt")

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
        """Load from hybrid format."""
        base_dir = filepath.with_suffix("")

        # Load metadata
        with open(base_dir / "metadata.json", "r") as f:
            metadata = json.load(f)

        # Create VideoFrames instance
        video_frames = cls(
            window_size=metadata["window_size"],
            window_index=metadata["window_index"],
            window_index_map={
                int(k): tuple(v) for k, v in metadata["window_index_map"].items()
            },
        )

        # Restore subtitle fields
        video_frames.subtitles = metadata.get("subtitles", [])
        video_frames.subtitles_by_window = {
            int(k): v for k, v in metadata.get("subtitles_by_window", {}).items()
        }
        video_frames.unique_subtitles_by_window = {
            int(k): v for k, v in metadata.get("unique_subtitles_by_window", {}).items()
        }

        # Load frames
        frames_dir = base_dir / "frames"
        for frame_meta in metadata["frames"]:
            image_data = np.load(frames_dir / frame_meta["image_file"])
            frame = SingleFrame(
                frame_idx=frame_meta["frame_idx"],
                real_video_index=frame_meta["real_video_index"],
                image=image_data["image"],
            )
            video_frames.frames.append(frame)

        # Load windowed frames
        for window_idx_str, window_frames_meta in metadata["frames_by_window"].items():
            window_idx = int(window_idx_str)
            window_frames = []
            window_dir = frames_dir / f"window_{window_idx}"

            for frame_meta in window_frames_meta:
                image_data = np.load(window_dir / frame_meta["image_file"])
                frame = SingleFrame(
                    frame_idx=frame_meta["frame_idx"],
                    real_video_index=frame_meta["real_video_index"],
                    image=image_data["image"],
                )
                window_frames.append(frame)

            video_frames.frames_by_window[window_idx] = window_frames

        # Load embeddings
        embeddings_file = base_dir / "embeddings.pt"
        if embeddings_file.exists():
            video_frames.embeddings = torch.load(embeddings_file)

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
