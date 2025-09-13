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
    # Frame windown operation
    frames_by_window: dict[int, list[SingleFrame]] = Field(default_factory=dict)
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
