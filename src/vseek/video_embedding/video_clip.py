"""ViClip class to get feature vectors of a video clip."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from vseek.utils.torch import get_device
from vseek.video_embedding._base import (
    ComputerVisionModelVideoEmbeddingBase,
)
from vseek.video_embedding.lib_viclip import frames2tensor, get_viclip, get_vid_feat

if TYPE_CHECKING:
    from torch import Tensor

EXPECTED_FRAME_DIMENSIONS = 4
EXPECTED_SINGLE_FRAME_DIMENSIONS = 3


class ViClip(ComputerVisionModelVideoEmbeddingBase):
    """ViClip class to get feature vectors of a video clip."""

    def __init__(
        self,
        pretrained_model_path: str,
        size: str = "l",
        gpu_number: int = 0,
    ) -> None:
        """Initialize the ViClip model."""
        self.model_cfg = {
            "size": size,
            "pretrained": pretrained_model_path,
        }
        self.model = get_viclip(self.model_cfg["size"], self.model_cfg["pretrained"])
        self.device = get_device(gpu_number)
        self.clip = self.model["viclip"]
        self.tokenizer = self.model["tokenizer"]

        self.clip = self.clip.to(self.device)

    def get_text_embedding(self, text: str) -> Tensor:
        """Get text embedding of a text.

        Args:
            text: The text to get the embedding of

        Returns:
            The embedding of the text as a PyTorch tensor
        """
        return self.clip.get_text_features(text, self.tokenizer)

    def get_feature(self, frames: np.ndarray | list[np.ndarray]) -> Tensor:
        """Get feature vectors of a video clip.

        Args:
            frames: frames of the video clip
                (numpy array or list of numpy arrays)

        Returns:
            The feature vectors of the video clip as a PyTorch tensor
        """
        if isinstance(frames, np.ndarray):
            if len(frames.shape) != EXPECTED_FRAME_DIMENSIONS:
                msg = (
                    f"Dimension of frames should be {EXPECTED_FRAME_DIMENSIONS}",  # noqa: E501
                    "(num_frames, height, width, channels).",
                )
                raise ValueError(msg)
        elif isinstance(frames, list):
            expanded_frames = []
            for frame in frames:
                if (
                    not isinstance(frame, np.ndarray)
                    or len(frame.shape) != EXPECTED_SINGLE_FRAME_DIMENSIONS
                ):
                    msg = (
                        "All frames in the list must be",
                        f"{EXPECTED_SINGLE_FRAME_DIMENSIONS}D",
                        "numpy arrays (height, width, channels).",
                    )
                    raise ValueError(msg)
                expanded_frames.append(
                    np.expand_dims(frame, axis=0)
                )  # Shape: (1, height, width, channels)

            # Concatenate along the first dimension
            frames = np.concatenate(
                expanded_frames, axis=0
            )  # Shape: (num_frames, height, width, channels)

        frames_tensor = frames2tensor(frames, device=self.device)
        return get_vid_feat(frames_tensor, self.clip).cpu()
