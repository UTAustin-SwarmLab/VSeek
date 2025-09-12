"""Abstract base class for getting feature vectors of a video clip."""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class ComputerVisionModelVideoEmbeddingBase(abc.ABC):
    """Abstract base class for getting feature vectors of a video clip."""

    @abc.abstractmethod
    def get_feature(self, frames: list[np.ndarray]) -> any:
        """Get feature vectors of a video clip.

        Args:
            frames: frames of the video clip

        Returns:
            The feature vectors of the video clip
        """
