"""Coargus's torch utility functions."""

from __future__ import annotations

import torch


def get_device(gpu_number: int | None = None) -> torch.device:
    """Get the device on which to run the code.

    Args:
        gpu_number (int, optional): GPU number to use. Defaults to None.

    Returns:
        torch.device: The device to use
    """
    if torch.cuda.is_available():
        if gpu_number:
            return torch.device(f"cuda:{gpu_number}")
        return torch.device("cuda")
    return torch.device("cpu")
