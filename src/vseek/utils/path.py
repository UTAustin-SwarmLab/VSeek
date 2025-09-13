"""Coargus's path utility functions."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path


def add_datetime_to_dir_or_file(
    directory_name: str, file_name: str | None = None
) -> str:
    """Add a datetime string to a directory or file name.

    Args:
        directory_name (str): Directory name.
        file_name (str | None): File name.
    """
    current_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")  # noqa: DTZ005
    if file_name:
        # Check for a file extension
        if "." not in file_name:
            msg = "File name must include an extension (e.g., 'file.txt')."
            raise ValueError(msg)

        # Extract base name and extension
        base_name, ext = file_name.rsplit(".", 1)
        path = f"{base_name}_{current_time}.{ext}"
    else:
        path = f"{directory_name}_{current_time}"

    return path


def mkdir_with_datetime(directory_name: str) -> Path:
    """Create a directory with the current datetime appended to the name.

    Args:
        directory_name (str): Directory name.
    """
    path = Path(add_datetime_to_dir_or_file(directory_name))
    path.mkdir(parents=True, exist_ok=True)
    logging.info("Created directory %s", str(path))

    return path
