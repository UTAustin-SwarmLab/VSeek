"""Coargus's Config utilities."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .omega_conf import load_config_from_dict, load_config_from_yaml

if TYPE_CHECKING:
    from omegaconf import DictConfig


def validate_and_read_config(
    config_path: str | Path | None = None, config: dict | DictConfig = None
) -> None:
    """Validate and read the config."""
    if (config_path is not None) == (config is not None):
        msg = "Exactly one of 'config_path' or 'config' must be provided"
        raise ValueError(msg)

    if config_path is not None:
        if isinstance(config_path, Path):
            config_path = str(config_path)
        config = load_config_from_yaml(config_path)

    elif isinstance(config, dict):
        config = load_config_from_dict(config)

    return config
