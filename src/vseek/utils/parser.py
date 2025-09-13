"""Coargus's parsing utility functions."""

from __future__ import annotations


def parse_f_str(f_string: str) -> str:
    """Parse long lines of f-string into a single line."""
    return " ".join(f_string.split())
