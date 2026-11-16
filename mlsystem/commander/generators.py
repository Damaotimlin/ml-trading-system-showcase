"""Integration point (stubbed in this showcase).

In the full system this module turns the market state into concrete tasks and
knows where each combo's tuned hyper-parameters (best.json) live.
"""
from pathlib import Path
from typing import Optional

from .registry import ComboKey


def find_best_json_path(combo: ComboKey) -> Optional[Path]:
    return None
