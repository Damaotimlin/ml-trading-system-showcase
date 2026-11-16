"""Integration point (stubbed in this showcase).

In the full system this module manages the per-combo store of seed-ensemble
weights on disk and answers whether they are fresh enough to reuse.
"""
from pathlib import Path
from typing import Optional


def weights_fresh(ticker: str, interval_name: str, arch: str, mode: str, predict_days: float, *,
                  n_seeds: int, best_json_path: Optional[Path], max_age_days: int) -> tuple[bool, str]:
    return False, "model store not included in showcase"


def weights_usable(ticker: str, interval_name: str, arch: str, mode: str, predict_days: float, *,
                   n_seeds: int) -> tuple[bool, str]:
    return False, "model store not included in showcase"
