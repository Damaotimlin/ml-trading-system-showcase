"""Integration point (stubbed in this showcase).

In the full system, combos with no tuned hyper-parameters of their own can
inherit them from a "representative" combo with a similar horizon.
"""
from pathlib import Path
from typing import Optional


def copy_best_json_from_representative(interval: str, predict_days: float, blueprint: str, model_dir: Path,
                                       mode: Optional[str] = None, force: bool = False,
                                       verbose: bool = True) -> bool:
    return False
