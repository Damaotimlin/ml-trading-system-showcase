from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from .tree import DecisionTreeSpec


def load_spec_file(path: Path) -> DecisionTreeSpec:
    """Load a decision tree specification from a YAML or JSON file."""
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    elif path.suffix.lower() == ".json":
        import json

        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        raise ValueError(f"Unsupported spec file format: {path.suffix}")

    if not isinstance(data, Dict):
        raise ValueError("Decision tree spec file must contain a mapping at the top level")

    return DecisionTreeSpec.from_dict(data)
