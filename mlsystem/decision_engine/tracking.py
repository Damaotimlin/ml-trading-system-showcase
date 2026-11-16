from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from zoneinfo import ZoneInfo


@dataclass(slots=True)
class DecisionRecord:
    """Structured record of a single decision node evaluation."""

    path: List[str]
    decision: str
    outcome: str
    detail: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Serialise record to a JSON-friendly dict."""
        return {
            "path": self.path,
            "decision": self.decision,
            "outcome": self.outcome,
            "detail": self.detail,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
        }


class DecisionTracker:
    """Utility to collect and render decision records."""

    def __init__(self) -> None:
        self._records: List[DecisionRecord] = []

    def add(
        self,
        path: Sequence[str],
        decision: str,
        outcome: str,
        detail: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._records.append(
            DecisionRecord(
                path=list(path),
                decision=decision,
                outcome=outcome,
                detail=detail,
                data=data or {},
            )
        )

    def extend(self, other: "DecisionTracker") -> None:
        self._records.extend(other._records)

    def clear(self) -> None:
        self._records.clear()

    def empty(self) -> bool:
        return len(self._records) == 0

    def records(self) -> List[DecisionRecord]:
        return list(self._records)

    def to_dicts(self) -> List[Dict[str, Any]]:
        return [record.to_dict() for record in self._records]

    def render(
        self,
        title: str = "Decision Tree",
        final_outcome: Optional[str] = None,
        final_detail: Optional[str] = None,
    ) -> None:
        # Get current time in ET timezone
        et_tz = ZoneInfo("America/New_York")
        now_et = datetime.now(et_tz)
        timestamp = now_et.strftime('%Y-%m/%d %H:%M ET')

        if not self._records:
            print(f"\n📌 {title} [{timestamp}]: No decisions recorded")
            return

        print(f"\n📌 {title} [{timestamp}]")
        last_path: List[str] = []

        for record in self._records:
            for depth, label in enumerate(record.path):
                should_print_label = len(last_path) <= depth or last_path[depth] != label
                if should_print_label:
                    indent = "  " * depth
                    connector = "└─" if depth else "-"
                    print(f"{indent}{connector} {label}")
            indent = "  " * len(record.path)
            detail_text = f" — {record.detail}" if record.detail else ""
            print(f"{indent}   ↳ {record.decision} → {record.outcome}{detail_text}")
            last_path = record.path

        if final_outcome:
            print(f"\n➡️  Final Outcome: {final_outcome}")
            if final_detail:
                print(f"    {final_detail}")

    def export_jsonl(self, path: Path) -> None:
        """Write records to a JSON Lines file, appending if file exists."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            for record in self._records:
                fh.write(json.dumps(record.to_dict()) + "\n")
