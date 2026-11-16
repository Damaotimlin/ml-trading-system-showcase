"""
Ensemble forecasts into one signal.

Admitted members vote with their forecast, weighted by their quality score. The
weighted average gives the direction; forecasts inside the flat band are treated
as "no view". Confidence is the share of weight that agrees with the direction,
so a split ensemble produces a weak signal even when the average is large.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Member:
    name: str
    forecast: float     # expected move, signed
    weight: float       # quality score, > 0


@dataclass(frozen=True)
class Signal:
    direction: str      # "UP", "DOWN" or "FLAT"
    confidence: float   # 0..1
    forecast: float     # weighted average forecast
    members: int


def build_signal(members: Sequence[Member], flat_band: float) -> Signal:
    members = [m for m in members if m.weight > 0]
    total = sum(m.weight for m in members)
    if not members or total <= 0:
        return Signal("FLAT", 0.0, 0.0, 0)
    forecast = sum(m.forecast * m.weight for m in members) / total
    if abs(forecast) <= flat_band:
        return Signal("FLAT", 0.0, forecast, len(members))
    sign = 1 if forecast > 0 else -1
    agreeing = sum(m.weight for m in members if m.forecast * sign > flat_band)
    return Signal("UP" if sign > 0 else "DOWN", agreeing / total, forecast, len(members))
