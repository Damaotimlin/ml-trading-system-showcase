"""
Combo-level quality control: which models have earned a place in the ensemble.

Every model combination ("combo") is graded continuously against what actually
happened. Raw averages are misleading for combos with few graded predictions,
so the score is built from three conservative pieces:

- credibility shrinkage  pulls the average of a small sample toward zero
- Wilson lower bound     a pessimistic estimate of the true hit rate
- staleness decay        fades a score that has not been refreshed recently

A combo is admitted only if both its shrunk score and its hit-rate floor clear
the policy. The policy values are supplied by the caller.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable


def wilson_lower_bound(successes: float, n: float, z: float = 1.96) -> float:
    """Lower bound of the Wilson score interval for a binomial proportion (0.0 when n == 0)."""
    if n <= 0:
        return 0.0
    p = min(1.0, max(0.0, successes / n))
    z2 = z * z
    centre = p + z2 / (2.0 * n)
    spread = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    return max(0.0, (centre - spread) / (1.0 + z2 / n))


def shrunk_score(mean: float, n: float, k: float) -> float:
    """Credibility-shrunk mean: ``mean * n / (n + k)``. ``k`` is the sample size worth half weight."""
    if n <= 0:
        return 0.0
    return mean * n / (n + k)


def staleness_decay(stale_days: float, grace_days: float, half_life_days: float) -> float:
    """1.0 inside the grace period, then halving every ``half_life_days``."""
    return 0.5 ** (max(0.0, stale_days - grace_days) / half_life_days)


@dataclass(frozen=True)
class ComboRecord:
    """Graded track record of one combo."""
    name: str
    hits: int               # predictions with the right direction
    n: int                  # graded predictions
    mean_points: float      # average grade per prediction (higher is better)
    stale_days: float = 0.0


@dataclass(frozen=True)
class ScorePolicy:
    credibility_k: float
    min_score: float
    min_hit_floor: float
    grace_days: float = 30.0
    half_life_days: float = 15.0


@dataclass(frozen=True)
class ComboScore:
    name: str
    score: float
    hit_floor: float
    admitted: bool
    reason: str


def score_combo(record: ComboRecord, policy: ScorePolicy) -> ComboScore:
    decay = staleness_decay(record.stale_days, policy.grace_days, policy.half_life_days)
    score = shrunk_score(record.mean_points, record.n, policy.credibility_k) * decay
    hit_floor = wilson_lower_bound(record.hits, record.n)
    if score < policy.min_score:
        reason = f"score {score:.3f} below {policy.min_score:.3f}"
    elif hit_floor < policy.min_hit_floor:
        reason = f"hit-rate floor {hit_floor:.1%} below {policy.min_hit_floor:.1%}"
    else:
        reason = "admitted"
    return ComboScore(record.name, score, hit_floor, reason == "admitted", reason)


def select_combos(records: Iterable[ComboRecord], policy: ScorePolicy) -> list[ComboScore]:
    """Score every combo, best first."""
    return sorted((score_combo(r, policy) for r in records), key=lambda s: s.score, reverse=True)
