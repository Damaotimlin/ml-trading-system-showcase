"""
Walk-forward validation and the generalization gate.

Time-series models must be judged on data that comes strictly *after* the data
they were fitted on. ``walk_forward_splits`` produces such chronological folds,
with an optional embargo between train and test so that overlapping labels
cannot leak across the boundary. ``GeneralizationGate`` then decides whether a
tuned candidate is allowed any further: a good in-sample number is not evidence,
only the out-of-sample folds are.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import fmean
from typing import Sequence


@dataclass(frozen=True)
class Fold:
    """One chronological split. ``train`` always ends before ``test`` begins."""
    train: slice
    test: slice


def walk_forward_splits(n_samples: int, *, n_folds: int, test_size: int, train_size: int | None = None,
                        embargo: int = 0) -> list[Fold]:
    """Return ``n_folds`` chronological folds over ``n_samples`` ordered observations.

    The test blocks are consecutive and end at the last sample. With ``train_size``
    the training window rolls (fixed length); without it the window expands from
    the first sample. ``embargo`` samples are dropped between train and test.
    """
    if min(n_folds, test_size) < 1 or embargo < 0:
        raise ValueError("n_folds and test_size must be positive and embargo non-negative")
    first_test = n_samples - n_folds * test_size
    min_train = train_size if train_size is not None else 1
    if first_test - embargo < min_train:
        raise ValueError("not enough samples for the requested folds")
    folds = []
    for k in range(n_folds):
        test_start = first_test + k * test_size
        train_end = test_start - embargo
        train_start = 0 if train_size is None else train_end - train_size
        folds.append(Fold(slice(train_start, train_end), slice(test_start, test_start + test_size)))
    return folds


@dataclass(frozen=True)
class FoldResult:
    """Skill of one candidate on one fold. Higher is better and 0 means no skill."""
    in_sample: float
    out_of_sample: float


@dataclass(frozen=True)
class GatePolicy:
    """Thresholds a candidate must clear (callers supply their own values)."""
    min_out_of_sample: float      # mean out-of-sample skill must exceed this
    min_positive_folds: float     # share of folds with positive out-of-sample skill
    min_retention: float          # out-of-sample skill kept, as a fraction of in-sample


@dataclass(frozen=True)
class GateVerdict:
    passed: bool
    in_sample: float
    out_of_sample: float
    positive_folds: float
    retention: float
    reasons: list[str] = field(default_factory=list)


class GeneralizationGate:
    """Hard gate: a candidate that fails is discarded however good it looked in-sample."""

    def __init__(self, policy: GatePolicy):
        self.policy = policy

    def evaluate(self, results: Sequence[FoldResult]) -> GateVerdict:
        if not results:
            return GateVerdict(False, 0.0, 0.0, 0.0, 0.0, ["no out-of-sample evidence"])
        in_sample = fmean(r.in_sample for r in results)
        out_of_sample = fmean(r.out_of_sample for r in results)
        positive = sum(r.out_of_sample > 0 for r in results) / len(results)
        retention = out_of_sample / in_sample if in_sample > 0 else 0.0

        policy, reasons = self.policy, []
        if out_of_sample <= policy.min_out_of_sample:
            reasons.append(f"out-of-sample skill {out_of_sample:+.3f} not above {policy.min_out_of_sample:+.3f}")
        if positive < policy.min_positive_folds:
            reasons.append(f"only {positive:.0%} of folds positive (need {policy.min_positive_folds:.0%})")
        if retention < policy.min_retention:
            reasons.append(f"kept {retention:.0%} of in-sample skill (need {policy.min_retention:.0%})")
        return GateVerdict(not reasons, in_sample, out_of_sample, positive, retention, reasons)
