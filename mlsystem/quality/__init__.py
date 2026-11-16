"""
Model quality control: the statistics that decide which models are trusted.

Four small, dependency-light pieces that mirror stages of the full system:

- ``walk_forward``  chronological splits with an embargo, and the generalization gate
- ``convergence``   plateau detection that stops a hyper-parameter search early
- ``scoring``       credibility-shrunk scores that keep unproven models out of the ensemble
- ``ensemble``      score-weighted average turned into an UP / DOWN / FLAT signal

These are compact re-implementations written for this showcase. Every threshold
is a demo value passed in by the caller; production settings are not included.
See ``examples/pipeline_demo.py`` for all four running end to end on synthetic data.
"""

from .convergence import ConvergenceTracker
from .ensemble import Member, Signal, build_signal
from .scoring import ComboRecord, ComboScore, ScorePolicy, score_combo, select_combos, shrunk_score, \
    staleness_decay, wilson_lower_bound
from .walk_forward import Fold, FoldResult, GatePolicy, GateVerdict, GeneralizationGate, walk_forward_splits

__all__ = [
    "ConvergenceTracker",
    "Member", "Signal", "build_signal",
    "ComboRecord", "ComboScore", "ScorePolicy", "score_combo", "select_combos",
    "shrunk_score", "staleness_decay", "wilson_lower_bound",
    "Fold", "FoldResult", "GatePolicy", "GateVerdict", "GeneralizationGate", "walk_forward_splits",
]
