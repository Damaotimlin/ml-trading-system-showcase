"""
Convergence-aware tuning: stop a search once it has stopped improving.

A fixed trial budget wastes compute on searches that converged early and cuts
off ones that are still improving. The tracker records every new best value and
reports a plateau when the best has not improved by a meaningful margin for
``patience`` consecutive trials. It is search-library agnostic: call ``record``
from any trial callback.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ConvergenceTracker:
    patience: int                    # trials without meaningful improvement before stopping
    min_improvement: float = 0.0     # relative gain over the best value needed to count
    best_value: float | None = None
    best_trial: int | None = None
    trials_seen: int = 0
    history: list[tuple[int, float]] = field(default_factory=list)   # (trial, new best)

    def record(self, trial: int, value: float) -> bool:
        """Record one finished trial (maximising). Returns True if it set a meaningful new best."""
        self.trials_seen = max(self.trials_seen, trial + 1)
        if self.best_value is not None:
            needed = self.best_value + abs(self.best_value) * self.min_improvement
            if value <= needed:
                return False
        self.best_value, self.best_trial = value, trial
        self.history.append((trial, value))
        return True

    @property
    def plateaued(self) -> bool:
        if self.best_trial is None:
            return False
        return self.trials_seen - 1 - self.best_trial >= self.patience

    def summary(self) -> dict:
        return {"trials": self.trials_seen, "best_value": self.best_value, "best_trial": self.best_trial,
                "improvements": len(self.history), "plateaued": self.plateaued}
