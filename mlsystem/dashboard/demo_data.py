"""
Synthetic data behind the demo dashboard.

The seeded pipeline demo is run once; its combos then forecast a "live" stretch
of the synthetic series one step at a time. At every step each combo's score is
computed only from predictions that were already graded at that moment, so the
replay follows the same no-lookahead rule as the real system: a combo has to
earn its weight before it can influence the signal.

``snapshot(cursor)`` returns everything the pages show for one moment of the
replay. It is a pure function of ``cursor``, which keeps the views testable.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np

from mlsystem.decision_engine import DecisionContext, DecisionTracker, DecisionTreeBuilder, load_spec_file
from mlsystem.decision_engine.examples.example_nodes import example_registry
from mlsystem.quality import ComboRecord, Member, build_signal, score_combo, walk_forward_splits
from mlsystem.quality.examples import pipeline_demo as demo

TARGET_NAME = "DEMO – Close"
INTERVAL = "5m"
BAR = timedelta(minutes=5)
START = datetime(2026, 1, 5, 14, 30, tzinfo=timezone.utc)     # arbitrary origin for the synthetic clock
RETURN_SCALE = 0.004                                         # synthetic units -> fractional price move
WARM_UP = 100                                                # graded predictions before the replay window opens
CHART_BARS = 120
TABLE_ROWS = 40
NEW_COMBO_HISTORY = 8                                        # "compact-new" has only this many graded predictions
SECONDS_PER_BAR = 3                                          # replay speed of the auto-updating page


class Replay:
    """Everything that does not depend on the cursor, computed once."""

    def __init__(self) -> None:
        self.pipeline = demo.run(verbose=False)
        rng = np.random.default_rng(demo.SEED)
        features, self.target = demo.make_data(rng)
        self.price = 100.0 * np.exp(np.cumsum(self.target * RETURN_SCALE))

        folds = walk_forward_splits(demo.N_SAMPLES - 1, n_folds=5, test_size=50,
                                    train_size=demo.TRAIN_WINDOW, embargo=demo.EMBARGO)
        self.first = folds[0].test.start                     # first step of the live stretch
        self.last = folds[-1].test.stop - 1
        self.survivors = [name for name, verdict in self.pipeline["verdicts"].items() if verdict.passed]
        self.forecast, self.hit = {}, {}
        for name in self.survivors:
            x = features[:, demo.COMBOS[name]]
            parts = [x[f.test] @ demo.fit_ridge(x[f.train], self.target[f.train], self.pipeline["tuned"][name])
                     for f in folds]
            self.forecast[name] = np.concatenate(parts)
            self.hit[name] = np.sign(self.forecast[name]) == np.sign(self.target[self.first:self.last + 1])
        self.tree = DecisionTreeBuilder(example_registry).build(
            load_spec_file(Path(demo.__file__).parents[2] / "decision_engine" / "examples" / "example_tree.yaml"))

    # -- per-step views (all strictly causal: step i uses grades from steps < i) --

    def records(self, step: int) -> list[ComboRecord]:
        graded = step - self.first
        out = []
        for name in self.survivors:
            hits = self.hit[name][:graded]
            out.append(ComboRecord(name, int(hits.sum()), graded, float(2 * hits.mean() - 1) if graded else 0.0))
            if name == "compact":
                recent = hits[-NEW_COMBO_HISTORY:]
                out.append(ComboRecord("compact-new", int(recent.sum()), recent.size,
                                       float(2 * recent.mean() - 1) if recent.size else 0.0))
        return out

    def members(self, step: int) -> list[Member]:
        offset = step - self.first
        out = []
        for record in self.records(step):
            score = score_combo(record, demo.SCORING)
            if score.admitted:
                source = record.name.removesuffix("-new")
                out.append(Member(record.name, float(self.forecast[source][offset]), score.score))
        return out

    def signal(self, step: int):
        return build_signal(self.members(step), demo.FLAT_BAND)

    def time_of(self, step: int) -> datetime:
        return START + BAR * step


@lru_cache(maxsize=1)
def replay() -> Replay:
    return Replay()


def cursor_now(now: float | None = None) -> int:
    """Replay position for wall-clock ``now``: one bar every few seconds, looping."""
    r = replay()
    span = r.last - (r.first + WARM_UP)
    return r.first + WARM_UP + int((time.time() if now is None else now) // SECONDS_PER_BAR) % span


def _iso(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%d %H:%M")


def snapshot(cursor: int | None = None) -> dict:
    """All data for the pages at replay position ``cursor`` (the latest *completed* bar)."""
    r = replay()
    cursor = cursor_now() if cursor is None else max(r.first + WARM_UP, min(int(cursor), r.last - 1))

    # Chart: actual close and the ensemble's one-step-ahead prediction for the same bar.
    steps = range(max(r.first + 1, cursor - CHART_BARS + 1), cursor + 1)
    signals = {step: r.signal(step) for step in steps}
    series = {
        "time": [_iso(r.time_of(s)) for s in steps],
        "actual": [round(float(r.price[s]), 3) for s in steps],
        "predicted": [round(float(r.price[s - 1] * np.exp(signals[s].forecast * RETURN_SCALE)), 3) if signals[s].members
                      else None for s in steps],
        "direction": [signals[s].direction for s in steps],
    }

    # Table: the most recent graded predictions, one row per combo and bar.
    rows = []
    for step in range(cursor, max(r.first, cursor - TABLE_ROWS // max(1, len(r.survivors))), -1):
        scores = {rec.name: score_combo(rec, demo.SCORING) for rec in r.records(step)}
        for name in r.survivors:
            forecast = float(r.forecast[name][step - r.first])
            live, actual = float(r.price[step - 1]), float(r.price[step])
            predicted = live * float(np.exp(forecast * RETURN_SCALE))
            rows.append({
                "live_time": _iso(r.time_of(step - 1)), "live_value": round(live, 3),
                "predict_time": _iso(r.time_of(step)), "predicted_value": round(predicted, 3),
                "diff_pct": round((predicted / live - 1) * 100, 3), "interval": INTERVAL, "model": name,
                "combo_score": round(scores[name].score, 3), "hit_floor": round(scores[name].hit_floor, 3),
                "error_pct": round((predicted / actual - 1) * 100, 3), "hit": bool(r.hit[name][step - r.first]),
            })

    # Cards: what the ensemble called recently, and the signal for the bar that has not closed yet.
    recent = [s for s in range(cursor - 49, cursor + 1) if r.signal(s).direction != "FLAT"]
    recent_hits = sum((r.signal(s).direction == "UP") == (r.target[s] > 0) for s in recent)
    upcoming = cursor + 1
    signal = r.signal(upcoming)
    now = r.time_of(cursor)
    context = DecisionContext(run_id=f"replay-{upcoming}", timestamp=now, config={}, payload={
        "signal": {"direction": signal.direction, "confidence": signal.confidence},
        "last_action_at": now - timedelta(hours=2)})
    tracker = DecisionTracker()
    result, _ = r.tree.run(context, tracker)

    # Model health: tuning, gate and current score for every combo, including the rejected ones.
    current = {rec.name: (rec, score_combo(rec, demo.SCORING)) for rec in r.records(upcoming)}
    combos = []
    for name, verdict in r.pipeline["verdicts"].items():
        search = r.pipeline["searches"][name]
        rec, score = current.get(name, (None, None))
        combos.append({
            "name": name, "features": demo.COMBOS[name].stop - demo.COMBOS[name].start,
            "trials": search["trials"], "best_trial": search["best_trial"], "validation_edge": round(search["best_value"], 3),
            "in_sample": round(verdict.in_sample, 3), "out_of_sample": round(verdict.out_of_sample, 3),
            "gate": "pass" if verdict.passed else "reject", "gate_reasons": verdict.reasons,
            "graded": rec.n if rec else 0, "hits": rec.hits if rec else 0,
            "score": round(score.score, 3) if score else None, "hit_floor": round(score.hit_floor, 3) if score else None,
            "status": ("admitted" if score.admitted else "not trusted yet") if score else "gated out",
            "status_reason": score.reason if score else "; ".join(verdict.reasons),
        })
    rec, score = current["compact-new"]
    combos.append({"name": "compact-new", "features": 3, "trials": None, "best_trial": None, "validation_edge": None,
                   "in_sample": None, "out_of_sample": None, "gate": "pass", "gate_reasons": [],
                   "graded": rec.n, "hits": rec.hits, "score": round(score.score, 3),
                   "hit_floor": round(score.hit_floor, 3),
                   "status": "admitted" if score.admitted else "not trusted yet", "status_reason": score.reason})

    return {
        "meta": {"target": TARGET_NAME, "interval": INTERVAL, "synthetic": True, "cursor": cursor,
                 "as_of": _iso(now), "replay_seconds_per_bar": SECONDS_PER_BAR},
        "cards": {
            "signal": {"direction": signal.direction, "confidence": round(signal.confidence, 3),
                       "forecast_pct": round((float(np.exp(signal.forecast * RETURN_SCALE)) - 1) * 100, 3),
                       "members": signal.members, "for_bar": _iso(r.time_of(upcoming))},
            "recent_calls": {"hits": int(recent_hits), "n": len(recent), "window": 50},
            "combos": {"admitted": sum(c["status"] == "admitted" for c in combos), "total": len(combos),
                       "gated_out": sum(c["gate"] == "reject" for c in combos)},
            "decision": {"outcome": result.label, "trail": [
                {"path": " / ".join(rec.path), "decision": rec.decision, "outcome": rec.outcome, "detail": rec.detail}
                for rec in tracker.records()]},
        },
        "series": series,
        "rows": rows,
        "combos": combos,
        "thresholds": {"min_score": demo.SCORING.min_score, "min_hit_floor": demo.SCORING.min_hit_floor},
    }
