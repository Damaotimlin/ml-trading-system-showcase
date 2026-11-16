"""
Synthetic data behind the demo dashboard.

The seeded pipeline demo is run once; its combos then forecast a "live" stretch
of the synthetic series. Each combo forecasts several horizons ahead (30
minutes to 6.5 hours) with one directly fitted model per horizon, so the page can draw where the
ensemble expects the series to go, beyond the last bar that has actually closed.

Two rules keep the replay honest, the same ones the real system follows:

- a combo's weight at any moment comes only from predictions already graded then;
- a model for horizon h is never trained on a label that ends after its training
  window, so no future bar leaks into a forecast.

``snapshot(cursor)`` returns everything the pages show for one moment of the
replay. It is a pure function of ``cursor``, which keeps the views testable.
"""
from __future__ import annotations

import time
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
from zoneinfo import ZoneInfo

from mlsystem.decision_engine import DecisionContext, DecisionTracker, DecisionTreeBuilder, load_spec_file
from mlsystem.decision_engine.examples.example_nodes import example_registry
from mlsystem.quality import ComboRecord, Fold, Member, Signal, build_signal, score_combo, walk_forward_splits
from mlsystem.quality.examples import pipeline_demo as demo

TARGET_NAME = "SPY – Close"                                  # label only: the prices below are synthetic
INTERVAL = "30m"
BAR_MINUTES = 30
ET = ZoneInfo("America/New_York")
PRE_MARKET, REGULAR, POST_MARKET, CLOSE = (4, 0), (9, 30), (16, 0), (20, 0)      # session boundaries, ET
MIDPOINT = (12, 30)
SESSION_BARS = 32                                            # 30-minute bars from 04:00 to 20:00 ET
LAST_BAR = datetime(2026, 8, 28, 20, 0)                      # the synthetic series ends at this close (ET)
MARKET_HOLIDAYS = {date(2026, 4, 3), date(2026, 5, 25), date(2026, 6, 19), date(2026, 7, 3), date(2026, 9, 7)}
BASE_PRICE = 700.0
RETURN_SCALE = 0.0008                                        # synthetic units -> fractional price move per bar
HORIZONS = (1, 2, 4, 7, 13)                                  # bars ahead each combo forecasts (30 min .. 6.5 hours)
TREND_HORIZON = HORIZONS[-1]                                 # the "future trend" the cards and the tree act on
GRADED_HORIZON = 4                                           # earlier trend calls graded on the card (2 hours ahead)
TABLE_HORIZONS = (1, 4, 13)
# A lock freezes the ensemble's forecast path at a fixed time of day; it is never redrawn afterwards.
LOCKS = {"premarket": (PRE_MARKET, 13), "regular": (REGULAR, 13), "midpoint": (MIDPOINT, 7)}   # (time, bars covered)
WARM_UP = 100                                                # graded predictions before the replay window opens
CHART_BARS = 5 * SESSION_BARS                                # five trading days of history
TABLE_BARS = 4
NEW_COMBO_HISTORY = 8                                        # "compact-new" has only this many graded predictions
SECONDS_PER_BAR = 3                                          # replay speed of the auto-updating page


def trading_days() -> list[date]:
    days, day = [], date(2026, 1, 2)
    while day <= LAST_BAR.date() + timedelta(days=14):
        if day.weekday() < 5 and day not in MARKET_HOLIDAYS:
            days.append(day)
        day += timedelta(days=1)
    return days


def session_clock(n_steps: int) -> list[datetime]:
    """Bar-close times (ET) for ``n_steps`` bars plus a day ahead, on trading days only.

    The series is synthetic; only its timestamps follow the real calendar, so that the page reads
    like a recent stretch of trading days: weekdays, 04:00-20:00 ET, nights and weekends skipped.
    """
    bars = [datetime(d.year, d.month, d.day, *PRE_MARKET) + timedelta(minutes=BAR_MINUTES) * k
            for d in trading_days() for k in range(1, SESSION_BARS + 1)]
    end = bars.index(LAST_BAR)
    return bars[end - (n_steps - 1):end + SESSION_BARS + 2]


def training_rows(fold: Fold, horizon: int) -> slice:
    """Rows usable to train a ``horizon``-step model on ``fold``.

    The label of row t covers steps t .. t+horizon-1, so the last ``horizon - 1`` rows of the
    window are dropped: their labels would reach past the end of the training data.
    """
    return slice(fold.train.start, fold.train.stop - (horizon - 1))


class Replay:
    """Everything that does not depend on the cursor, computed once."""

    def __init__(self) -> None:
        self.pipeline = demo.run(verbose=False)
        self._signals: dict[tuple[int, int], Signal] = {}
        rng = np.random.default_rng(demo.SEED)
        features, self.target = demo.make_data(rng)
        self.price = BASE_PRICE * np.exp(np.cumsum(self.target * RETURN_SCALE))
        self.clock = session_clock(demo.N_SAMPLES)
        self.day_number = {day: i for i, day in enumerate(trading_days())}

        folds = walk_forward_splits(demo.N_SAMPLES - 1, n_folds=5, test_size=50,
                                    train_size=demo.TRAIN_WINDOW, embargo=demo.EMBARGO)
        self.first = folds[0].test.start                     # first step of the live stretch
        self.last = folds[-1].test.stop - 1
        self.survivors = [name for name, verdict in self.pipeline["verdicts"].items() if verdict.passed]
        # forecast[name][h][k]: move expected over steps first+k .. first+k+h-1, known before step first+k.
        cumulative = np.concatenate([[0.0], np.cumsum(self.target)])
        self.forecast, self.hit = {}, {}
        for name in self.survivors:
            x, alpha = features[:, demo.COMBOS[name]], self.pipeline["tuned"][name]
            self.forecast[name] = {}
            for h in HORIZONS:
                label = cumulative[h:] - cumulative[:-h]             # label[t] = sum of target[t .. t+h-1]
                parts = []
                for fold in folds:
                    rows = training_rows(fold, h)
                    parts.append(x[fold.test] @ demo.fit_ridge(x[rows], label[rows], alpha))
                self.forecast[name][h] = np.concatenate(parts)
            self.hit[name] = np.sign(self.forecast[name][1]) == np.sign(self.target[self.first:self.last + 1])
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

    def members(self, step: int, horizon: int = 1) -> list[Member]:
        """Admitted combos with their ``horizon``-step forecast made just before ``step``."""
        offset = step - self.first
        out = []
        for record in self.records(step):
            score = score_combo(record, demo.SCORING)
            if score.admitted:
                source = record.name.removesuffix("-new")
                out.append(Member(record.name, float(self.forecast[source][horizon][offset]), score.score))
        return out

    def signal(self, step: int, horizon: int = 1) -> Signal:
        key = (step, horizon)
        if key not in self._signals:
            # The flat band widens with the horizon like the noise does (square root of time).
            self._signals[key] = build_signal(self.members(step, horizon), demo.FLAT_BAND * horizon ** 0.5)
        return self._signals[key]

    def skill_by_horizon(self, cursor: int) -> list[dict]:
        """How the ensemble's calls at each horizon turned out, counting only calls whose outcome is known."""
        out = []
        for h in HORIZONS:
            results = []
            for origin in range(self.first + WARM_UP // 2, cursor - h + 2):     # target bar origin+h-1 <= cursor
                call = self.signal(origin, h)
                if call.direction != "FLAT":
                    results.append((call.direction == "UP") == (self.actual_move(origin, h) > 0))
            out.append({"bars_ahead": h, "hits": int(sum(results)), "n": len(results)})
        return out

    def predicted_price(self, step: int, horizon: int) -> float | None:
        """Ensemble price for bar ``step + horizon - 1``, forecast before ``step``; None if nobody may vote."""
        signal = self.signal(step, horizon)
        if not signal.members:
            return None
        return float(self.price[step - 1] * np.exp(signal.forecast * RETURN_SCALE))

    def actual_move(self, step: int, horizon: int) -> float:
        return float(self.target[step:step + horizon].sum())

    def average_prediction(self, target_bar: int, latest_origin: int) -> float | None:
        """AvgPred: mean of every ensemble forecast for ``target_bar`` made at or before ``latest_origin``."""
        made = [self.predicted_price(target_bar - h + 1, h) for h in HORIZONS
                if self.first < target_bar - h + 1 <= latest_origin]
        made = [value for value in made if value is not None]
        return sum(made) / len(made) if made else None

    def time_of(self, step: int) -> datetime:
        return self.clock[step]

    def x_of(self, moment: datetime) -> int:
        """Chart x in minutes, with non-trading days removed (nights stay, as on the real chart)."""
        return self.day_number[moment.date()] * 1440 + moment.hour * 60 + moment.minute

    def lock(self, kind: str, origin: int) -> dict:
        """The forecast path frozen just before step ``origin``: start price, then one point per horizon."""
        _, span = LOCKS[kind]
        start = float(self.price[origin - 1])
        xs, ys = [self.x_of(self.time_of(origin)) - BAR_MINUTES], [round(start, 2)]
        for h in (h for h in HORIZONS if h <= span):
            value = self.predicted_price(origin, h)
            if value is None:
                return {}
            xs.append(self.x_of(self.time_of(origin + h - 1)))
            ys.append(round(value, 2))
        band = float(np.exp(RETURN_SCALE * demo.FLAT_BAND * span ** 0.5))     # inside this, the call is FLAT
        return {"kind": kind, "x": xs, "y": ys, "flat": [round(start / band, 2), round(start * band, 2)],
                "direction": self.signal(origin, span).direction}


@lru_cache(maxsize=1)
def replay() -> Replay:
    return Replay()


def cursor_now(now: float | None = None) -> int:
    """Replay position for wall-clock ``now``: one bar every few seconds, looping."""
    r = replay()
    span = r.last - (r.first + WARM_UP)
    return r.first + WARM_UP + int((time.time() if now is None else now) // SECONDS_PER_BAR) % span


def _iso(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%d %H:%M")                # Eastern Time, like the real dashboard


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 2)


def snapshot(cursor: int | None = None) -> dict:
    """All data for the pages at replay position ``cursor`` (the latest *completed* bar)."""
    r = replay()
    cursor = cursor_now() if cursor is None else max(r.first + WARM_UP, min(int(cursor), r.last - 1))

    upcoming = cursor + 1                                     # first bar that has not closed yet
    now = r.time_of(cursor)

    # Chart, laid out like the real one: the live line up to "now"; AvgPred, the average of every forecast
    # for a bar, running on past the live line; the paths locked at fixed times of day with their flat
    # zones; and the individual forecasts whose target bar has not closed yet.
    past = range(max(r.first + TREND_HORIZON, cursor - CHART_BARS + 1), cursor + 1)
    future = range(cursor + 1, cursor + TREND_HORIZON + 1)
    bars = [*past, *future]
    x = {bar: r.x_of(r.time_of(bar)) for bar in bars}
    open_predictions = []
    for origin in range(upcoming - TREND_HORIZON + 1, upcoming + 1):
        for h in HORIZONS:
            target_bar = origin + h - 1
            value = r.predicted_price(origin, h) if target_bar > cursor and origin - 1 in x else None
            if value is not None:
                open_predictions.append({"x_from": x[origin - 1], "y_from": round(float(r.price[origin - 1]), 2),
                                         "x": x[target_bar], "y": round(value, 2), "bars_ahead": h})
    lock_times = {(hour, minute): kind for kind, ((hour, minute), _) in LOCKS.items()}
    locks = []
    for bar in past:
        opened = r.time_of(bar) - timedelta(minutes=BAR_MINUTES)
        kind = lock_times.get((opened.hour, opened.minute))
        if kind and bar <= upcoming and bar - 1 in x and (lock := r.lock(kind, bar)):
            locks.append(lock)
    days = []
    for day in sorted({r.time_of(bar).date() for bar in bars}):
        at = lambda hm: r.x_of(datetime(day.year, day.month, day.day, *hm))
        days.append({"label": day.strftime("%m/%d"), "midnight": at((0, 0)), "premarket": [at(PRE_MARKET), at(REGULAR)],
                     "regular": [at(REGULAR), at(POST_MARKET)], "postmarket": [at(POST_MARKET), at(CLOSE)],
                     "midpoint": at(MIDPOINT)})
    series = {
        "x": [x[bar] for bar in bars],
        "time": [_iso(r.time_of(bar)) for bar in bars],
        "actual": [round(float(r.price[bar]), 2) for bar in past] + [None] * len(future),
        "avg_pred": [_round(r.average_prediction(bar, upcoming)) for bar in bars],
        "now_index": len(past) - 1,
        "days": days, "locks": locks, "open_predictions": open_predictions,
        "horizons": list(HORIZONS), "bar_minutes": BAR_MINUTES,
    }

    # Table: the latest predictions that can be graded, one row per combo, horizon and target bar.
    rows = []
    for target_bar in range(cursor, cursor - TABLE_BARS, -1):
        for h in TABLE_HORIZONS:
            origin = target_bar - h + 1                       # forecast made just before this step
            scores = {rec.name: score_combo(rec, demo.SCORING) for rec in r.records(origin)}
            for name in r.survivors:
                forecast = float(r.forecast[name][h][origin - r.first])
                live, actual = float(r.price[origin - 1]), float(r.price[target_bar])
                predicted = live * float(np.exp(forecast * RETURN_SCALE))
                rows.append({
                    "live_time": _iso(r.time_of(origin - 1)), "live_value": round(live, 3),
                    "predict_time": _iso(r.time_of(target_bar)), "bars_ahead": h,
                    "predicted_value": round(predicted, 3), "diff_pct": round((predicted / live - 1) * 100, 3),
                    "interval": INTERVAL, "model": name,
                    "combo_score": round(scores[name].score, 3), "in_ensemble": scores[name].admitted,
                    "error_pct": round((predicted / actual - 1) * 100, 3),
                    "hit": bool((forecast > 0) == (r.actual_move(origin, h) > 0)),
                })

    # Cards: the trend the ensemble expects from here, and how its earlier trend calls turned out.
    signal = r.signal(upcoming, TREND_HORIZON)
    graded = [(r.signal(s - GRADED_HORIZON + 1, GRADED_HORIZON), r.actual_move(s - GRADED_HORIZON + 1, GRADED_HORIZON))
              for s in range(cursor - 49, cursor + 1)]
    calls = [(sig.direction == "UP") == (move > 0) for sig, move in graded if sig.direction != "FLAT"]
    now = now.replace(tzinfo=ET)
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
                       "members": signal.members, "bars_ahead": TREND_HORIZON,
                       "until": _iso(r.time_of(cursor + TREND_HORIZON))},
            "recent_calls": {"hits": int(sum(calls)), "n": len(calls), "window": 50, "bars_ahead": GRADED_HORIZON},
            "combos": {"admitted": sum(c["status"] == "admitted" for c in combos), "total": len(combos),
                       "gated_out": sum(c["gate"] == "reject" for c in combos)},
            "decision": {"outcome": result.label, "trail": [
                {"path": " / ".join(rec.path), "decision": rec.decision, "outcome": rec.outcome, "detail": rec.detail}
                for rec in tracker.records()]},
        },
        "series": series,
        "rows": rows,
        "combos": combos,
        "horizon_skill": r.skill_by_horizon(cursor),
        "thresholds": {"min_score": demo.SCORING.min_score, "min_hit_floor": demo.SCORING.min_hit_floor},
    }
