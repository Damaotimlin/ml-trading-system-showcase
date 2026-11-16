"""
Tune -> gate -> score -> ensemble -> decide, end to end on synthetic data:

    python -m mlsystem.quality.examples.pipeline_demo

Everything here is generated from a fixed random seed: a noisy series whose next
value depends weakly on 3 of 40 features. Four ridge-regression "combos" stand
in for the neural networks of the full system; what the demo shows is the
quality-control path around them, which is the same shape as the real one.
No market data, strategy or production threshold is involved.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from mlsystem.decision_engine import DecisionContext, DecisionTracker, DecisionTreeBuilder, load_spec_file
from mlsystem.decision_engine.examples.example_nodes import example_registry
from mlsystem.quality import (ComboRecord, ConvergenceTracker, FoldResult, GatePolicy, GeneralizationGate, Member,
                              ScorePolicy, build_signal, select_combos, walk_forward_splits)

SEED = 7
N_SAMPLES, N_FEATURES = 1400, 40
TUNE_END, GATE_END = 800, 1100          # tune | gate (never seen while tuning) | live track record
TRAIN_WINDOW, EMBARGO = 120, 2
COMBOS = {                              # name -> feature columns the combo may use
    "compact": slice(0, 3),             # exactly the informative features
    "medium": slice(0, 12),             # informative + some noise
    "wide": slice(0, 40),               # everything: easy to overfit on a short window
    "noise-only": slice(10, 40),        # no real information at all
}
GATE = GatePolicy(min_out_of_sample=0.02, min_positive_folds=0.6, min_retention=0.4)
SCORING = ScorePolicy(credibility_k=40, min_score=0.08, min_hit_floor=0.5)
FLAT_BAND = 0.05


def make_data(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    features = rng.standard_normal((N_SAMPLES, N_FEATURES))
    target = features[:, :3] @ np.array([0.30, 0.22, -0.18]) + rng.standard_normal(N_SAMPLES)
    return features, target


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    return np.linalg.solve(x.T @ x + alpha * np.eye(x.shape[1]), x.T @ y)


def edge(weights: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    """Directional skill: share of correct signs minus the 50% a coin would get."""
    return float(np.mean(np.sign(x @ weights) == np.sign(y)) - 0.5)


def evaluate(x: np.ndarray, y: np.ndarray, alpha: float, folds) -> list[FoldResult]:
    results = []
    for fold in folds:
        weights = fit_ridge(x[fold.train], y[fold.train], alpha)
        results.append(FoldResult(edge(weights, x[fold.train], y[fold.train]), edge(weights, x[fold.test], y[fold.test])))
    return results


def tune(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, max_trials: int = 80) -> tuple[float, ConvergenceTracker]:
    """Random search over the ridge penalty, stopped as soon as it plateaus."""
    folds = walk_forward_splits(TUNE_END, n_folds=5, test_size=60, train_size=TRAIN_WINDOW, embargo=EMBARGO)
    tracker, best_alpha = ConvergenceTracker(patience=15, min_improvement=0.02), 1.0
    for trial in range(max_trials):
        alpha = float(10 ** rng.uniform(-2, 3))
        value = float(np.mean([r.out_of_sample for r in evaluate(x[:TUNE_END], y[:TUNE_END], alpha, folds)]))
        if tracker.record(trial, value):
            best_alpha = alpha
        if tracker.plateaued:
            break
    return best_alpha, tracker


def run(verbose: bool = True) -> dict:
    say = print if verbose else (lambda *a, **k: None)
    rng = np.random.default_rng(SEED)
    features, target = make_data(rng)

    say("1) Tune each combo (random search, stops on plateau)")
    tuned, searches = {}, {}
    for name, columns in COMBOS.items():
        alpha, tracker = tune(features[:, columns], target, rng)
        tuned[name], searches[name] = alpha, tracker.summary()
        say(f"   {name:<11} alpha={alpha:8.3f}  best validation edge {tracker.best_value:+.3f}"
            f"  stopped after {tracker.trials_seen} of 80 trials")

    say("\n2) Generalization gate (walk-forward on data the tuner never saw)")
    gate_folds = walk_forward_splits(GATE_END, n_folds=6, test_size=50, train_size=TRAIN_WINDOW, embargo=EMBARGO)
    gate, verdicts = GeneralizationGate(GATE), {}
    for name, columns in COMBOS.items():
        verdict = verdicts[name] = gate.evaluate(evaluate(features[:, columns], target, tuned[name], gate_folds))
        say(f"   {name:<11} in-sample {verdict.in_sample:+.3f}  out-of-sample {verdict.out_of_sample:+.3f}"
            f"  -> {'PASS' if verdict.passed else 'REJECT: ' + '; '.join(verdict.reasons)}")
    survivors = [name for name, verdict in verdicts.items() if verdict.passed]

    say("\n3) Grade the survivors on a live-style track record, then score them")
    live_folds = walk_forward_splits(N_SAMPLES - 1, n_folds=5, test_size=50, train_size=TRAIN_WINDOW, embargo=EMBARGO)
    records = []
    for name in survivors:
        x = features[:, COMBOS[name]]
        hits = np.concatenate([np.sign(x[f.test] @ fit_ridge(x[f.train], target[f.train], tuned[name])) == np.sign(target[f.test])
                               for f in live_folds])
        records.append(ComboRecord(name, int(hits.sum()), hits.size, float(2 * hits.mean() - 1)))
        if name == "compact":   # the same model as a newly added combo: only its last 8 predictions are graded
            recent = hits[-8:]
            records.append(ComboRecord("compact-new", int(recent.sum()), recent.size, float(2 * recent.mean() - 1)))
    scores = select_combos(records, SCORING)
    by_name = {r.name: r for r in records}
    for s in scores:
        r = by_name[s.name]
        say(f"   {s.name:<11} {r.hits:>3}/{r.n:<3} hits  raw {r.mean_points:+.3f}  shrunk {s.score:+.3f}"
            f"  hit-rate floor {s.hit_floor:.1%}  -> {s.reason}")

    say("\n4) Ensemble the admitted combos into one signal for the next step")
    last = N_SAMPLES - 1
    train = slice(last - EMBARGO - TRAIN_WINDOW, last - EMBARGO)
    members = []
    for s in scores:
        if s.admitted:
            x = features[:, COMBOS[s.name.removesuffix("-new")]]
            forecast = float(x[last] @ fit_ridge(x[train], target[train], tuned[s.name.removesuffix("-new")]))
            members.append(Member(s.name, forecast, s.score))
            say(f"   {s.name:<11} forecast {forecast:+.3f}  weight {s.score:.3f}")
    signal = build_signal(members, FLAT_BAND)
    say(f"   signal: {signal.direction}  confidence {signal.confidence:.2f}  forecast {signal.forecast:+.3f}"
        f"  (actual next value {target[last]:+.3f})")

    say("\n5) Hand the signal to the YAML decision tree")
    spec = load_spec_file(Path(__file__).parents[2] / "decision_engine" / "examples" / "example_tree.yaml")
    now = datetime.now(timezone.utc)
    context = DecisionContext(run_id="pipeline-demo", timestamp=now, config={}, payload={
        "signal": {"direction": signal.direction, "confidence": signal.confidence},
        "last_action_at": now - timedelta(hours=2)})
    tracker = DecisionTracker()
    result, context = DecisionTreeBuilder(example_registry).build(spec).run(context, tracker)
    if verbose:
        tracker.render(title="Pipeline demo", final_outcome=result.label, final_detail=f"intent={context.payload.get('intent')}")
    return {"tuned": tuned, "searches": searches, "verdicts": verdicts, "scores": scores, "signal": signal, "outcome": result.label}


if __name__ == "__main__":
    run()
