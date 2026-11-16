"""Quality control: walk-forward splits, generalization gate, convergence, scoring, ensemble."""
import pytest

from mlsystem.quality import (ComboRecord, ConvergenceTracker, FoldResult, GatePolicy, GeneralizationGate, Member,
                              ScorePolicy, build_signal, score_combo, select_combos, shrunk_score, staleness_decay,
                              walk_forward_splits, wilson_lower_bound)
from mlsystem.quality.examples import pipeline_demo

GATE = GeneralizationGate(GatePolicy(min_out_of_sample=0.02, min_positive_folds=0.6, min_retention=0.4))
POLICY = ScorePolicy(credibility_k=40, min_score=0.08, min_hit_floor=0.5)


# ---- walk-forward splits -------------------------------------------------

@pytest.mark.parametrize("train_size", [None, 100])
def test_folds_are_chronological_with_embargo(train_size):
    folds = walk_forward_splits(500, n_folds=4, test_size=50, train_size=train_size, embargo=5)
    assert [f.test.start for f in folds] == [300, 350, 400, 450]
    assert folds[-1].test.stop == 500
    for fold in folds:
        assert fold.train.stop + 5 == fold.test.start          # nothing from the embargo or the future in train
        assert fold.train.start == (0 if train_size is None else fold.train.stop - 100)


def test_folds_reject_too_little_history():
    with pytest.raises(ValueError):
        walk_forward_splits(200, n_folds=4, test_size=50, train_size=100)
    with pytest.raises(ValueError):
        walk_forward_splits(200, n_folds=0, test_size=50)


# ---- generalization gate -------------------------------------------------

def test_gate_passes_a_candidate_that_holds_up_out_of_sample():
    verdict = GATE.evaluate([FoldResult(0.12, 0.10), FoldResult(0.11, 0.09), FoldResult(0.13, 0.11)])
    assert verdict.passed and verdict.reasons == []


def test_gate_rejects_overfit_candidate_despite_strong_in_sample():
    verdict = GATE.evaluate([FoldResult(0.40, 0.05), FoldResult(0.42, 0.04), FoldResult(0.38, 0.06)])
    assert not verdict.passed
    assert any("in-sample skill" in reason for reason in verdict.reasons)


def test_gate_rejects_inconsistent_and_missing_evidence():
    lucky_once = GATE.evaluate([FoldResult(0.1, 0.5), FoldResult(0.1, -0.05), FoldResult(0.1, -0.05)])
    assert not lucky_once.passed and any("folds positive" in reason for reason in lucky_once.reasons)
    assert not GATE.evaluate([]).passed


# ---- convergence ---------------------------------------------------------

def test_tracker_stops_after_patience_trials_without_improvement():
    tracker = ConvergenceTracker(patience=3, min_improvement=0.05)
    assert tracker.record(0, 1.00)
    assert not tracker.record(1, 1.01)        # +1% is below the 5% bar
    assert tracker.record(2, 1.10)
    for trial in (3, 4):
        tracker.record(trial, 0.5)
    assert not tracker.plateaued
    tracker.record(5, 1.12)                   # still not a meaningful gain
    assert tracker.plateaued
    assert tracker.summary()["best_trial"] == 2 and tracker.summary()["improvements"] == 2


def test_a_new_best_resets_the_plateau():
    tracker = ConvergenceTracker(patience=2)
    for trial, value in enumerate([1.0, 0.9, 0.9]):
        tracker.record(trial, value)
    assert tracker.plateaued
    tracker.record(3, 2.0)
    assert not tracker.plateaued


# ---- scoring -------------------------------------------------------------

def test_wilson_lower_bound_known_values():
    assert wilson_lower_bound(0, 0) == 0.0
    assert wilson_lower_bound(8, 10) == pytest.approx(0.4902, abs=1e-4)
    assert wilson_lower_bound(80, 100) == pytest.approx(0.7112, abs=1e-4)
    assert wilson_lower_bound(8, 10) < wilson_lower_bound(80, 100) < 0.8     # same rate, more evidence


def test_shrinkage_and_staleness():
    assert shrunk_score(0.5, 0, k=10) == 0.0
    assert shrunk_score(0.5, 10, k=10) == pytest.approx(0.25)                # n == k keeps half
    assert shrunk_score(0.5, 990, k=10) == pytest.approx(0.495)
    assert staleness_decay(10, grace_days=30, half_life_days=15) == 1.0
    assert staleness_decay(45, grace_days=30, half_life_days=15) == pytest.approx(0.5)


def test_small_sample_cannot_outrank_a_proven_combo():
    proven = ComboRecord("proven", hits=150, n=250, mean_points=0.20)
    lucky = ComboRecord("lucky", hits=6, n=6, mean_points=1.00)            # perfect, but six predictions
    ranked = select_combos([lucky, proven], POLICY)
    assert [s.name for s in ranked] == ["proven", "lucky"]
    assert ranked[0].admitted
    assert not score_combo(ComboRecord("thin", 5, 8, 0.25), POLICY).admitted


def test_stale_score_fades_out():
    fresh = ComboRecord("combo", 150, 250, 0.20)
    stale = ComboRecord("combo", 150, 250, 0.20, stale_days=90)
    assert score_combo(fresh, POLICY).admitted
    assert not score_combo(stale, POLICY).admitted


# ---- ensemble ------------------------------------------------------------

def test_signal_direction_confidence_and_flat_band():
    agree = build_signal([Member("a", 0.4, 2.0), Member("b", 0.2, 1.0)], flat_band=0.05)
    assert (agree.direction, agree.confidence) == ("UP", 1.0)
    assert agree.forecast == pytest.approx(1.0 / 3)

    split = build_signal([Member("a", -0.9, 1.0), Member("b", 0.1, 1.0), Member("c", 0.1, 1.0)], flat_band=0.05)
    assert split.direction == "DOWN" and split.confidence == pytest.approx(1 / 3)

    assert build_signal([Member("a", 0.03, 1.0), Member("b", -0.01, 1.0)], flat_band=0.05).direction == "FLAT"
    assert build_signal([], flat_band=0.05) == build_signal([Member("a", 1.0, 0.0)], flat_band=0.05)


# ---- end to end ----------------------------------------------------------

def test_pipeline_demo_is_reproducible_and_filters_what_it_should():
    first, second = pipeline_demo.run(verbose=False), pipeline_demo.run(verbose=False)
    assert first["signal"] == second["signal"] and first["tuned"] == second["tuned"]

    verdicts = first["verdicts"]
    assert verdicts["compact"].passed                     # the combo that sees only real information
    assert not verdicts["noise-only"].passed              # the combo with no information is gated out
    scores = {s.name: s for s in first["scores"]}
    assert scores["compact"].admitted
    assert 0 < scores["compact-new"].score < scores["compact"].score   # same model, less evidence, less weight
    assert not scores["medium"].admitted                               # passing the gate is not enough on its own
    assert first["signal"].direction in {"UP", "DOWN", "FLAT"}
