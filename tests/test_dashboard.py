"""Demo dashboard: pages render, the snapshot is deterministic, and the replay never looks ahead."""
import pytest

from mlsystem.dashboard import create_app, demo_data
from mlsystem.quality import score_combo, walk_forward_splits
from mlsystem.quality.examples import pipeline_demo


@pytest.fixture(scope="module")
def client():
    return create_app().test_client()


@pytest.mark.parametrize("path, marker", [("/", "Prediction: SPY"), ("/model-health", "Model Health")])
def test_pages_render_and_say_the_data_is_synthetic(client, path, marker):
    response = client.get(f"{path}?cursor=1300")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert marker in html and "Synthetic data." in html


def test_snapshot_api_is_deterministic_and_not_cached(client):
    first, second = client.get("/api/snapshot?cursor=1300"), client.get("/api/snapshot?cursor=1300")
    assert first.json == second.json
    assert first.headers["Cache-Control"] == "no-store"
    assert first.json["meta"]["synthetic"] is True and first.json["meta"]["cursor"] == 1300
    assert client.get("/healthz").json == {"status": "ok", "synthetic": True}


def test_snapshot_shape():
    snap = demo_data.snapshot(1300)
    series = snap["series"]
    assert len(series["x"]) == len(series["time"]) == len(series["actual"]) == len(series["avg_pred"])
    assert series["x"] == sorted(series["x"]) and series["time"] == sorted(series["time"])
    assert {row["model"] for row in snap["rows"]} == {"compact", "medium", "wide"}
    assert {row["bars_ahead"] for row in snap["rows"]} == set(demo_data.TABLE_HORIZONS)
    assert snap["cards"]["signal"]["direction"] in {"UP", "DOWN", "FLAT"}
    assert snap["cards"]["decision"]["outcome"] in {"act", "skip"}
    statuses = {combo["name"]: combo["status"] for combo in snap["combos"]}
    assert statuses["compact"] == "admitted" and statuses["noise-only"] == "gated out"
    assert [h["bars_ahead"] for h in snap["horizon_skill"]] == list(demo_data.HORIZONS)


def test_clock_follows_recent_trading_sessions():
    r = demo_data.replay()
    times = [r.time_of(step) for step in range(len(r.clock))]
    assert r.time_of(pipeline_demo.N_SAMPLES - 1) == demo_data.LAST_BAR
    assert times == sorted(times) and len(set(times)) == len(times)
    assert all(t.weekday() < 5 and t.date() not in demo_data.MARKET_HOLIDAYS for t in times)
    assert {(t.hour, t.minute) for t in times} <= {(h, m) for h in range(4, 21) for m in (0, 30)} - {(4, 0), (20, 30)}
    replayed = {r.time_of(step).strftime("%Y-%m") for step in range(r.first + demo_data.WARM_UP, r.last)}
    assert replayed == {"2026-08"}                                      # the whole replay sits in August 2026


def test_average_prediction_runs_past_the_last_live_bar():
    series = demo_data.snapshot(1300)["series"]
    now, ahead = series["now_index"], demo_data.TREND_HORIZON
    assert len(series["x"]) == now + 1 + ahead
    assert all(v is not None for v in series["actual"][:now + 1])
    assert series["actual"][now + 1:] == [None] * ahead                 # the future has no live values
    assert all(v is not None for v in series["avg_pred"][now + 1:])     # ...but it does have a prediction
    assert series["open_predictions"] and all(p["x"] > series["x"][now] for p in series["open_predictions"])


def test_locks_are_frozen_once_made():
    """A path locked earlier in the day must be identical in every later snapshot."""
    def locks(cursor):
        return {(lock["kind"], lock["x"][0]): lock for lock in demo_data.snapshot(cursor)["series"]["locks"]}
    early, late = locks(1340), locks(1352)
    shared = early.keys() & late.keys()
    assert len(shared) >= 3 and all(early[key] == late[key] for key in shared)
    assert {kind for kind, _ in late} == {"premarket", "regular", "midpoint"}
    for lock in late.values():
        assert lock["x"] == sorted(lock["x"]) and lock["flat"][0] < lock["y"][0] < lock["flat"][1]


def test_snapshot_never_reads_bars_after_the_cursor():
    """Everything shown at the cursor must be unchanged when all later outcomes are rewritten."""
    r, cursor = demo_data.replay(), 1300
    before = demo_data.snapshot(cursor)
    saved = r.target.copy(), r.price.copy(), {name: hits.copy() for name, hits in r.hit.items()}
    try:
        r.target[cursor + 1:] = -5.0
        r.price[cursor + 1:] = 1.0
        for hits in r.hit.values():
            hits[cursor + 1 - r.first:] = ~hits[cursor + 1 - r.first:]
        r._signals.clear()
        assert demo_data.snapshot(cursor) == before
    finally:
        r.target[:], r.price[:] = saved[0], saved[1]
        r.hit.update(saved[2])
        r._signals.clear()


def test_horizon_models_are_not_trained_on_labels_from_the_future():
    fold = walk_forward_splits(500, n_folds=2, test_size=50, train_size=120, embargo=2)[0]
    for horizon in demo_data.HORIZONS:
        rows = demo_data.training_rows(fold, horizon)
        last_label_step = rows.stop - 1 + horizon - 1                   # label of the last row ends here
        assert last_label_step < fold.train.stop <= fold.test.start - 2


def test_cursor_is_clamped_to_the_replay_window():
    r = demo_data.replay()
    assert demo_data.snapshot(0)["meta"]["cursor"] == r.first + demo_data.WARM_UP
    assert demo_data.snapshot(10**6)["meta"]["cursor"] == r.last - 1
    assert r.first + demo_data.WARM_UP <= demo_data.cursor_now(now=123456.0) < r.last


def test_scores_use_only_predictions_already_graded():
    """The weight a combo carries at step t must not change when later outcomes are altered."""
    r = demo_data.replay()
    step = 1300
    before = [score_combo(rec, pipeline_demo.SCORING) for rec in r.records(step)]
    original = {name: hits.copy() for name, hits in r.hit.items()}
    try:
        for hits in r.hit.values():
            hits[step - r.first:] = ~hits[step - r.first:]          # rewrite the future
        assert [score_combo(rec, pipeline_demo.SCORING) for rec in r.records(step)] == before
    finally:
        r.hit.update(original)
    assert all(rec.n <= step - r.first for rec in r.records(step))
