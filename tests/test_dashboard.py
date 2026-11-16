"""Demo dashboard: pages render, the snapshot is deterministic, and the replay never looks ahead."""
import pytest

from mlsystem.dashboard import create_app, demo_data
from mlsystem.quality import score_combo
from mlsystem.quality.examples import pipeline_demo


@pytest.fixture(scope="module")
def client():
    return create_app().test_client()


@pytest.mark.parametrize("path, marker", [("/", "Prediction: DEMO"), ("/model-health", "Model Health")])
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
    assert len(series["time"]) == len(series["actual"]) == len(series["predicted"]) == demo_data.CHART_BARS
    assert series["time"] == sorted(series["time"])
    assert {row["model"] for row in snap["rows"]} == {"compact", "medium", "wide"}
    assert snap["cards"]["signal"]["direction"] in {"UP", "DOWN", "FLAT"}
    assert snap["cards"]["decision"]["outcome"] in {"act", "skip"}
    statuses = {combo["name"]: combo["status"] for combo in snap["combos"]}
    assert statuses["noise-only"] == "gated out" and statuses["compact-new"] == "not trusted yet"


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
