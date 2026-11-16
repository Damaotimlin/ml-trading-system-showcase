from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from mlsystem.decision_engine import (
    DecisionContext, DecisionStatus, DecisionTracker, DecisionTreeBuilder, DecisionTreeSpec,
    NodeRegistrationError, NodeRegistry, load_spec_file,
)
from mlsystem.decision_engine.examples.example_nodes import SignalConfidence, example_registry

SPEC = Path(__file__).resolve().parents[1] / "mlsystem/decision_engine/examples/example_tree.yaml"
NOW = datetime(2026, 1, 5, 15, 0, tzinfo=timezone.utc)


def run(confidence, last_action_at=None):
    tree = DecisionTreeBuilder(example_registry).build(load_spec_file(SPEC))
    ctx = DecisionContext(run_id="t", timestamp=NOW, config={},
                          payload={"signal": {"direction": "DOWN", "confidence": confidence},
                                   "last_action_at": last_action_at})
    tracker = DecisionTracker()
    result, ctx = tree.run(ctx, tracker)
    return result, ctx, tracker


def test_confident_signal_without_recent_action_acts():
    result, ctx, tracker = run(0.8, NOW - timedelta(hours=1))
    assert result.label == "act" and result.status is DecisionStatus.PASS
    assert ctx.payload["intent"] == "DOWN"
    assert [r.decision for r in tracker.records()] == ["Confidence", "Cooldown", "Intent"]


def test_low_confidence_skips_without_checking_cooldown():
    result, ctx, tracker = run(0.4)
    assert result.label == "skip" and "intent" not in ctx.payload
    assert [r.decision for r in tracker.records()] == ["Confidence", "Intent"]


def test_recent_action_is_held_by_cooldown():
    result, _, _ = run(0.9, NOW - timedelta(minutes=5))
    assert result.label == "skip"


def test_context_is_immutable_between_nodes():
    _, ctx, _ = run(0.8)
    original = DecisionContext(run_id="t", timestamp=NOW, config={}, payload={})
    assert original.with_payload(x=1).payload == {"x": 1} and original.payload == {}
    assert "intent" in ctx.payload


def test_spec_requires_start_and_known_nodes():
    with pytest.raises(ValueError):
        DecisionTreeSpec.from_dict({"nodes": [{"key": "a", "id": "signal_confidence"}]})
    spec = DecisionTreeSpec.from_dict({"start": "a", "nodes": [{"key": "a", "id": "not_registered"}]})
    with pytest.raises(NodeRegistrationError):
        DecisionTreeBuilder(example_registry).build(spec)


def test_registry_rejects_a_different_class_under_the_same_name():
    reg = NodeRegistry()
    reg.add("gate", SignalConfidence)
    reg.add("gate", SignalConfidence)  # same class again is fine
    with pytest.raises(NodeRegistrationError):
        reg.add("gate", type("OtherGate", (SignalConfidence,), {}))
