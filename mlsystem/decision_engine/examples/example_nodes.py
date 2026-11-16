"""
Toy nodes for the decision-engine example.

They use their own registry so they never mix with any other node library.
"""
from __future__ import annotations

from datetime import timedelta

from ..context import DecisionContext
from ..nodes import DecisionResult, DecisionStatus, JudgeNode, TaskNode
from ..registry import NodeRegistry
from ..tracking import DecisionTracker

example_registry = NodeRegistry()


@example_registry.register("signal_confidence")
class SignalConfidence(JudgeNode):
    """Pass when the incoming signal's confidence reaches ``min_confidence``."""

    def run(self, context: DecisionContext, tracker: DecisionTracker) -> DecisionResult:
        confidence = float(context.payload.get("signal", {}).get("confidence", 0.0))
        threshold = float(self.config.get("min_confidence", 0.5))
        passed = confidence >= threshold
        tracker.add(["Example", "Signal"], "Confidence", "PASS" if passed else "FAIL",
                    detail=f"{confidence:.2f} vs min {threshold:.2f}")
        return DecisionResult(DecisionStatus.PASS if passed else DecisionStatus.FAIL, "confidence")


@example_registry.register("cooldown_guard")
class CooldownGuard(JudgeNode):
    """Fail while the previous action is more recent than ``cooldown_minutes``."""

    def run(self, context: DecisionContext, tracker: DecisionTracker) -> DecisionResult:
        last_action = context.payload.get("last_action_at")
        cooldown = timedelta(minutes=int(self.config.get("cooldown_minutes", 30)))
        waiting = last_action is not None and context.timestamp - last_action < cooldown
        tracker.add(["Example", "Guards"], "Cooldown", "FAIL" if waiting else "PASS",
                    detail=f"cooldown {cooldown}")
        return DecisionResult(DecisionStatus.FAIL if waiting else DecisionStatus.PASS, "cooldown")


@example_registry.register("record_intent")
class RecordIntent(TaskNode):
    """Terminal action: record the intent to act (a real node would call a service here)."""

    def run(self, context: DecisionContext, tracker: DecisionTracker):
        direction = context.payload.get("signal", {}).get("direction", "UP")
        tracker.add(["Example", "Action"], "Intent", direction)
        return DecisionResult(DecisionStatus.PASS, "act", data={"direction": direction}), \
            context.with_payload(intent=direction)


@example_registry.register("record_skip")
class RecordSkip(TaskNode):
    """Terminal action: record that this cycle does nothing."""

    def run(self, context: DecisionContext, tracker: DecisionTracker) -> DecisionResult:
        tracker.add(["Example", "Action"], "Intent", "NONE")
        return DecisionResult(DecisionStatus.SKIP, "skip")
