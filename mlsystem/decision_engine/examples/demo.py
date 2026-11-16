"""Run the example tree once and print the decision trail:  python -m mlsystem.decision_engine.examples.demo"""
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ..context import DecisionContext
from ..specs import load_spec_file
from ..tracking import DecisionTracker
from ..tree import DecisionTreeBuilder
from .example_nodes import example_registry


def main() -> None:
    spec = load_spec_file(Path(__file__).with_name("example_tree.yaml"))
    tree = DecisionTreeBuilder(example_registry).build(spec)
    now = datetime.now(timezone.utc)
    context = DecisionContext(
        run_id="demo-1",
        timestamp=now,
        config={},
        payload={"signal": {"direction": "UP", "confidence": 0.72}, "last_action_at": now - timedelta(hours=2)},
    )
    tracker = DecisionTracker()
    result, context = tree.run(context, tracker)
    tracker.render(title="Example signal gate", final_outcome=result.label, final_detail=f"intent={context.payload.get('intent')}")


if __name__ == "__main__":
    main()
