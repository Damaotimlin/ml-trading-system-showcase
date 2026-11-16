from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .context import DecisionContext
    from .tracking import DecisionTracker


class DecisionStatus(Enum):
    """High-level outcome of a decision node."""

    PASS = "pass"
    FAIL = "fail"
    CONTINUE = "continue"
    SKIP = "skip"
    ERROR = "error"


@dataclass(slots=True)
class DecisionResult:
    """
    Standardised payload returned by all decision nodes.

    Attributes:
        status: Outcome classification.
        label: Short message describing the decision.
        detail: Optional human-readable detail.
        data: Arbitrary structured data for logging/UI.
        next_node: Optional override for the next node to execute.
    """

    status: DecisionStatus
    label: str
    detail: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    next_node: Optional[str] = None


class DecisionNode:
    """Base node type in the decision graph."""

    node_type = "generic"

    def __init__(self, node_id: str, description: str = "", **config: Any) -> None:
        self.node_id = node_id
        self.description = description or self.__class__.__name__
        self.config = config

    def evaluate(
        self,
        context: "DecisionContext",
        tracker: "DecisionTracker",
    ) -> tuple[DecisionResult, "DecisionContext"]:
        """Entry point invoked by the tree executor."""
        outcome = self.run(context, tracker)

        if isinstance(outcome, tuple):
            result, new_context = outcome
        else:
            result = outcome
            new_context = context

        if not isinstance(result, DecisionResult):
            raise TypeError(
                f"{self.__class__.__name__}.run() must return DecisionResult "
                f"or (DecisionResult, DecisionContext)"
            )

        return result, new_context

    def run(self, context: "DecisionContext", tracker: "DecisionTracker") -> DecisionResult:
        """Subclasses override to implement behaviour."""
        raise NotImplementedError(f"{self.__class__.__name__}.run() not implemented")

    @classmethod
    def build(cls, node_id: str, description: str = "", **config: Any) -> "DecisionNode":
        """Factory used by the registry/builder."""
        return cls(node_id=node_id, description=description, **config)


class JudgeNode(DecisionNode):
    """Specialised node informing branch selection."""

    node_type = "judge"


class TaskNode(DecisionNode):
    """Node responsible for performing side effects."""

    node_type = "task"
