"""
Decision engine: modular decision trees declared in YAML/JSON.

Nodes are small classes registered by name; a tree spec wires them together by
outcome (pass / fail / continue / skip / error), so strategies change in config
rather than in code. The production node library is not part of this showcase;
see ``examples/`` for a self-contained demonstration.
"""

from .context import DecisionContext
from .nodes import DecisionNode, JudgeNode, TaskNode, DecisionResult, DecisionStatus
from .registry import NodeRegistry, NodeRegistrationError
from .tracking import DecisionRecord, DecisionTracker
from .tree import DecisionTree, DecisionTreeBuilder, DecisionTreeSpec, NodeSpec
from .specs import load_spec_file

__all__ = [
    "DecisionContext",
    "DecisionNode",
    "JudgeNode",
    "TaskNode",
    "DecisionResult",
    "DecisionStatus",
    "NodeRegistry",
    "NodeRegistrationError",
    "DecisionRecord",
    "DecisionTracker",
    "DecisionTree",
    "DecisionTreeBuilder",
    "DecisionTreeSpec",
    "NodeSpec",
    "load_spec_file",
]
