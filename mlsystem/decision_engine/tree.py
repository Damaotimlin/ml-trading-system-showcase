from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .context import DecisionContext
from .nodes import DecisionNode, DecisionResult, DecisionStatus
from .registry import NodeRegistry, registry as default_registry
from .tracking import DecisionTracker


@dataclass(slots=True)
class NodeSpec:
    """Declarative description of a node within the tree."""

    key: str
    node_id: str
    description: str = ""
    config: Dict[str, object] = field(default_factory=dict)
    transitions: Dict[str, Optional[str]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "node_id": self.node_id,
            "description": self.description,
            "config": dict(self.config),
            "transitions": dict(self.transitions),
        }


@dataclass(slots=True)
class DecisionTreeSpec:
    """High-level specification used to build a tree."""

    nodes: Iterable[NodeSpec]
    start: str
    metadata: Dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "DecisionTreeSpec":
        nodes_section = raw.get("nodes", [])
        node_specs: List[NodeSpec] = []

        for entry in nodes_section:
            node_id = entry.get("node_id") or entry.get("id")
            if not node_id:
                raise ValueError("Decision tree node entry missing 'id'/'node_id'")

            key = entry.get("key")
            if not key:
                raise ValueError(f"Decision tree node '{node_id}' missing 'key'")

            transitions_raw = entry.get("transitions") or {}
            transitions: Dict[str, Optional[str]] = {
                str(status): target for status, target in transitions_raw.items()
            }

            node_specs.append(
                NodeSpec(
                    key=key,
                    node_id=node_id,
                    description=entry.get("description", ""),
                    config=entry.get("config") or {},
                    transitions=transitions,
                )
            )

        start = raw.get("start")
        if not start:
            raise ValueError("Decision tree spec missing 'start' node identifier")

        metadata = raw.get("metadata") or {}

        return cls(nodes=node_specs, start=start, metadata=metadata)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": self.start,
            "nodes": [node.to_dict() for node in self.nodes],
            "metadata": dict(self.metadata),
        }


class DecisionTree:
    """Executable decision tree produced by the builder."""

    def __init__(
        self,
        nodes: Dict[str, DecisionNode],
        transitions: Dict[str, Dict[str, str]],
        start_node: str,
    ) -> None:
        self._nodes = nodes
        self._transitions = transitions
        self._start_node = start_node

    @property
    def nodes(self) -> Dict[str, DecisionNode]:
        return self._nodes

    def run(self, context: DecisionContext, tracker: DecisionTracker) -> Tuple[DecisionResult, DecisionContext]:
        """
        Execute the tree.

        Returns the last `DecisionResult` and potentially updated `DecisionContext`.
        """
        current_id = self._start_node
        last_result: Optional[DecisionResult] = None

        while current_id:
            node = self._nodes[current_id]
            result, context = node.evaluate(context, tracker)
            last_result = result

            if result.next_node:
                current_id = result.next_node
                continue

            transition_map = self._transitions.get(node.node_id, {})
            next_id = transition_map.get(result.status.value)
            current_id = next_id

        if last_result is None:
            raise RuntimeError("Decision tree finished without executing any node")

        return last_result, context


class DecisionTreeBuilder:
    """Factory class that converts specs into executable trees."""

    def __init__(self, node_registry: Optional[NodeRegistry] = None) -> None:
        self.registry = node_registry or default_registry

    def build(self, spec: DecisionTreeSpec) -> DecisionTree:
        nodes: Dict[str, DecisionNode] = {}
        transitions: Dict[str, Dict[str, str]] = {}

        for node_spec in spec.nodes:
            # Use node_id as the registry lookup key (matches @registry.register names)
            # Use key as the unique identifier within this tree (for transitions)
            node = self.registry.create(
                key=node_spec.node_id,  # Registry lookup uses node_id (e.g., "min_confidence")
                node_id=node_spec.key,  # Node instance gets the unique key (e.g., "entry_min_confidence")
                description=node_spec.description,
                **node_spec.config,
            )
            nodes[node_spec.key] = node  # Use key for tree lookup (matches transitions)
            transitions[node_spec.key] = {k: v for k, v in node_spec.transitions.items()}

        if spec.start not in nodes:
            raise ValueError(f"Start node '{spec.start}' not defined in spec")

        return DecisionTree(nodes=nodes, transitions=transitions, start_node=spec.start)
