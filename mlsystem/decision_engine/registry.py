from __future__ import annotations

from typing import Any, Dict, Optional, Type

from .nodes import DecisionNode


class NodeRegistrationError(RuntimeError):
    """Raised when node registration fails or duplicates are detected."""


class NodeRegistry:
    """Registry that manages decision node classes."""

    def __init__(self) -> None:
        self._nodes: Dict[str, Type[DecisionNode]] = {}

    def register(self, name: Optional[str] = None):
        """
        Decorator to register a node class.

        Example:
            @registry.register("accuracy_gate")
            class AccuracyGate(JudgeNode):
                ...
        """

        def _decorator(cls: Type[DecisionNode]) -> Type[DecisionNode]:
            key = (name or cls.__name__).lower()
            if key in self._nodes and self._nodes[key] is not cls:
                raise NodeRegistrationError(f"Node '{key}' already registered")
            self._nodes[key] = cls
            return cls

        return _decorator

    def add(self, key: str, cls: Type[DecisionNode]) -> None:
        """Register a node class programmatically."""
        key = key.lower()
        if key in self._nodes and self._nodes[key] is not cls:
            raise NodeRegistrationError(f"Node '{key}' already registered")
        self._nodes[key] = cls

    def get(self, key: str) -> Type[DecisionNode]:
        """Return the node class for a given key."""
        key = key.lower()
        if key not in self._nodes:
            raise NodeRegistrationError(f"Node '{key}' is not registered")
        return self._nodes[key]

    def create(self, key: str, node_id: str, description: str = "", **config: Any) -> DecisionNode:
        """Instantiate a node by key."""
        cls = self.get(key)
        return cls.build(node_id=node_id, description=description, **config)

    def all_nodes(self) -> Dict[str, Type[DecisionNode]]:
        """Return a copy of the registered node mapping."""
        return dict(self._nodes)

    def describe_nodes(self) -> list[dict[str, Any]]:
        """Return metadata for all registered nodes."""
        descriptions: list[dict[str, Any]] = []
        for key, cls in sorted(self._nodes.items()):
            doc = (cls.__doc__ or "").strip()
            summary = doc.split("\n")[0] if doc else ""
            descriptions.append({
                "key": key,
                "class_name": cls.__name__,
                "module": cls.__module__,
                "node_type": getattr(cls, "node_type", "generic"),
                "description": summary,
                "docstring": doc,
            })
        return descriptions


# Default shared registry
registry = NodeRegistry()

__all__ = ["NodeRegistry", "NodeRegistrationError", "registry"]
