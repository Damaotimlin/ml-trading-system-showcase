from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass(frozen=True, slots=True)
class DecisionContext:
    """
    Immutable execution context provided to each decision node.

    Attributes:
        run_id: Correlates all decisions for a single evaluation cycle.
        timestamp: Wall-clock timestamp captured at the start of evaluation.
        config: Strategy configuration snapshot (already resolved via config store).
        payload: Arbitrary data associated with this run (predictions, market data).
        services: Lazily-instantiated service/adapters keyed by identifier.
    """

    run_id: str
    timestamp: datetime
    config: Dict[str, Any]
    payload: Dict[str, Any] = field(default_factory=dict)
    services: Dict[str, Any] = field(default_factory=dict)

    def get_service(self, name: str, default: Optional[Any] = None) -> Any:
        """Return a service adapter by name."""
        return self.services.get(name, default)

    def require_service(self, name: str) -> Any:
        """Return a service adapter or raise KeyError if missing."""
        if name not in self.services:
            raise KeyError(f"Service '{name}' not available in DecisionContext")
        return self.services[name]

    def with_payload(self, **updates: Any) -> "DecisionContext":
        """Return a new context with updated payload entries."""
        new_payload = dict(self.payload)
        new_payload.update(updates)
        return replace(self, payload=new_payload)

    def with_services(self, **updates: Any) -> "DecisionContext":
        """Return a new context with additional/updated services."""
        new_services = dict(self.services)
        new_services.update(updates)
        return replace(self, services=new_services)
