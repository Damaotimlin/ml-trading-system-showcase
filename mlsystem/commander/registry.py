"""
Combo Registry - Real-time tracking of active training combos
Prevents conflicts between GPU and CPU executors
"""
import time
from dataclasses import dataclass
from enum import Enum
from threading import Lock
from typing import Dict, Optional


class ResourceType(Enum):
    """Resource types that can run tasks"""
    GPU = "gpu"
    CPU = "cpu"
    PREDICT = "predict"  # Light prediction tasks (2 parallel workers)
    DATA = "data"
    RESEARCH = "research"


@dataclass(frozen=True)
class ComboKey:
    """Unique identifier for a training combo"""
    interval_code: int
    predict_days: float
    arch_name: str
    mode: str  # "delta" or "absolute"

    def __str__(self) -> str:
        return f"i={self.interval_code} d={self.predict_days} arch={self.arch_name} mode={self.mode}"


class ComboRegistry:
    """
    Thread-safe registry tracking which resource owns each combo

    This prevents race conditions where GPU and CPU try to run the same combo
    """

    def __init__(self):
        self._active: Dict[ComboKey, ResourceType] = {}
        self._timestamps: Dict[ComboKey, float] = {}  # Track when combo started
        self._lock = Lock()

    def can_acquire(self, combo: ComboKey, resource: ResourceType) -> bool:
        """
        Check if this resource can acquire the combo

        Returns:
            True if combo is free, False if already owned by ANY resource

        CRITICAL: Combo must be completely free to be acquired. This prevents duplicate
        assignments when multiple workers of the same resource type exist.
        """
        with self._lock:
            owner = self._active.get(combo)
            # CRITICAL FIX: Only allow acquisition if combo is completely free
            # This prevents duplicate assignments when GPU_POOL_SIZE > 1
            return owner is None

    def can_acquire_for_predict(self, combo: ComboKey, resource: ResourceType) -> bool:
        """
        Check if this resource can acquire the combo for PREDICT tasks.

        PREDICT tasks (which just read model files) are allowed to run even when
        the combo is owned by CPU (fine-tune). This allows GPU predict to run
        in parallel with CPU fine-tune, maximizing utilization.

        Returns:
            True if combo is free OR owned by CPU (allow predict to run alongside fine-tune)
            False if owned by GPU or PREDICT (actual conflict)
        """
        with self._lock:
            owner = self._active.get(combo)
            if owner is None:
                return True  # Free - can acquire
            # Allow predict to run alongside CPU fine-tune
            # CPU fine-tune modifies model, predict just reads - no actual conflict
            if owner == ResourceType.CPU:
                return True  # CPU owns it but predict can run
            # Owned by GPU or PREDICT - real conflict
            return False

    def acquire(self, combo: ComboKey, resource: ResourceType) -> bool:
        """
        Try to acquire combo for this resource

        Returns:
            True if acquired, False if already owned by ANY resource (including same type)

        CRITICAL: Each combo can only be assigned to ONE worker at a time, even if
        multiple workers of the same resource type exist (e.g., GPU_POOL_SIZE=2).
        This prevents duplicate tasks from running simultaneously.
        """
        with self._lock:
            owner = self._active.get(combo)
            if owner is None:
                self._active[combo] = resource
                self._timestamps[combo] = time.time()  # Record start time
                return True
            # CRITICAL FIX: Combo already owned - ALWAYS return False, even if same resource type
            # This prevents duplicate assignments when GPU_POOL_SIZE > 1
            return False  # Already owned - cannot re-acquire

    def release(self, combo: ComboKey, resource: ResourceType) -> None:
        """Release combo if owned by this resource"""
        with self._lock:
            owner = self._active.get(combo)
            if owner == resource:
                del self._active[combo]
                self._timestamps.pop(combo, None)  # Clean up timestamp

    def get_active_combos(self, resource: Optional[ResourceType] = None) -> list[ComboKey]:
        """
        Get all active combos, optionally filtered by resource

        Args:
            resource: If specified, only return combos owned by this resource
        """
        with self._lock:
            if resource is None:
                return list(self._active.keys())
            return [combo for combo, owner in self._active.items() if owner == resource]

    def is_active(self, combo: ComboKey) -> bool:
        """Check if combo is currently active"""
        with self._lock:
            return combo in self._active

    def get_owner(self, combo: ComboKey) -> Optional[ResourceType]:
        """Get the resource that owns this combo, or None if free"""
        with self._lock:
            return self._active.get(combo)

    def clear(self) -> None:
        """Clear all active combos (for testing or cleanup)"""
        with self._lock:
            self._active.clear()
            self._timestamps.clear()

    def __len__(self) -> int:
        """Return number of active combos"""
        with self._lock:
            return len(self._active)

    def get_active_intervals(self, resource: Optional[ResourceType] = None) -> set[int]:
        """
        Get set of intervals currently being processed

        Args:
            resource: If specified, only return intervals for this resource

        Returns:
            Set of interval codes currently active
        """
        with self._lock:
            if resource is None:
                combos = list(self._active.keys())
            else:
                combos = [combo for combo, owner in self._active.items() if owner == resource]
            return {combo.interval_code for combo in combos}

    def is_interval_active(self, interval_code: int, exclude_resource: Optional[ResourceType] = None) -> bool:
        """
        Check if any task is processing this interval

        Args:
            interval_code: Interval to check
            exclude_resource: If specified, ignore tasks from this resource

        Returns:
            True if interval is being processed by any resource (excluding exclude_resource)
        """
        with self._lock:
            for combo, owner in self._active.items():
                if combo.interval_code == interval_code:
                    if exclude_resource is None or owner != exclude_resource:
                        return True
            return False

    def get_interval_elapsed_time(self, interval_code: int, resource: Optional[ResourceType] = None) -> Optional[float]:
        """
        Get elapsed time since interval started being processed

        Args:
            interval_code: Interval to check
            resource: If specified, only check this resource's tasks

        Returns:
            Elapsed time in seconds, or None if interval not active
        """
        with self._lock:
            earliest_time = None
            for combo, owner in self._active.items():
                if combo.interval_code == interval_code:
                    if resource is None or owner == resource:
                        start_time = self._timestamps.get(combo)
                        if start_time is not None:
                            if earliest_time is None or start_time < earliest_time:
                                earliest_time = start_time

            if earliest_time is not None:
                return time.time() - earliest_time
            return None
