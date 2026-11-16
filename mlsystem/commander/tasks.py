"""
Task classes representing different types of work
Each task knows its priority, resource requirements, and execution command
"""
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Callable, Any
from datetime import datetime

from .registry import ComboKey, ResourceType
from .best_json_utils import copy_best_json_from_representative


class TaskPriority(Enum):
    """Task priority levels (lower number = higher priority)"""
    CRITICAL = 0  # Data fetch, blocking operations
    HIGH = 1      # Deep-search tune (GPU only)
    NORMAL = 2    # Regular train/predict
    FALLBACK = 3  # Market-time fallback
    LOW = 4       # Housekeeping, studies


class TaskStatus(Enum):
    """Current status of a task"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Task:
    """
    Base task class

    All tasks share:
    - Unique ID
    - Priority level
    - Resource requirements
    - Combo (if applicable)
    - Execution command
    - Status tracking
    """
    task_id: str
    label: str
    priority: TaskPriority
    allowed_resources: list[ResourceType]
    command: list[str]
    combo: Optional[ComboKey] = None
    log_path: Optional[Path] = None
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    exit_code: Optional[int] = None
    duration: Optional[float] = None

    # Task metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    # Execution control
    preemptible: bool = True  # Can be interrupted
    background: bool = False  # Runs in background (GPU deep-search)
    blocks_training: bool = False  # Blocks other tasks (data fetch)

    # Ordering
    order: int = 0  # For --no-shuffle, preserves declared order

    # Callbacks
    on_complete: Optional[Callable] = None
    on_fail: Optional[Callable] = None

    def __post_init__(self):
        """Validate task after initialization"""
        if not self.task_id:
            raise ValueError("Task must have an ID")
        if not self.allowed_resources:
            raise ValueError("Task must specify allowed resources")
        if not self.command:
            raise ValueError("Task must have a command")

    def can_run_on(self, resource: ResourceType) -> bool:
        """Check if this task can run on the given resource"""
        return resource in self.allowed_resources

    def pre_execute(self) -> bool:
        """
        Hook called before task execution.

        Returns:
            True if task should proceed, False to skip
        """
        return True

    def mark_running(self) -> None:
        """Mark task as running"""
        self.status = TaskStatus.RUNNING
        self.started_at = datetime.now()

    def mark_completed(self, exit_code: int, duration: float) -> None:
        """Mark task as completed"""
        self.status = TaskStatus.COMPLETED if exit_code == 0 else TaskStatus.FAILED
        self.exit_code = exit_code
        self.duration = duration
        self.completed_at = datetime.now()

    def mark_skipped(self, reason: str) -> None:
        """Mark task as skipped"""
        self.status = TaskStatus.SKIPPED
        self.metadata["skip_reason"] = reason

    def __str__(self) -> str:
        """Human-readable task description"""
        combo_str = f" {self.combo}" if self.combo else ""
        return f"[{self.priority.name}] {self.label}{combo_str}"


# ============================================================================
# MODEL STORE HELPERS
# ============================================================================
# Defined here (not in generators.py) so tasks.py can use them without importing
# generators at module level - generators.py imports tasks.py, which would be a
# circular import. generators.py re-exports these three names.

def store_args(combo: ComboKey) -> dict:
    """Keyword args identifying this combo's slot in the model store."""
    from . import config

    interval_name = config.INTERVAL_CODE_TO_NAME.get(
        combo.interval_code, str(combo.interval_code)
    )
    return {
        "ticker": config.TICKER,
        "interval_name": interval_name,
        "arch": combo.arch_name,
        "mode": combo.mode,
        "predict_days": combo.predict_days,
    }


def weights_fresh_for(combo: Optional[ComboKey]) -> tuple[bool, str]:
    """Strict freshness check (versions + age + newer than best.json)."""
    if combo is None:
        return False, "no combo"
    try:
        from . import config
        from .generators import find_best_json_path  # lazy: avoids circular import
        from . import model_store

        a = store_args(combo)
        return model_store.weights_fresh(
            a["ticker"], a["interval_name"], a["arch"], a["mode"], a["predict_days"],
            n_seeds=config.ENSEMBLE_SEEDS,
            best_json_path=find_best_json_path(combo),
            max_age_days=config.MODEL_STORE_MAX_AGE_DAYS,
        )
    except Exception as e:
        return False, f"store check failed: {e}"


def weights_usable_for(combo: Optional[ComboKey]) -> tuple[bool, str]:
    """Loose check: seeds exist and match versions (age / best.json ignored)."""
    if combo is None:
        return False, "no combo"
    try:
        from . import config
        from . import model_store

        a = store_args(combo)
        return model_store.weights_usable(
            a["ticker"], a["interval_name"], a["arch"], a["mode"], a["predict_days"],
            n_seeds=config.ENSEMBLE_SEEDS,
        )
    except Exception as e:
        return False, f"store check failed: {e}"


# ============================================================================
# SPECIALIZED TASK TYPES
# ============================================================================

@dataclass
class DeepSearchTask(Task):
    """
    Deep hyperparameter search (GPU only, high priority)

    Created for Tier 0 combos that need full Optuna search
    """
    def __init__(
        self,
        task_id: str,
        combo: ComboKey,
        trials: int,
        epochs: int,
        command: list[str],
        log_path: Path,
        **kwargs
    ):
        super().__init__(
            task_id=task_id,
            label=f"deep-tune",
            priority=TaskPriority.HIGH,
            allowed_resources=[ResourceType.GPU],
            command=command,
            combo=combo,
            log_path=log_path,
            preemptible=False,  # Don't interrupt deep search
            background=kwargs.get("background", False),
            metadata={
                "trials": trials,
                "epochs": epochs,
                "is_deep_search": True,
            },
            **{k: v for k, v in kwargs.items() if k != "background"}
        )


@dataclass
class CPUDeepSearchTask(Task):
    """
    CPU-assisted deep hyperparameter search (CPU only, LOW priority)

    Used when GPU is busy with deep-search and CPU has no fine-tune tasks.
    Helps explore architecture space in parallel with GPU.

    Key differences from DeepSearchTask:
    - Runs on CPU only (slower but parallel with GPU)
    - Lower priority (doesn't interfere with fine-tuning)
    - Fewer trials (typically 15-25 vs 100+ for GPU)
    - Uses separate Optuna storage to avoid conflicts
    """
    def __init__(
        self,
        task_id: str,
        combo: ComboKey,
        trials: int,
        epochs: int,
        command: list[str],
        log_path: Path,
        **kwargs
    ):
        super().__init__(
            task_id=task_id,
            label="cpu-deep-tune",
            priority=TaskPriority.LOW,  # Lower than NORMAL fine-tune
            allowed_resources=[ResourceType.CPU],  # CPU only
            command=command,
            combo=combo,
            log_path=log_path,
            preemptible=True,  # Can be interrupted for higher priority tasks
            metadata={
                "trials": trials,
                "epochs": epochs,
                "is_cpu_deep_search": True,
            },
            **kwargs
        )


@dataclass
class FineTuneTask(Task):
    """
    Fine-tune hyperparameter search (CPU or GPU, normal priority)

    Quick narrow search around existing best.json using FINETUNE=1 mode.
    Typically used for:
    - CPU fine-tuning during market_off while GPU does deep-search
    - Quick refinement of existing models
    """
    def __init__(
        self,
        task_id: str,
        combo: ComboKey,
        trials: int,
        epochs: int,
        command: list[str],
        log_path: Path,
        allowed_resources: list[ResourceType] = None,
        **kwargs
    ):
        # Default to both CPU and GPU if not specified
        if allowed_resources is None:
            allowed_resources = [ResourceType.CPU, ResourceType.GPU]

        super().__init__(
            task_id=task_id,
            label="finetune",
            priority=TaskPriority.NORMAL,
            allowed_resources=allowed_resources,
            command=command,
            combo=combo,
            log_path=log_path,
            preemptible=True,
            metadata={
                "trials": trials,
                "epochs": epochs,
                "is_finetune": True,
            },
            **kwargs
        )


@dataclass
class TrainWeightsTask(Task):
    """
    Train the model-store seeds for one combo from its best.json (GPU or CPU).

    main.py --model-store fits ENSEMBLE_SEEDS models with the stored
    hyperparameters and writes <Model>/store/<interval>/*.keras + *_meta.json.
    Predict tasks with --from-store then load those weights instead of refitting.

    force=True rebuilds even when the store is already fresh (weekend rebuild).
    """
    def __init__(
        self,
        task_id: str,
        combo: ComboKey,
        command: list[str],
        log_path: Path,
        best_json_path: Path,
        allowed_resources: list[ResourceType] = None,
        priority: TaskPriority = None,
        force: bool = False,
        **kwargs
    ):
        if allowed_resources is None:
            allowed_resources = [ResourceType.GPU, ResourceType.CPU]

        metadata = dict(kwargs.pop("metadata", None) or {})
        metadata.setdefault("best_json", str(best_json_path))
        metadata.setdefault("is_train_weights", True)
        metadata.setdefault("force", bool(force))

        super().__init__(
            task_id=task_id,
            label="train-weights",
            priority=priority or TaskPriority.NORMAL,
            allowed_resources=allowed_resources,
            command=command,
            combo=combo,
            log_path=log_path,
            preemptible=True,
            metadata=metadata,
            **kwargs
        )
        self.best_json_path = best_json_path
        self.force = bool(force)

    def pre_execute(self) -> bool:
        """Copy best.json from a representative if missing, then skip if fresh."""
        from . import config

        if self.best_json_path and not self.best_json_path.exists() and self.combo:
            # Try to copy from representative combo
            model_dir = self.best_json_path.parent
            # Convert interval_code to interval name (e.g., 0 -> "1m")
            interval_name = config.INTERVAL_CODE_TO_NAME.get(self.combo.interval_code, str(self.combo.interval_code))
            copied = copy_best_json_from_representative(
                interval=interval_name,
                predict_days=self.combo.predict_days,
                blueprint=self.combo.arch_name,
                model_dir=model_dir,
                force=False,
                verbose=True
            )
            if copied and self.best_json_path.exists():
                print(f"   \U0001f4cb Using best.json from representative combo")

        # Another worker (or an earlier cycle) may already have built these weights.
        if not self.force:
            fresh, reason = weights_fresh_for(self.combo)
            if fresh:
                print(f"   \u2713 Model store already fresh for {self.combo} - skipping train-weights")
                self.metadata["skip_reason"] = f"weights fresh ({reason})"
                return False

        return True


# Backwards-compatible alias: the old "train from best.json" task is now the
# model-store weight builder. Nothing in the tree constructs TrainTask directly
# except the (unused) generators.generate_train_task().
TrainTask = TrainWeightsTask


@dataclass
class WarmStartTask(Task):
    """
    Refresh already-stored seeds with a few epochs on the newest data.

    Runs after the close (post-market hook), never emitted by bucket generators.
    Skips itself when there is nothing in the store to warm-start from.
    """
    def __init__(
        self,
        task_id: str,
        combo: ComboKey,
        command: list[str],
        log_path: Path,
        best_json_path: Path,
        allowed_resources: list[ResourceType] = None,
        priority: TaskPriority = None,
        **kwargs
    ):
        if allowed_resources is None:
            allowed_resources = [ResourceType.GPU, ResourceType.CPU]

        metadata = dict(kwargs.pop("metadata", None) or {})
        metadata.setdefault("best_json", str(best_json_path))
        metadata.setdefault("is_warm_start", True)

        super().__init__(
            task_id=task_id,
            label="warm-start",
            priority=priority or TaskPriority.LOW,
            allowed_resources=allowed_resources,
            command=command,
            combo=combo,
            log_path=log_path,
            preemptible=True,
            metadata=metadata,
            **kwargs
        )
        self.best_json_path = best_json_path

    def pre_execute(self) -> bool:
        """Nothing to warm-start if the store has no usable seeds for this combo."""
        usable, reason = weights_usable_for(self.combo)
        if not usable:
            print(f"   \u23ed\ufe0f  No usable stored weights for {self.combo} ({reason}) - skipping warm-start")
            self.metadata["skip_reason"] = f"no usable weights ({reason})"
            return False
        return True


@dataclass
class PredictTask(Task):
    """
    Quick prediction using existing model

    By default uses PREDICT pool (dedicated prediction workers)
    Can optionally use GPU when idle for faster predictions (allow_gpu=True)
    CPU is reserved exclusively for fine-tuning and tier management
    """
    def __init__(
        self,
        task_id: str,
        combo: ComboKey,
        command: list[str],
        log_path: Path,
        best_json_path: Path,
        allow_gpu: bool = False,
        **kwargs
    ):
        from . import config

        # Primary: PREDICT pool for dedicated prediction workers
        # Optional: GPU when idle for acceleration (if allow_gpu=True)
        # CPU is reserved for fine-tuning only
        if allow_gpu:
            # Prefer GPU when idle for faster predictions, fallback to PREDICT pool
            # Order matters: GPU tried first, then PREDICT
            # This accelerates predictions when GPU has no training work
            allowed_resources = [ResourceType.GPU, ResourceType.PREDICT]
        else:
            # Default: PREDICT pool only (safer for concurrent training)
            allowed_resources = [ResourceType.PREDICT]

        super().__init__(
            task_id=task_id,
            label="predict",
            priority=TaskPriority.NORMAL,
            allowed_resources=allowed_resources,
            command=command,
            combo=combo,
            log_path=log_path,
            metadata={
                "best_json": str(best_json_path),
            },
            **kwargs
        )


@dataclass
class FallbackTuneTask(Task):
    """
    Market-time fallback tuning (CPU or GPU based on availability)

    Quick tune when best.json is missing during market hours
    """
    def __init__(
        self,
        task_id: str,
        combo: ComboKey,
        trials: int,
        epochs: int,
        command: list[str],
        log_path: Path,
        allow_gpu: bool = False,
        best_json_path: Optional[Path] = None,
        **kwargs
    ):
        resources = [ResourceType.CPU]
        if allow_gpu:
            resources.append(ResourceType.GPU)

        super().__init__(
            task_id=task_id,
            label="fallback-tune",
            priority=TaskPriority.FALLBACK,
            allowed_resources=resources,
            command=command,
            combo=combo,
            log_path=log_path,
            metadata={
                "trials": trials,
                "epochs": epochs,
                "is_fallback": True,
            },
            **kwargs
        )
        self.best_json_path = best_json_path

    def pre_execute(self) -> bool:
        """
        Try to copy best.json from representative combo before fallback tune.

        If successful, skip fallback tune (return False).
        If copy fails, proceed with fallback tune (return True).
        """
        from . import config

        if self.best_json_path and not self.best_json_path.exists() and self.combo:
            # Try to copy from representative combo
            model_dir = self.best_json_path.parent
            # Convert interval_code to interval name (e.g., 0 -> "1m")
            interval_name = config.INTERVAL_CODE_TO_NAME.get(self.combo.interval_code, str(self.combo.interval_code))
            copied = copy_best_json_from_representative(
                interval=interval_name,
                predict_days=self.combo.predict_days,
                blueprint=self.combo.arch_name,
                model_dir=model_dir,
                force=False,
                verbose=True
            )
            if copied and self.best_json_path.exists():
                print(f"   ✓ Using best.json from representative - skipping fallback tune")
                return False  # Skip fallback tune since we have best.json now

        return True  # Proceed with fallback tune if no best.json available


@dataclass
class DataFetchTask(Task):
    """
    Data fetching task (blocks training until complete)

    Critical priority - must complete before training can proceed
    """
    def __init__(
        self,
        task_id: str,
        interval_code: int,
        command: list[str],
        log_path: Path,
        **kwargs
    ):
        super().__init__(
            task_id=task_id,
            label=f"fetch i={interval_code}",
            priority=TaskPriority.CRITICAL,
            allowed_resources=[ResourceType.DATA],
            command=command,
            combo=None,  # Data fetch doesn't have combo
            log_path=log_path,
            preemptible=False,
            blocks_training=True,
            metadata={
                "interval_code": interval_code,
            },
            **kwargs
        )


@dataclass
class StudyTask(Task):
    """
    Analysis/study tasks (low priority, runs when resources idle)

    Includes:
    - Optuna study analysis
    - Prediction evaluation
    - Model exploration
    """
    def __init__(
        self,
        task_id: str,
        study_type: str,  # "optuna" / "predict_eval" / "model_explore"
        command: list[str],
        log_path: Path,
        **kwargs
    ):
        super().__init__(
            task_id=task_id,
            label=f"study:{study_type}",
            priority=TaskPriority.LOW,
            allowed_resources=[ResourceType.RESEARCH],
            command=command,
            combo=None,
            log_path=log_path,
            metadata={
                "study_type": study_type,
            },
            **kwargs
        )


@dataclass
class HousekeepingTask(Task):
    """
    Housekeeping tasks (prune, backup, cleanup)

    Low priority, runs during idle time
    """
    def __init__(
        self,
        task_id: str,
        operation: str,  # "prune" / "backup" / "cleanup"
        command: list[str],
        log_path: Path,
        **kwargs
    ):
        super().__init__(
            task_id=task_id,
            label=f"housekeeping:{operation}",
            priority=TaskPriority.LOW,
            allowed_resources=[ResourceType.DATA],  # Lightweight ops; CPU_POOL_SIZE may be 0
            command=command,
            combo=None,
            log_path=log_path,
            metadata={
                "operation": operation,
            },
            **kwargs
        )
