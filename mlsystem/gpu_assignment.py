"""
Market-Aware GPU Assignment Utility with Graceful Transitions
and Automatic GPU Detection

Provides dynamic GPU/CPU assignment based on market session and workload type.
Implements graceful transitions without force-killing running tasks.
Automatically detects multiple GPUs and enables support GPU if available.
"""
import os
import subprocess
import logging
from datetime import datetime
from typing import Literal, Tuple, Optional, List
from zoneinfo import ZoneInfo
from dataclasses import dataclass
import threading

log = logging.getLogger(__name__)

MarketSession = Literal["pre_market", "market_in", "post_market", "overnight", "weekend"]
WorkloadType = Literal["deep_search", "finetune", "predict", "predict_heavy", "train"]
ResourceType = Literal["gpu", "cpu", "disabled", "wait"]
GPU0Mode = Literal["predict", "heavy_work", "transitioning"]
GPU1Mode = Literal["predict", "finetune"]

ET = ZoneInfo("America/New_York")


# =============================================================================
# AUTOMATIC GPU DETECTION
# =============================================================================

@dataclass
class GPUInfo:
    """Information about a detected GPU."""
    index: int
    name: str
    memory_total_mb: int
    memory_free_mb: int
    uuid: str


class GPUDetector:
    """
    Automatically detects available GPUs and determines optimal configuration.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._gpus: List[GPUInfo] = []
        self._detection_time: Optional[datetime] = None
        self._cache_duration = 300  # 5 minutes cache
        self._detect_gpus()

    def _detect_gpus(self) -> None:
        """Detect all available NVIDIA GPUs using nvidia-smi."""
        self._gpus = []
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.total,memory.free,uuid",
                    "--format=csv,noheader,nounits"
                ],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode != 0:
                log.warning(f"nvidia-smi failed: {result.stderr}")
                return

            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 5:
                    gpu = GPUInfo(
                        index=int(parts[0]),
                        name=parts[1],
                        memory_total_mb=int(parts[2]),
                        memory_free_mb=int(parts[3]),
                        uuid=parts[4]
                    )
                    self._gpus.append(gpu)
                    log.info(f"Detected GPU {gpu.index}: {gpu.name} ({gpu.memory_total_mb}MB)")

            self._detection_time = datetime.now()

        except FileNotFoundError:
            log.warning("nvidia-smi not found - no GPU available")
        except subprocess.TimeoutExpired:
            log.warning("nvidia-smi timed out")
        except Exception as e:
            log.warning(f"GPU detection failed: {e}")

    def refresh(self) -> None:
        """Force refresh GPU detection."""
        self._detect_gpus()

    def _maybe_refresh(self) -> None:
        """Refresh if cache expired."""
        if self._detection_time is None:
            self._detect_gpus()
        elif (datetime.now() - self._detection_time).seconds > self._cache_duration:
            self._detect_gpus()

    @property
    def gpu_count(self) -> int:
        self._maybe_refresh()
        return len(self._gpus)

    @property
    def has_multiple_gpus(self) -> bool:
        return self.gpu_count > 1

    @property
    def primary_gpu(self) -> Optional[GPUInfo]:
        self._maybe_refresh()
        return self._gpus[0] if self._gpus else None

    @property
    def secondary_gpu(self) -> Optional[GPUInfo]:
        self._maybe_refresh()
        return self._gpus[1] if len(self._gpus) >= 2 else None

    @property
    def all_gpus(self) -> List[GPUInfo]:
        self._maybe_refresh()
        return self._gpus.copy()


# Global GPU detector instance
_gpu_detector: Optional[GPUDetector] = None


def _get_detector() -> GPUDetector:
    global _gpu_detector
    if _gpu_detector is None:
        _gpu_detector = GPUDetector()
    return _gpu_detector


def get_gpu_count() -> int:
    return _get_detector().gpu_count


def has_secondary_gpu() -> bool:
    return _get_detector().has_multiple_gpus


def get_primary_gpu_id() -> str:
    gpu = _get_detector().primary_gpu
    return str(gpu.index) if gpu else "0"


def get_secondary_gpu_id() -> Optional[str]:
    gpu = _get_detector().secondary_gpu
    return str(gpu.index) if gpu else None


def get_primary_gpu_vram_mb() -> int:
    """Get primary GPU (GPU 0) VRAM in MB."""
    gpu = _get_detector().primary_gpu
    return gpu.memory_total_mb if gpu else 0


def get_secondary_gpu_vram_mb() -> int:
    """Get secondary GPU (GPU 1) VRAM in MB, or 0 if not available."""
    gpu = _get_detector().secondary_gpu
    return gpu.memory_total_mb if gpu else 0


def is_gpu0_larger_vram() -> bool:
    """
    Check if GPU 0 has larger or equal VRAM compared to GPU 1.
    Returns True if GPU 0 >= GPU 1, or if only one GPU exists.
    """
    gpu0_vram = get_primary_gpu_vram_mb()
    gpu1_vram = get_secondary_gpu_vram_mb()
    return gpu0_vram >= gpu1_vram


def get_gpu_vram_config() -> dict:
    """
    Get VRAM configuration for all GPUs.

    Returns dict with:
      - gpu0_vram_mb: GPU 0 VRAM in MB
      - gpu1_vram_mb: GPU 1 VRAM in MB (0 if not present)
      - heavy_gpu_id: GPU index for heavy workloads (larger VRAM)
      - light_gpu_id: GPU index for light workloads (smaller VRAM), or None
      - has_dual_gpu: True if secondary GPU exists
    """
    gpu0_vram = get_primary_gpu_vram_mb()
    gpu1_vram = get_secondary_gpu_vram_mb()
    has_dual = has_secondary_gpu()

    if not has_dual:
        return {
            "gpu0_vram_mb": gpu0_vram,
            "gpu1_vram_mb": 0,
            "heavy_gpu_id": 0,
            "light_gpu_id": None,
            "has_dual_gpu": False,
        }

    # Determine heavy/light GPU based on VRAM
    if gpu0_vram >= gpu1_vram:
        return {
            "gpu0_vram_mb": gpu0_vram,
            "gpu1_vram_mb": gpu1_vram,
            "heavy_gpu_id": 0,
            "light_gpu_id": 1,
            "has_dual_gpu": True,
        }
    else:
        return {
            "gpu0_vram_mb": gpu0_vram,
            "gpu1_vram_mb": gpu1_vram,
            "heavy_gpu_id": 1,
            "light_gpu_id": 0,
            "has_dual_gpu": True,
        }


def is_dual_gpu_mode_enabled() -> bool:
    """Check if dual GPU mode master switch is enabled."""
    return os.getenv("DUAL_GPU_MODE", "false").lower() == "true"


def is_dual_gpu_enabled() -> bool:
    """
    Check if dual GPU mode should be active.
    Requires: DUAL_GPU_MODE=true AND (DUAL_GPU_AUTO with multiple GPUs OR DUAL_GPU_ENABLED=true)
    """
    if not is_dual_gpu_mode_enabled():
        return False

    explicit = os.getenv("DUAL_GPU_ENABLED", "").lower()
    if explicit == "true":
        return True
    if explicit == "false":
        return False

    auto_detect = os.getenv("DUAL_GPU_AUTO", "true").lower() == "true"
    if auto_detect and has_secondary_gpu():
        log.info(f"Auto-enabled dual GPU mode: detected {get_gpu_count()} GPUs")
        return True

    return False


def get_gpu_config_summary() -> str:
    """Get a summary of GPU configuration for logging."""
    gpus = _get_detector().all_gpus
    if not gpus:
        return "No GPUs detected - CPU only mode"

    lines = [f"Detected {len(gpus)} GPU(s):"]
    for gpu in gpus:
        role = "Primary" if gpu.index == 0 else "Secondary (support)"
        lines.append(f"  GPU {gpu.index}: {gpu.name} ({gpu.memory_total_mb}MB) - {role}")

    if is_dual_gpu_enabled():
        lines.append("Dual GPU mode: ENABLED")
        lines.append("Market hours: GPU 0 predict only, CPU→fine-tune")
    elif is_dual_gpu_mode_enabled():
        lines.append("Dual GPU mode: ENABLED (single GPU detected)")
        lines.append("Market hours: GPU 0 predict only, CPU→fine-tune")
    else:
        lines.append("Dual GPU mode: DISABLED")

    return "\n".join(lines)


# =============================================================================
# MARKET SESSION DETECTION
# =============================================================================

def get_market_session() -> MarketSession:
    """Determine current market session based on ET time."""
    now = datetime.now(ET)
    weekday = now.weekday()
    hour = now.hour
    minute = now.minute
    time_mins = hour * 60 + minute

    if weekday >= 5:
        return "weekend"

    pre_market_start = 4 * 60      # 4:00 AM
    regular_start = 9 * 60 + 30    # 9:30 AM
    regular_end = 16 * 60          # 4:00 PM
    post_market_end = 20 * 60      # 8:00 PM

    if pre_market_start <= time_mins < regular_start:
        return "pre_market"
    elif regular_start <= time_mins < regular_end:
        return "market_in"
    elif regular_end <= time_mins < post_market_end:
        return "post_market"
    else:
        return "overnight"


def is_market_hours() -> bool:
    """Check if currently in market hours (pre_market or market_in)."""
    return get_market_session() in ["pre_market", "market_in"]


def is_off_market() -> bool:
    """Check if currently off-market (post_market, overnight, weekend)."""
    return get_market_session() in ["post_market", "overnight", "weekend"]


# =============================================================================
# GPU 0 TRANSITION STATE
# =============================================================================

class TransitionState:
    """Manages graceful GPU 0 transitions between predict and heavy work modes."""

    def __init__(self):
        self._current_mode: GPU0Mode = "heavy_work"
        self._target_mode: GPU0Mode = "heavy_work"
        self._gpu0_busy: bool = False
        self._lock = threading.Lock()

    @property
    def current_mode(self) -> GPU0Mode:
        return self._current_mode

    @property
    def is_transitioning(self) -> bool:
        return self._current_mode != self._target_mode

    def set_gpu0_busy(self, busy: bool):
        with self._lock:
            self._gpu0_busy = busy
            if not busy and self.is_transitioning:
                self._current_mode = self._target_mode
                log.info(f"GPU 0 transition complete: now in {self._current_mode} mode")

    def request_transition(self, target: GPU0Mode):
        with self._lock:
            if self._current_mode == target:
                return

            self._target_mode = target
            log.info(f"GPU 0 transition requested: {self._current_mode} -> {target}")

            if not self._gpu0_busy:
                self._current_mode = target
                log.info(f"GPU 0 transition immediate: now in {target} mode")

    def can_start_workload(self, workload: WorkloadType) -> bool:
        with self._lock:
            if workload in ["deep_search", "finetune"]:
                return self._current_mode == "heavy_work"
            elif workload == "predict":
                return self._current_mode == "predict"
            return True


_transition_state = TransitionState()


def get_gpu0_target_mode() -> GPU0Mode:
    """Determine what mode GPU 0 should be in based on market session."""
    if is_market_hours():
        return "predict"
    return "heavy_work"


def update_transition_state():
    """Called periodically to update transition state based on market session."""
    target = get_gpu0_target_mode()
    _transition_state.request_transition(target)


def notify_task_start(workload: WorkloadType, gpu_id: str):
    """Called when a task starts on a GPU."""
    if gpu_id == os.getenv("DEEP_SEARCH_GPU", "0"):
        _transition_state.set_gpu0_busy(True)


def notify_task_complete(workload: WorkloadType, gpu_id: str):
    """Called when a task completes on a GPU."""
    if gpu_id == os.getenv("DEEP_SEARCH_GPU", "0"):
        _transition_state.set_gpu0_busy(False)


# =============================================================================
# GPU 1 DYNAMIC MODE (PREDICT → FINE-TUNE)
# =============================================================================

class GPU1ModeManager:
    """Manages GPU 1's dynamic mode switching between predict and fine-tune."""

    def __init__(self):
        self._current_mode: GPU1Mode = "finetune"
        self._lock = threading.Lock()
        self._predict_queue_size: int = 0

    @property
    def current_mode(self) -> GPU1Mode:
        with self._lock:
            return self._current_mode

    def update_predict_queue_size(self, size: int):
        with self._lock:
            self._predict_queue_size = size
            self._evaluate_mode()

    def _evaluate_mode(self):
        session = get_market_session()

        if session in ["pre_market", "market_in"]:
            self._current_mode = "finetune"
        else:
            if self._predict_queue_size > 0:
                self._current_mode = "predict"
            else:
                self._current_mode = "finetune"

    def should_gpu1_predict(self) -> bool:
        return self.current_mode == "predict"

    def should_cpu_predict(self) -> bool:
        """
        Check if CPU should run predict tasks.

        During market hours with dual GPU mode:
        - GPU 0 handles all predict tasks (fast enough with 4 workers)
        - CPU focuses entirely on fine-tune for better throughput

        Off-market without dual GPU:
        - CPU handles predict tasks as fallback
        """
        session = get_market_session()
        # During market hours with dual GPU: GPU 0 handles predict, CPU does fine-tune only
        if session in ["pre_market", "market_in"]:
            if is_dual_gpu_mode_enabled():
                return False  # GPU 0 is fast enough, CPU should focus on fine-tune
            return True  # Single GPU mode: CPU helps with predict
        # Off-market: CPU predict only if no dual GPU (GPU 1 handles predict off-market)
        return not is_dual_gpu_enabled()


_gpu1_mode = GPU1ModeManager()


def get_gpu1_mode() -> GPU1Mode:
    """Get current GPU 1 mode."""
    return _gpu1_mode.current_mode


def update_gpu1_predict_queue(size: int):
    """Update GPU 1 mode based on predict queue size."""
    _gpu1_mode.update_predict_queue_size(size)


# =============================================================================
# GPU 0 INTERVAL PRIORITY FUNCTIONS
# =============================================================================

def is_gpu0_predict_mode() -> bool:
    """Check if GPU 0 is currently in predict mode."""
    return _transition_state.current_mode == "predict"


def should_gpu0_process_interval(interval_idx: int, predict_queue: List[dict]) -> bool:
    """
    Check if GPU 0 should process this interval based on priority.
    Priority order: i=0 (1m) > i=1 (30m) > i=2 (60m)
    """
    if interval_idx == 0:
        return True

    for task in predict_queue:
        task_interval = task.get("interval_idx", 0)
        if task_interval < interval_idx:
            return False

    return True


def get_gpu0_interval_stats(predict_queue: List[dict]) -> dict:
    """Get statistics about pending predict tasks by interval for GPU 0."""
    counts = {0: 0, 1: 0, 2: 0}
    for task in predict_queue:
        idx = task.get("interval_idx", 0)
        if idx in counts:
            counts[idx] += 1
        else:
            counts[2] += 1

    current_priority = None
    for idx in [0, 1, 2]:
        if counts[idx] > 0:
            current_priority = idx
            break

    return {
        "counts": counts,
        "total": sum(counts.values()),
        "current_priority": current_priority,
        "interval_names": {0: "1m", 1: "30m", 2: "60m"}
    }


# =============================================================================
# RESOURCE ASSIGNMENT
# =============================================================================

def get_resource_assignment(workload: WorkloadType) -> Tuple[ResourceType, str]:
    """Get resource assignment based on market session and workload."""
    if not is_dual_gpu_mode_enabled():
        return _legacy_assignment(workload)
    return _market_aware_assignment(workload)


def _legacy_assignment(workload: WorkloadType) -> Tuple[ResourceType, str]:
    """Legacy single-GPU assignment."""
    if workload in ["deep_search", "finetune"]:
        return ("gpu", "")
    elif workload == "predict":
        if os.getenv("ALLOW_GPU_PREDICT", "false").lower() == "true":
            return ("gpu", "")
        return ("cpu", "-1")
    return ("cpu", "-1")


def _market_aware_assignment(workload: WorkloadType) -> Tuple[ResourceType, str]:
    """Market-aware resource assignment for DUAL GPU setup."""
    gpu_0 = os.getenv("DEEP_SEARCH_GPU", "0")
    gpu_1 = os.getenv("SECONDARY_GPU", "1")

    gpu0_mode = _transition_state.current_mode
    gpu1_mode = _gpu1_mode.current_mode
    dual_gpu = is_dual_gpu_enabled()

    # Heavy RAM combos - always CPU
    if workload == "predict_heavy":
        return ("cpu", "-1")

    # Train - always CPU
    if workload == "train":
        return ("cpu", "-1")

    # Fine-tune
    if workload == "finetune":
        if dual_gpu and gpu1_mode == "finetune":
            return ("gpu", gpu_1)
        if gpu0_mode == "heavy_work":
            return ("gpu", gpu_0)
        return ("cpu", "-1")

    # Deep search - GPU 0 only in heavy_work mode
    if workload == "deep_search":
        if gpu0_mode == "heavy_work":
            return ("gpu", gpu_0)
        elif gpu0_mode == "transitioning":
            return ("wait", "-1")
        return ("wait", "-1")

    # Predict
    if workload == "predict":
        if gpu0_mode == "predict":
            return ("gpu", gpu_0)
        elif gpu0_mode == "transitioning":
            return ("wait", "-1")

        if dual_gpu and gpu1_mode == "predict":
            return ("gpu", gpu_1)

        if _gpu1_mode.should_cpu_predict():
            return ("cpu", "-1")

        # During market hours with dual GPU: don't assign predict to CPU
        # GPU 0 handles predict fast enough, CPU should focus on fine-tune
        return ("disabled", "-1")

    return ("cpu", "-1")


def should_run_workload(workload: WorkloadType) -> bool:
    """Check if workload should run now or wait."""
    resource_type, _ = get_resource_assignment(workload)
    return resource_type not in ["disabled", "wait"]


def get_pool_size(workload: WorkloadType, resource: ResourceType = "gpu") -> int:
    """Get recommended pool size for workload."""
    session = get_market_session()
    dual_gpu = is_dual_gpu_enabled()
    off_market = session in ["post_market", "overnight", "weekend"]

    if resource == "gpu":
        pool_sizes = {
            "predict": int(os.getenv("GPU0_PREDICT_POOL", "2")),
            "finetune": int(os.getenv("GPU1_FINETUNE_POOL", "1")),
            "deep_search": int(os.getenv("GPU0_DEEPSEARCH_POOL", "1")),
        }
        return pool_sizes.get(workload, 1)

    if resource == "cpu":
        if workload == "finetune":
            # During market hours with dual GPU: CPU focuses entirely on fine-tune
            # since GPU 0 handles predict fast enough
            if is_dual_gpu_mode_enabled() and not off_market:
                # Market hours + dual GPU: all CPU for fine-tune
                return int(os.getenv("CPU_FINETUNE_POOL_MARKET_DUALGPU", os.getenv("CPU_FINETUNE_POOL_OFFMARKET", "4")))
            if off_market and dual_gpu:
                return int(os.getenv("CPU_FINETUNE_POOL_OFFMARKET", "4"))
            return int(os.getenv("CPU_FINETUNE_POOL", "2"))

        if workload == "predict":
            # During market hours with dual GPU: GPU 0 handles all predict
            if is_dual_gpu_mode_enabled() and not off_market:
                return 0  # No CPU predict during market hours with dual GPU
            if not off_market:
                return int(os.getenv("CPU_PREDICT_POOL", "4"))
            if not dual_gpu:
                return int(os.getenv("CPU_PREDICT_POOL_OFFMARKET", "2"))
            return 0

        if workload == "predict_heavy":
            return int(os.getenv("CPU_HEAVY_POOL", "4"))

    return 1
