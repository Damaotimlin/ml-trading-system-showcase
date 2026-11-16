"""GPU / CPU assignment: session detection, GPU discovery, graceful transitions, routing."""
from datetime import datetime
from types import SimpleNamespace

import pytest

from mlsystem import gpu_assignment as ga

MONDAY, SATURDAY = (2026, 11, 16), (2026, 11, 21)
ENV_VARS = ["DUAL_GPU_MODE", "DUAL_GPU_ENABLED", "DUAL_GPU_AUTO", "DEEP_SEARCH_GPU", "SECONDARY_GPU",
            "ALLOW_GPU_PREDICT", "CPU_PREDICT_POOL", "CPU_FINETUNE_POOL", "CPU_FINETUNE_POOL_OFFMARKET",
            "CPU_FINETUNE_POOL_MARKET_DUALGPU", "CPU_PREDICT_POOL_OFFMARKET"]
TWO_GPUS = "0, Demo GPU A, 12000, 11000, GPU-aaaa\n1, Demo GPU B, 24000, 23000, GPU-bbbb\n"


@pytest.fixture
def clock(monkeypatch):
    """Freeze the module's idea of 'now' (Eastern Time) and start from clean state."""
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(ga, "_transition_state", ga.TransitionState())
    monkeypatch.setattr(ga, "_gpu1_mode", ga.GPU1ModeManager())

    def set_time(day, hour, minute=0):
        moment = datetime(*day, hour, minute, tzinfo=ga.ET)

        class FrozenDatetime(datetime):
            @classmethod
            def now(cls, tz=None):
                return moment if tz else moment.replace(tzinfo=None)

        monkeypatch.setattr(ga, "datetime", FrozenDatetime)

    return set_time


@pytest.fixture
def nvidia_smi(monkeypatch):
    """Replace the nvidia-smi call with canned output (or a missing binary)."""
    def install(stdout=None):
        def fake_run(*args, **kwargs):
            if stdout is None:
                raise FileNotFoundError("nvidia-smi")
            return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

        monkeypatch.setattr(ga.subprocess, "run", fake_run)
        monkeypatch.setattr(ga.GPUDetector, "_instance", None)
        monkeypatch.setattr(ga, "_gpu_detector", None)

    return install


@pytest.mark.parametrize("day, hour, minute, expected", [
    (MONDAY, 3, 59, "overnight"),
    (MONDAY, 4, 0, "pre_market"),
    (MONDAY, 9, 29, "pre_market"),
    (MONDAY, 9, 30, "market_in"),
    (MONDAY, 15, 59, "market_in"),
    (MONDAY, 16, 0, "post_market"),
    (MONDAY, 19, 59, "post_market"),
    (MONDAY, 20, 0, "overnight"),
    (SATURDAY, 10, 0, "weekend"),
])
def test_session_boundaries(clock, day, hour, minute, expected):
    clock(day, hour, minute)
    assert ga.get_market_session() == expected
    assert ga.is_market_hours() == (expected in {"pre_market", "market_in"})
    assert ga.is_off_market() != ga.is_market_hours()


def test_detects_gpus_and_gives_heavy_work_to_the_larger_one(nvidia_smi):
    nvidia_smi(TWO_GPUS)
    assert ga.get_gpu_count() == 2 and ga.has_secondary_gpu()
    config = ga.get_gpu_vram_config()
    assert (config["heavy_gpu_id"], config["light_gpu_id"]) == (1, 0)       # GPU 1 has more memory
    assert not ga.is_gpu0_larger_vram()


def test_missing_nvidia_smi_falls_back_to_cpu_only(nvidia_smi, clock):
    nvidia_smi(None)
    assert ga.get_gpu_count() == 0 and not ga.has_secondary_gpu()
    assert ga.get_gpu_vram_config()["has_dual_gpu"] is False
    assert "CPU only" in ga.get_gpu_config_summary()


def test_dual_gpu_needs_master_switch_and_a_second_gpu(nvidia_smi, clock, monkeypatch):
    nvidia_smi(TWO_GPUS)
    assert not ga.is_dual_gpu_enabled()                    # master switch off
    monkeypatch.setenv("DUAL_GPU_MODE", "true")
    assert ga.is_dual_gpu_enabled()                        # auto-detected
    monkeypatch.setenv("DUAL_GPU_ENABLED", "false")
    assert not ga.is_dual_gpu_enabled()                    # explicit override wins
    nvidia_smi("0, Demo GPU A, 12000, 11000, GPU-aaaa\n")
    monkeypatch.delenv("DUAL_GPU_ENABLED")
    assert not ga.is_dual_gpu_enabled()                    # only one GPU


def test_transition_waits_for_running_work_instead_of_killing_it():
    state = ga.TransitionState()
    state.set_gpu0_busy(True)
    state.request_transition("predict")
    assert state.is_transitioning and state.current_mode == "heavy_work"
    assert state.can_start_workload("deep_search") and not state.can_start_workload("predict")
    state.set_gpu0_busy(False)                             # the running task finishes
    assert not state.is_transitioning and state.current_mode == "predict"
    assert state.can_start_workload("predict") and not state.can_start_workload("deep_search")


def test_idle_gpu_transitions_immediately():
    state = ga.TransitionState()
    state.request_transition("predict")
    assert state.current_mode == "predict" and not state.is_transitioning


def test_single_gpu_legacy_routing(clock, monkeypatch):
    clock(MONDAY, 11)
    assert ga.get_resource_assignment("deep_search") == ("gpu", "")
    assert ga.get_resource_assignment("predict") == ("cpu", "-1")
    monkeypatch.setenv("ALLOW_GPU_PREDICT", "true")
    assert ga.get_resource_assignment("predict") == ("gpu", "")


def test_market_hours_routing_with_two_gpus(clock, nvidia_smi, monkeypatch):
    nvidia_smi(TWO_GPUS)
    monkeypatch.setenv("DUAL_GPU_MODE", "true")
    clock(MONDAY, 11)
    ga.update_transition_state()
    assert ga.is_gpu0_predict_mode()
    assert ga.get_resource_assignment("predict") == ("gpu", "0")        # GPU 0 serves predictions
    assert ga.get_resource_assignment("finetune") == ("gpu", "1")       # GPU 1 keeps fine-tuning
    assert ga.get_resource_assignment("deep_search") == ("wait", "-1")  # deep search waits for the close
    assert ga.get_resource_assignment("train") == ("cpu", "-1")
    assert not ga.should_run_workload("deep_search") and ga.should_run_workload("predict")
    assert ga.get_pool_size("predict", "cpu") == 0                      # CPU left free for fine-tuning


def test_off_market_routing_with_two_gpus(clock, nvidia_smi, monkeypatch):
    nvidia_smi(TWO_GPUS)
    monkeypatch.setenv("DUAL_GPU_MODE", "true")
    clock(MONDAY, 22)
    ga.update_transition_state()
    assert ga.get_resource_assignment("deep_search") == ("gpu", "0")    # GPU 0 back on heavy work
    ga.update_gpu1_predict_queue(3)
    assert ga.get_gpu1_mode() == "predict"
    assert ga.get_resource_assignment("predict") == ("gpu", "1")        # GPU 1 drains the predict queue
    ga.update_gpu1_predict_queue(0)
    assert ga.get_gpu1_mode() == "finetune"
    assert ga.get_resource_assignment("finetune") == ("gpu", "1")


def test_deep_search_holds_while_gpu0_finishes_predicting(clock, nvidia_smi, monkeypatch):
    nvidia_smi(TWO_GPUS)
    monkeypatch.setenv("DUAL_GPU_MODE", "true")
    clock(MONDAY, 15, 59)
    ga.update_transition_state()
    ga.notify_task_start("predict", "0")                   # a prediction is running at the close
    clock(MONDAY, 16, 0)
    ga.update_transition_state()
    assert ga.get_resource_assignment("deep_search") == ("wait", "-1")
    ga.notify_task_complete("predict", "0")
    assert ga.get_resource_assignment("deep_search") == ("gpu", "0")


def test_shorter_intervals_are_predicted_first():
    queue = [{"interval_idx": 1}, {"interval_idx": 2}, {"interval_idx": 2}]
    assert ga.should_gpu0_process_interval(0, queue)
    assert ga.should_gpu0_process_interval(1, queue)
    assert not ga.should_gpu0_process_interval(2, queue)   # a shorter interval is still waiting
    stats = ga.get_gpu0_interval_stats(queue)
    assert stats["counts"] == {0: 0, 1: 1, 2: 2} and stats["current_priority"] == 1
