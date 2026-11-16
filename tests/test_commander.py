import threading

import pytest

from mlsystem.commander.registry import ComboKey, ComboRegistry, ResourceType
from mlsystem.commander.tasks import Task, TaskPriority, TaskStatus

COMBO = ComboKey(interval_code=2, predict_days=1.0, arch_name="lstm_attn", mode="returns")


def test_combo_is_owned_by_one_worker_at_a_time():
    reg = ComboRegistry()
    assert reg.acquire(COMBO, ResourceType.GPU)
    assert not reg.acquire(COMBO, ResourceType.GPU)   # same resource type cannot double-book
    assert not reg.acquire(COMBO, ResourceType.CPU)
    reg.release(COMBO, ResourceType.CPU)               # only the owner can release
    assert reg.get_owner(COMBO) is ResourceType.GPU
    reg.release(COMBO, ResourceType.GPU)
    assert reg.get_owner(COMBO) is None


def test_predict_may_run_alongside_cpu_finetune_but_not_gpu():
    reg = ComboRegistry()
    reg.acquire(COMBO, ResourceType.CPU)
    assert reg.can_acquire_for_predict(COMBO, ResourceType.PREDICT)
    reg.release(COMBO, ResourceType.CPU)
    reg.acquire(COMBO, ResourceType.GPU)
    assert not reg.can_acquire_for_predict(COMBO, ResourceType.PREDICT)


def test_concurrent_acquire_has_exactly_one_winner():
    reg = ComboRegistry()
    wins = []
    barrier = threading.Barrier(16)

    def worker():
        barrier.wait()
        if reg.acquire(COMBO, ResourceType.GPU):
            wins.append(1)

    threads = [threading.Thread(target=worker) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(wins) == 1


def test_task_validation_and_lifecycle():
    with pytest.raises(ValueError):
        Task(task_id="", label="x", priority=TaskPriority.NORMAL, allowed_resources=[ResourceType.CPU], command=["true"])
    task = Task(task_id="t1", label="train", priority=TaskPriority.NORMAL,
                allowed_resources=[ResourceType.CPU], command=["python", "train.py"], combo=COMBO)
    assert task.can_run_on(ResourceType.CPU) and not task.can_run_on(ResourceType.GPU)
    task.mark_running()
    task.mark_completed(exit_code=1, duration=2.5)
    assert task.status is TaskStatus.FAILED and task.exit_code == 1
    assert str(task).startswith("[NORMAL] train")


def test_priorities_sort_critical_first():
    assert sorted(TaskPriority, key=lambda p: p.value)[0] is TaskPriority.CRITICAL
