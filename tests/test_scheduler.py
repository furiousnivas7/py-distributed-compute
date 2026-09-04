import pytest

from common.models import TaskStatus, WorkerStatus
from master.scheduler import (
    NoAvailableWorkerError,
    Scheduler,
    TaskNotFoundError,
)
from master.worker_manager import WorkerManager


@pytest.fixture
def scheduler():
    worker_manager = WorkerManager()

    worker_manager.register_worker(
        worker_id="worker-1",
        host="127.0.0.1",
        port=6001,
    )

    return Scheduler(worker_manager)


def test_submit_task(scheduler):
    task = scheduler.submit_task(
        task_id="task-1",
        task_type="MAP",
        payload={"numbers": [1, 2, 3]},
    )

    assert task.task_id == "task-1"
    assert task.task_type == "MAP"
    assert task.status == TaskStatus.PENDING
    assert task.assigned_worker_id is None


def test_assign_task(scheduler):
    scheduler.submit_task(
        task_id="task-1",
        task_type="MAP",
        payload={"numbers": [1, 2, 3]},
    )

    task = scheduler.assign_task("task-1")

    assert task.status == TaskStatus.ASSIGNED
    assert task.assigned_worker_id == "worker-1"

    worker = scheduler.worker_manager.get_worker("worker-1")
    assert worker.status == WorkerStatus.BUSY


def test_start_task(scheduler):
    scheduler.submit_task("task-1", "MAP", {})
    scheduler.assign_task("task-1")

    task = scheduler.start_task("task-1")

    assert task.status == TaskStatus.RUNNING


def test_complete_task_makes_worker_idle(scheduler):
    scheduler.submit_task("task-1", "MAP", {})
    scheduler.assign_task("task-1")
    scheduler.start_task("task-1")

    task = scheduler.complete_task("task-1")

    assert task.status == TaskStatus.COMPLETED

    worker = scheduler.worker_manager.get_worker("worker-1")
    assert worker.status == WorkerStatus.IDLE


def test_fail_task_makes_worker_idle(scheduler):
    scheduler.submit_task("task-1", "MAP", {})
    scheduler.assign_task("task-1")

    task = scheduler.fail_task("task-1")

    assert task.status == TaskStatus.FAILED

    worker = scheduler.worker_manager.get_worker("worker-1")
    assert worker.status == WorkerStatus.IDLE


def test_no_available_worker():
    worker_manager = WorkerManager()

    worker_manager.register_worker(
        worker_id="worker-1",
        host="127.0.0.1",
        port=6001,
    )

    scheduler = Scheduler(worker_manager)

    scheduler.submit_task("task-1", "MAP", {})
    scheduler.submit_task("task-2", "MAP", {})

    scheduler.assign_task("task-1")

    with pytest.raises(NoAvailableWorkerError):
        scheduler.assign_task("task-2")


def test_unknown_task(scheduler):
    with pytest.raises(TaskNotFoundError):
        scheduler.assign_task("missing-task")


def test_duplicate_task_id(scheduler):
    scheduler.submit_task("task-1", "MAP", {})

    with pytest.raises(ValueError):
        scheduler.submit_task("task-1", "REDUCE", {})


def test_assign_next_pending_task_picks_oldest_first(scheduler):
    scheduler.submit_task("task-1", "MAP", {})
    scheduler.submit_task("task-2", "MAP", {})

    task = scheduler.assign_next_pending_task()

    assert task.task_id == "task-1"
    assert task.status == TaskStatus.ASSIGNED
    assert task.assigned_worker_id == "worker-1"


def test_assign_next_pending_task_returns_none_when_no_tasks(scheduler):
    assert scheduler.assign_next_pending_task() is None


def test_assign_next_pending_task_returns_none_when_no_idle_worker():
    worker_manager = WorkerManager()
    worker_manager.register_worker("worker-1", "127.0.0.1", 6001)
    scheduler = Scheduler(worker_manager)

    scheduler.submit_task("task-1", "MAP", {})
    scheduler.assign_next_pending_task()  # takes the only idle worker

    scheduler.submit_task("task-2", "MAP", {})
    result = scheduler.assign_next_pending_task()

    assert result is None
    assert scheduler.get_task("task-2").status == TaskStatus.PENDING


def test_assign_next_pending_task_multiple_workers():
    worker_manager = WorkerManager()
    worker_manager.register_worker("worker-1", "127.0.0.1", 6001)
    worker_manager.register_worker("worker-2", "127.0.0.1", 6002)
    scheduler = Scheduler(worker_manager)

    scheduler.submit_task("task-1", "MAP", {})
    scheduler.submit_task("task-2", "MAP", {})
    scheduler.submit_task("task-3", "MAP", {})
    scheduler.submit_task("task-4", "MAP", {})

    first = scheduler.assign_next_pending_task()
    second = scheduler.assign_next_pending_task()
    third = scheduler.assign_next_pending_task()

    assert {first.assigned_worker_id, second.assigned_worker_id} == {"worker-1", "worker-2"}
    assert third is None

    statuses = {t.task_id: t.status for t in scheduler.get_all_tasks()}
    assert statuses["task-1"] == TaskStatus.ASSIGNED
    assert statuses["task-2"] == TaskStatus.ASSIGNED
    assert statuses["task-3"] == TaskStatus.PENDING
    assert statuses["task-4"] == TaskStatus.PENDING
