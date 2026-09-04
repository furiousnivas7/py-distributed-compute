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
