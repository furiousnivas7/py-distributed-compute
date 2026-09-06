import pytest

from common.models import TaskStatus, WorkerStatus
from master.scheduler import (
    MAX_TASK_ATTEMPTS,
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


def test_requeue_assigned_task_for_failed_worker():
    manager = WorkerManager()
    manager.register_worker("worker-1", "127.0.0.1", 6001)

    scheduler = Scheduler(manager)

    task = scheduler.submit_task(
        "task-1",
        "ADD",
        {"a": 10, "b": 20},
    )

    scheduler.assign_task("task-1")

    requeued = scheduler.requeue_tasks_for_worker("worker-1")

    assert task.status == TaskStatus.PENDING
    assert task.assigned_worker_id is None
    assert requeued == [task]


def test_requeue_running_task_for_failed_worker():
    manager = WorkerManager()
    manager.register_worker("worker-1", "127.0.0.1", 6001)

    scheduler = Scheduler(manager)

    task = scheduler.submit_task(
        "task-1",
        "ADD",
        {"a": 10, "b": 20},
    )

    scheduler.assign_task("task-1")
    scheduler.start_task("task-1")

    assert task.status == TaskStatus.RUNNING

    requeued = scheduler.requeue_tasks_for_worker("worker-1")

    assert task.status == TaskStatus.PENDING
    assert task.assigned_worker_id is None
    assert requeued == [task]


def test_requeue_does_not_affect_completed_task():
    manager = WorkerManager()
    manager.register_worker("worker-1", "127.0.0.1", 6001)

    scheduler = Scheduler(manager)

    task = scheduler.submit_task(
        "task-1",
        "ADD",
        {"a": 10, "b": 20},
    )

    scheduler.assign_task("task-1")
    scheduler.complete_task("task-1")

    requeued = scheduler.requeue_tasks_for_worker("worker-1")

    assert task.status == TaskStatus.COMPLETED
    assert requeued == []


def test_requeue_only_affects_tasks_of_failed_worker():
    manager = WorkerManager()
    manager.register_worker("worker-1", "127.0.0.1", 6001)
    manager.register_worker("worker-2", "127.0.0.1", 6002)

    scheduler = Scheduler(manager)

    task1 = scheduler.submit_task(
        "task-1",
        "ADD",
        {"a": 10, "b": 20},
    )

    task2 = scheduler.submit_task(
        "task-2",
        "ADD",
        {"a": 30, "b": 40},
    )

    scheduler.assign_task("task-1")
    scheduler.assign_task("task-2")

    assert task1.assigned_worker_id == "worker-1"
    assert task2.assigned_worker_id == "worker-2"

    requeued = scheduler.requeue_tasks_for_worker("worker-1")

    assert task1.status == TaskStatus.PENDING
    assert task1.assigned_worker_id is None

    assert task2.status == TaskStatus.ASSIGNED
    assert task2.assigned_worker_id == "worker-2"

    assert requeued == [task1]


def test_requeued_task_can_be_assigned_to_another_worker():
    manager = WorkerManager()
    manager.register_worker("worker-1", "127.0.0.1", 6001)
    manager.register_worker("worker-2", "127.0.0.1", 6002)

    scheduler = Scheduler(manager)

    task1 = scheduler.submit_task(
        "task-1",
        "ADD",
        {"a": 10, "b": 20},
    )

    scheduler.assign_task("task-1")

    assert task1.assigned_worker_id == "worker-1"

    scheduler.requeue_tasks_for_worker("worker-1")

    # worker-1 is still IDLE here in this isolated scheduler test,
    # so mark it FAILED to simulate the actual failure state.
    manager.update_status("worker-1", WorkerStatus.FAILED)

    reassigned = scheduler.assign_next_pending_task()

    assert reassigned == task1
    assert task1.status == TaskStatus.ASSIGNED
    assert task1.assigned_worker_id == "worker-2"


def test_new_task_starts_at_attempt_zero(scheduler):
    task = scheduler.submit_task("task-1", "MAP", {})
    assert task.attempt == 0


def test_first_assignment_sets_attempt_to_one(scheduler):
    scheduler.submit_task("task-1", "MAP", {})
    task = scheduler.assign_task("task-1")
    assert task.attempt == 1


def test_requeue_does_not_reset_attempt():
    manager = WorkerManager()
    manager.register_worker("worker-1", "127.0.0.1", 6001)
    scheduler = Scheduler(manager)

    task = scheduler.submit_task("task-1", "ADD", {"a": 1, "b": 1})
    scheduler.assign_task("task-1")
    assert task.attempt == 1

    scheduler.requeue_tasks_for_worker("worker-1")

    assert task.status == TaskStatus.PENDING
    assert task.attempt == 1


def test_reassignment_increments_attempt_to_two():
    manager = WorkerManager()
    manager.register_worker("worker-1", "127.0.0.1", 6001)
    manager.register_worker("worker-2", "127.0.0.1", 6002)
    scheduler = Scheduler(manager)

    task = scheduler.submit_task("task-1", "ADD", {"a": 1, "b": 1})
    scheduler.assign_task("task-1")
    assert task.attempt == 1

    scheduler.requeue_tasks_for_worker("worker-1")
    manager.update_status("worker-1", WorkerStatus.FAILED)

    reassigned = scheduler.assign_next_pending_task()

    assert reassigned.attempt == 2
    assert reassigned.assigned_worker_id == "worker-2"


def test_requeue_multiple_tasks_for_failed_worker():
    manager = WorkerManager()
    scheduler = Scheduler(manager)

    manager.register_worker("worker-1", "127.0.0.1", 6001)

    task1 = scheduler.submit_task("task-1", "ADD", {"a": 1, "b": 2})
    task2 = scheduler.submit_task("task-2", "ADD", {"a": 3, "b": 4})

    scheduler.assign_task("task-1")

    manager.update_status("worker-1", WorkerStatus.IDLE)

    scheduler.assign_task("task-2")

    requeued = scheduler.requeue_tasks_for_worker("worker-1")

    assert len(requeued) == 2
    assert task1.status == TaskStatus.PENDING
    assert task2.status == TaskStatus.PENDING
    assert task1.assigned_worker_id is None
    assert task2.assigned_worker_id is None


def test_worker_failure_does_not_requeue_completed_tasks():
    manager = WorkerManager()
    scheduler = Scheduler(manager)

    manager.register_worker("worker-1", "127.0.0.1", 6001)

    task = scheduler.submit_task("task-1", "ADD", {"a": 10, "b": 20})

    scheduler.assign_task("task-1")
    scheduler.complete_task("task-1")

    requeued = scheduler.requeue_tasks_for_worker("worker-1")

    assert requeued == []
    assert task.status == TaskStatus.COMPLETED


def test_task_can_be_retried_multiple_times():
    manager = WorkerManager()
    scheduler = Scheduler(manager)

    for i in range(1, 4):
        manager.register_worker(f"worker-{i}", "127.0.0.1", 6000 + i)

    task = scheduler.submit_task("task-1", "ADD", {"a": 5, "b": 7})

    # Attempt 1
    scheduler.assign_task("task-1")
    assert task.attempt == 1
    assert task.assigned_worker_id == "worker-1"

    scheduler.requeue_tasks_for_worker("worker-1")

    # Attempt 2
    scheduler.assign_next_pending_task()
    assert task.attempt == 2
    assert task.assigned_worker_id == "worker-2"

    scheduler.requeue_tasks_for_worker("worker-2")

    # Attempt 3
    scheduler.assign_next_pending_task()
    assert task.attempt == 3
    assert task.assigned_worker_id == "worker-3"

    scheduler.complete_task("task-1")

    assert task.status == TaskStatus.COMPLETED
    assert task.attempt == 3


def test_failed_worker_is_not_selected_for_retry():
    manager = WorkerManager()
    scheduler = Scheduler(manager)

    manager.register_worker("worker-1", "127.0.0.1", 6001)
    manager.register_worker("worker-2", "127.0.0.1", 6002)

    task = scheduler.submit_task("task-1", "ADD", {"a": 1, "b": 1})

    scheduler.assign_task("task-1")

    manager.update_status("worker-1", WorkerStatus.FAILED)

    scheduler.requeue_tasks_for_worker("worker-1")

    scheduler.assign_next_pending_task()

    assert task.assigned_worker_id == "worker-2"
    assert task.attempt == 2


def test_retry_exhaustion_marks_task_failed_after_max_attempts():
    manager = WorkerManager()
    scheduler = Scheduler(manager)

    for i in range(1, MAX_TASK_ATTEMPTS + 1):
        manager.register_worker(f"worker-{i}", "127.0.0.1", 6000 + i)

    task = scheduler.submit_task("task-1", "ADD", {"a": 1, "b": 1})

    for i in range(1, MAX_TASK_ATTEMPTS + 1):
        worker_id = f"worker-{i}"
        assigned = scheduler.assign_next_pending_task()
        assert assigned.assigned_worker_id == worker_id
        assert assigned.attempt == i

        manager.update_status(worker_id, WorkerStatus.FAILED)
        requeued = scheduler.requeue_tasks_for_worker(worker_id)

        if i < MAX_TASK_ATTEMPTS:
            assert requeued == [task]
            assert task.status == TaskStatus.PENDING
        else:
            assert requeued == []
            assert task.status == TaskStatus.FAILED

    # Exhausted: FAILED, not PENDING, so there's nothing left to retry.
    assert scheduler.assign_next_pending_task() is None
    assert task.status == TaskStatus.FAILED
    assert task.attempt == MAX_TASK_ATTEMPTS


def test_task_succeeding_before_exhaustion_is_not_marked_failed():
    manager = WorkerManager()
    scheduler = Scheduler(manager)

    manager.register_worker("worker-1", "127.0.0.1", 6001)
    manager.register_worker("worker-2", "127.0.0.1", 6002)

    task = scheduler.submit_task("task-1", "ADD", {"a": 1, "b": 1})

    scheduler.assign_task("task-1")  # attempt 1
    scheduler.requeue_tasks_for_worker("worker-1")

    scheduler.assign_next_pending_task()  # attempt 2, well under MAX_TASK_ATTEMPTS
    assert task.attempt == 2
    assert task.status == TaskStatus.ASSIGNED

    scheduler.complete_task("task-1")

    assert task.status == TaskStatus.COMPLETED
