"""Task scheduling and worker assignment."""

from common.models import Task, TaskStatus, WorkerStatus
from master.worker_manager import WorkerManager

# A worker failing while running a task means that specific attempt is lost,
# not necessarily that the task itself is unrunnable — so it's retried on a
# different worker. But without a cap, a task whose failure has nothing to
# do with *which* worker runs it (e.g. it crashes every worker that touches
# it) would retry forever. After this many attempts, requeue_tasks_for_worker
# gives up and marks the task FAILED instead of PENDING.
MAX_TASK_ATTEMPTS = 3


class TaskNotFoundError(KeyError):
    """Raised when a task does not exist."""


class NoAvailableWorkerError(RuntimeError):
    """Raised when no idle worker is available."""


class Scheduler:
    def __init__(self, worker_manager: WorkerManager):
        self.worker_manager = worker_manager
        self._tasks: dict[str, Task] = {}

    def submit_task(
        self,
        task_id: str,
        task_type: str,
        payload: dict,
    ) -> Task:
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("task_id must be a non-empty string")

        if not isinstance(task_type, str) or not task_type:
            raise ValueError("task_type must be a non-empty string")

        if not isinstance(payload, dict):
            raise ValueError("payload must be a dictionary")

        if task_id in self._tasks:
            raise ValueError(f"Task already exists: {task_id}")

        task = Task(
            task_id=task_id,
            task_type=task_type,
            payload=payload,
        )

        self._tasks[task_id] = task
        return task

    def get_task(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    def get_all_tasks(self) -> list[Task]:
        return list(self._tasks.values())

    def assign_next_pending_task(self) -> Task | None:
        """Assign the oldest PENDING task to an IDLE worker, if both exist.

        Returns the assigned task, or None if there is no pending task or
        no idle worker is currently available (the task stays PENDING).
        """
        for task in self._tasks.values():
            if task.status != TaskStatus.PENDING:
                continue

            try:
                return self.assign_task(task.task_id)
            except NoAvailableWorkerError:
                return None

        return None

    def assign_task(self, task_id: str) -> Task:
        task = self._tasks.get(task_id)

        if task is None:
            raise TaskNotFoundError(f"Unknown task: {task_id}")

        if task.status != TaskStatus.PENDING:
            raise ValueError(
                f"Task cannot be assigned from status: {task.status}"
            )

        workers = self.worker_manager.get_all_workers()

        for worker in workers:
            if worker.status == WorkerStatus.IDLE:
                task.attempt += 1
                task.assigned_worker_id = worker.worker_id
                task.status = TaskStatus.ASSIGNED

                self.worker_manager.update_status(
                    worker.worker_id,
                    WorkerStatus.BUSY,
                )

                return task

        raise NoAvailableWorkerError("No idle worker available")

    def start_task(self, task_id: str) -> Task:
        task = self._get_existing_task(task_id)

        if task.status != TaskStatus.ASSIGNED:
            raise ValueError(
                f"Task cannot start from status: {task.status}"
            )

        task.status = TaskStatus.RUNNING
        return task

    def complete_task(self, task_id: str) -> Task:
        task = self._get_existing_task(task_id)

        if task.status not in (
            TaskStatus.ASSIGNED,
            TaskStatus.RUNNING,
        ):
            raise ValueError(
                f"Task cannot complete from status: {task.status}"
            )

        task.status = TaskStatus.COMPLETED

        if task.assigned_worker_id is not None:
            self.worker_manager.update_status(
                task.assigned_worker_id,
                WorkerStatus.IDLE,
            )

        return task

    def fail_task(self, task_id: str) -> Task:
        task = self._get_existing_task(task_id)

        task.status = TaskStatus.FAILED

        if task.assigned_worker_id is not None:
            self.worker_manager.update_status(
                task.assigned_worker_id,
                WorkerStatus.IDLE,
            )

        return task

    def requeue_tasks_for_worker(self, worker_id: str) -> list[Task]:
        """Send a failed worker's in-flight tasks back to PENDING so the
        scheduler can hand them to another worker. COMPLETED/FAILED tasks
        for that worker are left untouched.

        A task that has already reached MAX_TASK_ATTEMPTS is not retried
        again — it's marked FAILED instead of PENDING, and is not included
        in the returned list (it wasn't requeued, it was given up on).
        """
        requeued_tasks = []

        for task in self._tasks.values():
            if task.assigned_worker_id != worker_id or task.status not in (
                TaskStatus.ASSIGNED,
                TaskStatus.RUNNING,
            ):
                continue

            task.assigned_worker_id = None

            if task.attempt >= MAX_TASK_ATTEMPTS:
                task.status = TaskStatus.FAILED
            else:
                task.status = TaskStatus.PENDING
                requeued_tasks.append(task)

        return requeued_tasks

    def _get_existing_task(self, task_id: str) -> Task:
        task = self._tasks.get(task_id)

        if task is None:
            raise TaskNotFoundError(f"Unknown task: {task_id}")

        return task

    def clear(self) -> None:
        self._tasks.clear()
