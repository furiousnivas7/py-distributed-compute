"""Map job orchestration: partition input data across MAP tasks and collect
their results back into one ordered list.

This sits above the existing task execution infrastructure. A MAP task is
just another task_type flowing through the same Scheduler / WorkerManager /
dispatch pipeline as ADD or MULTIPLY, so it gets assignment, concurrent
dispatch, retry, attempt protection, and failure recovery for free -- no
second scheduler, no MapReduce-specific dispatch logic.
"""

from common.models import Task
from master.scheduler import Scheduler


def partition(data: list, num_partitions: int) -> list[list]:
    """Split `data` into at most `num_partitions` roughly-equal chunks,
    preserving order. Returns [] for empty input, and never returns more
    chunks than there are items (so partition([1, 2], 5) gives 2 chunks)."""
    if num_partitions < 1:
        raise ValueError("num_partitions must be at least 1")

    if not data:
        return []

    num_partitions = min(num_partitions, len(data))
    chunk_size, remainder = divmod(len(data), num_partitions)

    chunks = []
    start = 0
    for i in range(num_partitions):
        size = chunk_size + (1 if i < remainder else 0)
        chunks.append(data[start : start + size])
        start += size

    return chunks


def build_map_job(
    scheduler: Scheduler,
    job_id: str,
    operation: str,
    data: list,
    num_partitions: int,
) -> list[Task]:
    """Partition `data` and submit one MAP task per partition.

    Returns the submitted tasks in partition order (index 0 is the first
    chunk) -- this order is what collect_map_results needs to reassemble
    the output correctly, since tasks can complete in any order.
    """
    chunks = partition(data, num_partitions)

    tasks = []
    for index, chunk in enumerate(chunks):
        task_id = f"{job_id}-map-{index}"
        task = scheduler.submit_task(task_id, "MAP", {"operation": operation, "data": chunk})
        tasks.append(task)

    return tasks


def collect_map_results(tasks: list[Task], responses: list[dict]) -> list:
    """Reassemble the full mapped list from MAP task responses.

    `responses` (e.g. from drain_pending_tasks) arrive in whatever order
    workers happened to finish in, which can differ from partition order --
    concurrent dispatch means a later partition can complete before an
    earlier one. Reordering by `tasks` (partition order) is what makes the
    result correct regardless of completion order.

    Raises ValueError if any task's response is missing or unsuccessful.
    """
    responses_by_task_id = {response["payload"]["task_id"]: response for response in responses}

    combined = []
    for task in tasks:
        response = responses_by_task_id.get(task.task_id)
        if response is None:
            raise ValueError(f"No response received for task {task.task_id}")

        payload = response["payload"]
        if payload["status"] != "success":
            raise ValueError(f"Map task {task.task_id} failed: {payload.get('message')}")

        combined.extend(payload["result"])

    return combined
