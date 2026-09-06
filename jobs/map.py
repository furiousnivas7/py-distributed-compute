"""Map job orchestration: partition input data across MAP tasks and collect
their results back into one ordered list.

This sits above the existing task execution infrastructure. A MAP task is
just another task_type flowing through the same Scheduler / WorkerManager /
dispatch pipeline as ADD or MULTIPLY, so it gets assignment, concurrent
dispatch, retry, attempt protection, and failure recovery for free -- no
second scheduler, no MapReduce-specific dispatch logic.
"""

from common.models import Task
from jobs.models import IntermediateResult, ResultStatus
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


def _index_responses_by_task_id(responses: list[dict]) -> dict[str, dict]:
    """Build a task_id -> response lookup, defensively.

    Most responses are worker-produced TASK_RESULT payloads (always have
    task_id/status). But dispatch_assigned_task can also hand back a
    master-generated error -- e.g. WORKER_UNREACHABLE when a worker's
    connection dies mid-dispatch -- and drain_pending_tasks' aggregate
    response list can contain BOTH a dead attempt's error and a
    successful retry's TASK_RESULT for the very same task_id in one call.
    Building this with .get() (never direct indexing) means a response
    that can't be identified is simply skipped rather than crashing the
    whole collection -- its task just looks "missing" to the caller.
    """
    responses_by_task_id: dict[str, dict] = {}
    for response in responses:
        task_id = response.get("payload", {}).get("task_id")
        if task_id is not None:
            responses_by_task_id[task_id] = response
    return responses_by_task_id


def collect_map_results(tasks: list[Task], responses: list[dict]) -> list:
    """Reassemble the full mapped list from MAP task responses.

    `responses` (e.g. from drain_pending_tasks) arrive in whatever order
    workers happened to finish in, which can differ from partition order --
    concurrent dispatch means a later partition can complete before an
    earlier one. Reordering by `tasks` (partition order) is what makes the
    result correct regardless of completion order.

    Raises ValueError if any task's response is missing or unsuccessful.
    """
    responses_by_task_id = _index_responses_by_task_id(responses)

    combined = []
    for task in tasks:
        response = responses_by_task_id.get(task.task_id)
        if response is None:
            raise ValueError(f"No response received for task {task.task_id}")

        payload = response["payload"]
        if payload.get("status") != "success":
            raise ValueError(f"Map task {task.task_id} failed: {payload.get('message') or payload.get('code')}")

        combined.extend(payload["result"])

    return combined


def build_intermediate_results(job_id: str, tasks: list[Task], responses: list[dict]) -> list[IntermediateResult]:
    """Turn a Map job's task responses into an ordered list of
    IntermediateResult, one per partition (partition_id = index in `tasks`,
    the same partition order build_map_job produced).

    Unlike collect_map_results (which raises on any failure or missing
    response and flattens straight to a final list), this tolerates
    failed/missing partitions by recording them as ERROR results instead
    of raising, and keeps each partition's provenance (worker_id, attempt)
    -- useful for inspecting a job's health, or for a later Shuffle/Reduce
    stage that needs structured input rather than a flat list.
    """
    responses_by_task_id = _index_responses_by_task_id(responses)

    results = []
    for partition_id, task in enumerate(tasks):
        response = responses_by_task_id.get(task.task_id)

        if response is None:
            results.append(
                IntermediateResult(
                    job_id=job_id,
                    partition_id=partition_id,
                    task_id=task.task_id,
                    status=ResultStatus.ERROR,
                    message="No response received",
                    worker_id=task.assigned_worker_id,
                    attempt=task.attempt,
                )
            )
            continue

        payload = response["payload"]
        attempt = payload.get("attempt", task.attempt)

        if payload.get("status") == ResultStatus.SUCCESS:
            results.append(
                IntermediateResult(
                    job_id=job_id,
                    partition_id=partition_id,
                    task_id=task.task_id,
                    status=ResultStatus.SUCCESS,
                    data=payload["result"],
                    worker_id=task.assigned_worker_id,
                    attempt=attempt,
                )
            )
        else:
            results.append(
                IntermediateResult(
                    job_id=job_id,
                    partition_id=partition_id,
                    task_id=task.task_id,
                    status=ResultStatus.ERROR,
                    message=payload.get("message") or payload.get("code"),
                    worker_id=task.assigned_worker_id,
                    attempt=attempt,
                )
            )

    return results
