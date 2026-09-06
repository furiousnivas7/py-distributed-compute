"""Reduce: aggregate Shuffle's grouped key/value data into a final result.

Two layers, deliberately kept separate:

- reduce_grouped(): pure, local computation over an already-grouped dict
  (jobs.shuffle.shuffle()'s output). No task dispatch -- matches how
  jobs.shuffle.shuffle() itself works, and is the "Reduce API" in the sense
  of `reduce_grouped(grouped, "SUM")`.

- build_reduce_job() / collect_reduce_results(): the same computation, but
  actually dispatched as REDUCE tasks through the existing Scheduler /
  dispatch pipeline, one task per key -- proving REDUCE is a real task
  type the distributed engine can run, not a special case. No distributed
  scheduling logic is added here: this is the same one-task-per-unit-of-
  work pattern jobs.map.build_map_job already uses, just keyed by key
  instead of ordered by partition index (Reduce has no inherent ordering).

Unlike jobs.shuffle.shuffle() (which silently skips failed/missing Map
partitions -- reasonable, since Shuffle is just reorganizing whatever
Map succeeded at), collect_reduce_results() raises on any failed or
missing key. A silently-dropped key here would corrupt the final result
with no indication anything was wrong, so Reduce failure must be explicit.
"""

from common.models import Task
from master.scheduler import Scheduler
from worker.executor import REDUCE_OPERATIONS, ExecutionError, execute_reduce


def reduce_grouped(grouped: dict, operation: str) -> dict:
    """Apply a named REDUCE operation to every key's value list, locally.

    Reuses worker.executor.execute_reduce so local and dispatched Reduce
    can never compute a different answer for the same (operation, values).
    Raises ValueError immediately on an unknown operation or invalid
    values for the given key -- there's no separate "task" here to fail
    independently of the caller finding out.
    """
    if operation not in REDUCE_OPERATIONS:
        raise ValueError(f"Unsupported REDUCE operation: {operation}")

    reduced = {}
    for key, values in grouped.items():
        try:
            reduced[key] = execute_reduce({"operation": operation, "values": values})
        except ExecutionError as exc:
            raise ValueError(f"Reduce for key {key!r} failed: {exc}") from exc

    return reduced


def build_reduce_job(scheduler: Scheduler, job_id: str, grouped: dict, operation: str) -> dict[str, Task]:
    """Submit one REDUCE task per key in `grouped`.

    Returns a dict mapping each key to its submitted Task -- keyed by key
    rather than an ordered list, since Reduce has no partition-order
    equivalent (each key's reduction is independent of every other key's).
    """
    tasks = {}
    for key, values in grouped.items():
        task_id = f"{job_id}-reduce-{key}"
        task = scheduler.submit_task(task_id, "REDUCE", {"operation": operation, "key": key, "values": values})
        tasks[key] = task

    return tasks


def collect_reduce_results(tasks: dict[str, Task], responses: list[dict]) -> dict:
    """Reassemble {key: reduced_value} from real REDUCE task responses.

    Raises ValueError if any key's task response is missing or
    unsuccessful -- a silently-dropped key would corrupt the final result
    with no indication anything went wrong, so Reduce failure must be
    explicit rather than tolerated like Shuffle's partial-failure handling.
    """
    responses_by_task_id = {response["payload"]["task_id"]: response for response in responses}

    reduced = {}
    for key, task in tasks.items():
        response = responses_by_task_id.get(task.task_id)
        if response is None:
            raise ValueError(f"No response received for key {key!r} (task {task.task_id})")

        payload = response["payload"]
        if payload["status"] != "success":
            raise ValueError(f"Reduce for key {key!r} failed: {payload.get('message')}")

        reduced[key] = payload["result"]

    return reduced
