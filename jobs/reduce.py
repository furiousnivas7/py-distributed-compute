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

from typing import Union

from common.models import Task
from jobs.models import ExecutionSpec
from master.scheduler import Scheduler
from worker import serialization
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


def build_reduce_job(
    scheduler: Scheduler, job_id: str, grouped: dict, operation: Union[str, ExecutionSpec]
) -> dict[str, Task]:
    """Submit one REDUCE task per key in `grouped`.

    `operation` is either a plain string -- a built-in (SUM, COUNT, MAX,
    MIN) or the name of a function registered via worker.registry, called
    as fn(values) with the whole per-key value list (matching the
    built-ins' own aggregate calling convention); worker.executor.
    execute_reduce falls back to the registry for whatever isn't a
    built-in -- or (Phase 10.5) an ExecutionSpec.serialized(fn) for an
    arbitrary callable that was never registered anywhere. One
    orchestration API for all three cases: a plain string is shorthand
    for ExecutionSpec.registered(that string) (see ExecutionSpec.coerce),
    so every existing caller passing a bare operation name keeps working
    completely unchanged.

    Returns a dict mapping each key to its submitted Task -- keyed by key
    rather than an ordered list, since Reduce has no partition-order
    equivalent (each key's reduction is independent of every other key's).
    """
    spec = ExecutionSpec.coerce(operation)
    encoded = (
        serialization.encode_for_wire(serialization.serialize_callable(spec.fn))
        if spec.execution_mode == "serialized_callable"
        else None
    )

    tasks = {}
    for key, values in grouped.items():
        task_id = f"{job_id}-reduce-{key}"
        if encoded is not None:
            payload = {"execution_mode": "serialized_callable", "callable": encoded, "key": key, "values": values}
        else:
            payload = {"operation": spec.operation, "key": key, "values": values}
        task = scheduler.submit_task(task_id, "REDUCE", payload)
        tasks[key] = task

    return tasks


def collect_reduce_results(tasks: dict[str, Task], responses: list[dict]) -> dict:
    """Reassemble {key: reduced_value} from real REDUCE task responses.

    Raises ValueError if any key's task response is missing or
    unsuccessful -- a silently-dropped key would corrupt the final result
    with no indication anything went wrong, so Reduce failure must be
    explicit rather than tolerated like Shuffle's partial-failure handling.
    """
    # Most responses are worker-produced TASK_RESULT payloads (always have
    # task_id/status). But dispatch_assigned_task can also hand back a
    # master-generated error -- e.g. WORKER_UNREACHABLE when a worker's
    # connection dies mid-dispatch -- and drain_pending_tasks' aggregate
    # response list can contain BOTH a dead attempt's error and a
    # successful retry's TASK_RESULT for the same task_id in one call.
    # Building this with .get() (never direct indexing) means an
    # unidentifiable response is simply skipped rather than crashing the
    # whole collection -- its key just looks "missing" below.
    responses_by_task_id: dict[str, dict] = {}
    for response in responses:
        task_id = response.get("payload", {}).get("task_id")
        if task_id is not None:
            responses_by_task_id[task_id] = response

    reduced = {}
    for key, task in tasks.items():
        response = responses_by_task_id.get(task.task_id)
        if response is None:
            raise ValueError(f"No response received for key {key!r} (task {task.task_id})")

        payload = response["payload"]
        if payload.get("status") != "success":
            raise ValueError(f"Reduce for key {key!r} failed: {payload.get('message') or payload.get('code')}")

        reduced[key] = payload["result"]

    return reduced
