"""Phase 10.1: submit a single registered-function CALL task through the
existing Scheduler / dispatch pipeline, mirroring jobs.map/jobs.reduce's
pattern of a thin submission helper over the same task_type machinery
ADD/MULTIPLY already use -- no separate scheduling logic, so CALL gets
assignment, concurrent dispatch, retry, and attempt protection for free.
"""

from common.models import Task
from master.scheduler import Scheduler


def submit_call(
    scheduler: Scheduler,
    task_id: str,
    function_name: str,
    args: list | None = None,
    kwargs: dict | None = None,
) -> Task:
    """Submit a CALL task invoking the function registered as
    `function_name` (see worker.registry) with JSON-safe `args`/`kwargs`.
    Returns the Task -- caller dispatches it same as any other task
    (e.g. via master.async_server.wait_for_tasks({task.task_id}))."""
    return scheduler.submit_task(
        task_id,
        "CALL",
        {"function": function_name, "args": args or [], "kwargs": kwargs or {}},
    )


def collect_call_result(response: dict):
    """Extract a CALL task's return value from its TASK_RESULT/ERROR
    response, raising ValueError with the worker's own error message on
    failure rather than returning some silent placeholder -- a caller
    asking for a specific function's result has no use for an ambiguous
    "it didn't work" outcome."""
    payload = response.get("payload", {})
    if payload.get("status") != "success":
        message = payload.get("message", "unknown error")
        raise ValueError(f"CALL task failed: {message}")
    return payload["result"]
