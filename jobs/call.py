"""Phase 10.2: submit a task invoking a registered function (see
worker.registry) through the existing Scheduler / dispatch pipeline,
mirroring jobs.map/jobs.reduce's pattern of a thin submission helper over
the same task_type machinery ADD/MULTIPLY already use.

A registered operation IS a task_type -- submit_call is barely more than
Scheduler.submit_task itself; it exists for the "operation name + keyword
arguments" framing (mirroring worker.executor.execute_registered's
fn(**payload) calling convention) rather than because CALL needs any
dispatch logic of its own. There isn't any: it gets assignment, retry,
and attempt protection for free through the same pipeline every other
task_type uses.
"""

from common.models import Task
from master.scheduler import Scheduler


def submit_call(scheduler: Scheduler, task_id: str, operation: str, **kwargs) -> Task:
    """Submit a task invoking the function registered as `operation`
    (see worker.registry.register) with `kwargs` as its arguments.
    Returns the Task -- caller dispatches it same as any other task
    (e.g. via master.async_server.wait_for_tasks({task.task_id}))."""
    return scheduler.submit_task(task_id, operation, kwargs)


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
