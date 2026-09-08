"""Submit function-execution tasks through the existing Scheduler /
dispatch pipeline, mirroring jobs.map/jobs.reduce's pattern of a thin
submission helper over the same task_type machinery ADD/MULTIPLY already
use -- no dispatch logic of its own in either helper below; both get
assignment, retry, and attempt protection for free through the same
pipeline every other task_type uses.

Two ways to submit a registered-function call, both landing on the same
worker.executor.execute_registered:

- submit_call() (Phase 10.2): task_type IS the operation name, payload IS
  the keyword arguments directly. The shorter form for the common case.
- submit_registered_call() (Phase 10.3): task_type is EXECUTE, payload is
  the explicit {"execution_mode": "registered", ...} envelope -- the same
  envelope shape submit_serialized_call() uses for the OTHER execution
  mode, so code that needs to handle "either kind of function task
  generically" (log it, retry it, inspect it) can do so via one shared
  shape instead of two unrelated ones. Neither form is more "correct"
  than the other; submit_call stays exactly as it was, unchanged.
"""

from common.models import Task
from master.scheduler import Scheduler
from worker import serialization


def submit_call(scheduler: Scheduler, task_id: str, operation: str, **kwargs) -> Task:
    """Submit a task invoking the function registered as `operation`
    (see worker.registry.register) with `kwargs` as its arguments."""
    return scheduler.submit_task(task_id, operation, kwargs)


def submit_registered_call(scheduler: Scheduler, task_id: str, operation: str, **kwargs) -> Task:
    """Envelope form of submit_call (Phase 10.3) -- same operation lookup
    and fn(**kwargs) calling convention, reached via the explicit EXECUTE
    task_type + 'registered' execution_mode instead of a bare task_type."""
    payload = {"execution_mode": "registered", "operation": operation, "payload": kwargs}
    return scheduler.submit_task(task_id, "EXECUTE", payload)


def submit_serialized_call(
    scheduler: Scheduler,
    task_id: str,
    fn,
    args: list | None = None,
    kwargs: dict | None = None,
) -> Task:
    """Submit a task invoking the arbitrary callable `fn` -- serialized
    here (see worker.serialization), shipped over the wire, and executed
    on whichever worker picks up the task. `fn` need not be pre-registered
    or even importable by the worker; it's serialized by value (including
    closures -- see worker.serialization's module docstring on
    cloudpickle), not by reference. See that module's docstring for the
    trust-boundary implications: this is remote code execution by design,
    appropriate for a trusted compute cluster, not something to submit
    code from an untrusted source into.
    """
    encoded = serialization.encode_for_wire(serialization.serialize_callable(fn))
    payload = {
        "execution_mode": "serialized_callable",
        "callable": encoded,
        "args": args or [],
        "kwargs": kwargs or {},
    }
    return scheduler.submit_task(task_id, "EXECUTE", payload)


def collect_call_result(response: dict):
    """Extract a function-execution task's return value from its
    TASK_RESULT/ERROR response, raising ValueError with the worker's own
    error message on failure rather than returning some silent
    placeholder -- a caller asking for a specific result has no use for
    an ambiguous "it didn't work" outcome."""
    payload = response.get("payload", {})
    if payload.get("status") != "success":
        message = payload.get("message", "unknown error")
        raise ValueError(f"CALL task failed: {message}")
    return payload["result"]
