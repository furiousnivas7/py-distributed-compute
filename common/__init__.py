"""Public API for the shared data models every other package's public
surface returns or accepts: ``Task``/``TaskStatus`` (``master.Scheduler``,
every ``jobs.*`` submission helper) and ``Worker``/``WorkerStatus``
(``master.WorkerManager``).

Stable import surface -- ``common.models`` keeps working exactly as
before; this module only adds re-exports.
"""

from common.models import Task, TaskStatus, Worker, WorkerStatus

__all__ = [
    "Task",
    "TaskStatus",
    "Worker",
    "WorkerStatus",
]
