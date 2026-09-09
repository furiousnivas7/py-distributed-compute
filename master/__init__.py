"""Public API for the master runtime.

This is the STABLE, documented import surface for anything outside this
package -- deep imports (``master.scheduler``, ``master.async_server``,
...) keep working exactly as before; nothing here moves or renames a
single file. This module only adds re-exports on top of what already
exists, so it's purely additive and cannot break an existing caller.

Only the async runtime (Phase 8.9 onward) is exported here.
``master/server.py`` is this project's earlier synchronous/threaded
master, superseded by ``master/async_server.py`` but kept around for its
own pre-existing tests -- it is intentionally not part of the public API,
so a new caller doesn't have to choose between two competing "the" master
implementations.

Typical usage::

    from master import scheduler, start_master, stop_master, wait_for_tasks

    server = await start_master(host, port)
    task = scheduler.submit_task("t1", "ADD", {"a": 1, "b": 2})
    [response] = await wait_for_tasks({task.task_id})
    await stop_master()

``scheduler`` is a process-wide singleton (see ``master/async_server.py``)
-- every task submitted through it is what ``start_master``'s dispatcher
and ``wait_for_tasks`` actually operate on. A caller that wants an
independent, non-singleton ``Scheduler`` (e.g. for ``jobs.map_reduce.
run_map_reduce``'s own ``scheduler`` argument in a test, or a from-scratch
in-process pipeline) can construct one directly: ``Scheduler(WorkerManager())``.
"""

from master.async_server import scheduler, start_master, stop_master, wait_for_tasks, wait_for_workers
from master.scheduler import NoAvailableWorkerError, Scheduler, TaskNotFoundError
from master.worker_manager import DuplicateWorkerError, WorkerManager, WorkerNotFoundError

__all__ = [
    "scheduler",
    "start_master",
    "stop_master",
    "wait_for_tasks",
    "wait_for_workers",
    "Scheduler",
    "TaskNotFoundError",
    "NoAvailableWorkerError",
    "WorkerManager",
    "DuplicateWorkerError",
    "WorkerNotFoundError",
]
