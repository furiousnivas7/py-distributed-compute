"""Shared data models."""

from dataclasses import dataclass


class WorkerStatus:
    REGISTERED = "REGISTERED"
    IDLE = "IDLE"
    BUSY = "BUSY"
    FAILED = "FAILED"
    # Phase 9.2.3 -- graceful shutdown. DRAINING: the worker asked to stop
    # (SHUTDOWN) but its current task, if any, is still allowed to finish;
    # it's excluded from new assignment (Scheduler.assign_task only picks
    # IDLE) without needing any change there. STOPPED: draining finished
    # and the worker disconnected cleanly -- a terminal state distinct
    # from FAILED so failure_monitor doesn't mistake an intentional exit
    # for a crash, but one that (like FAILED) still allows the same
    # worker_id to re-register later with a bumped generation.
    DRAINING = "DRAINING"
    STOPPED = "STOPPED"


class TaskStatus:
    PENDING = "PENDING"
    ASSIGNED = "ASSIGNED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass
class Worker:
    worker_id: str
    host: str
    port: int
    status: str = WorkerStatus.REGISTERED
    last_heartbeat: float | None = None
    # Bumped each time this worker_id successfully re-registers after being
    # marked FAILED (see WorkerManager.register_worker). worker_id alone is
    # a stable LOGICAL identity; generation distinguishes which physical
    # connection currently owns it, so state belonging to a superseded
    # connection can be told apart from the current one. See
    # master/async_server.py's handle_worker_connection for how the
    # connection layer enforces this.
    generation: int = 1


@dataclass
class Task:
    task_id: str
    task_type: str
    payload: dict
    status: str = TaskStatus.PENDING
    assigned_worker_id: str | None = None
    attempt: int = 0
