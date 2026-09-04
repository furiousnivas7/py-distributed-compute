"""Shared data models."""

from dataclasses import dataclass


class WorkerStatus:
    REGISTERED = "REGISTERED"
    IDLE = "IDLE"
    BUSY = "BUSY"
    FAILED = "FAILED"


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


@dataclass
class Task:
    task_id: str
    task_type: str
    payload: dict
    status: str = TaskStatus.PENDING
    assigned_worker_id: str | None = None
