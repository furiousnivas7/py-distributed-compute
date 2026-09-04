"""Shared data models."""

from dataclasses import dataclass


class WorkerStatus:
    REGISTERED = "REGISTERED"
    IDLE = "IDLE"
    BUSY = "BUSY"
    FAILED = "FAILED"


@dataclass
class Worker:
    worker_id: str
    host: str
    port: int
    status: str = WorkerStatus.REGISTERED
