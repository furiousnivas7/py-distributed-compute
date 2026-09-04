"""Shared data models."""

from dataclasses import dataclass


@dataclass
class Worker:
    worker_id: str
    host: str
    port: int
