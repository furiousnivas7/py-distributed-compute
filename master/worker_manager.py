"""In-memory registry of workers known to the master."""

import time

from common.models import Worker, WorkerStatus


class DuplicateWorkerError(ValueError):
    """Raised when registering a worker_id that is already registered."""


class WorkerNotFoundError(KeyError):
    """Raised when looking up a worker_id that isn't registered."""


VALID_WORKER_STATUSES = {
    WorkerStatus.REGISTERED,
    WorkerStatus.IDLE,
    WorkerStatus.BUSY,
    WorkerStatus.FAILED,
}


class WorkerManager:
    def __init__(self):
        self._workers: dict[str, Worker] = {}

    def has_worker(self, worker_id: str) -> bool:
        return worker_id in self._workers

    def register_worker(self, worker_id: str, host: str, port: int) -> Worker:
        if not isinstance(worker_id, str) or not worker_id:
            raise ValueError("worker_id must be a non-empty string")
        if not isinstance(host, str) or not host:
            raise ValueError("host must be a non-empty string")
        if not isinstance(port, int) or isinstance(port, bool) or not (0 < port < 65536):
            raise ValueError("port must be a valid TCP port number")

        if self.has_worker(worker_id):
            raise DuplicateWorkerError(f"Worker already registered: {worker_id}")

        worker = Worker(
            worker_id=worker_id,
            host=host,
            port=port,
            status=WorkerStatus.IDLE,
            last_heartbeat=time.time(),
        )
        self._workers[worker_id] = worker
        return worker

    def get_worker(self, worker_id: str) -> Worker | None:
        return self._workers.get(worker_id)

    def get_all_workers(self) -> list[Worker]:
        return list(self._workers.values())

    def update_status(self, worker_id: str, status: str) -> Worker:
        if status not in VALID_WORKER_STATUSES:
            raise ValueError(f"Invalid worker status: {status}")

        worker = self._workers.get(worker_id)
        if worker is None:
            raise WorkerNotFoundError(f"Unknown worker: {worker_id}")
        worker.status = status
        return worker

    def record_heartbeat(self, worker_id: str) -> Worker:
        worker = self._workers.get(worker_id)
        if worker is None:
            raise WorkerNotFoundError(f"Unknown worker: {worker_id}")
        worker.last_heartbeat = time.time()
        return worker

    def remove_worker(self, worker_id: str) -> None:
        if worker_id not in self._workers:
            raise WorkerNotFoundError(f"Unknown worker: {worker_id}")
        del self._workers[worker_id]

    def clear(self) -> None:
        self._workers.clear()
