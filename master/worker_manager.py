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
    WorkerStatus.DRAINING,
    WorkerStatus.STOPPED,
}

# A worker_id in either of these states has no live connection currently
# representing it, so a new registration for it is a legitimate reconnect
# (replacement, bumped generation) rather than a conflict. FAILED is a
# crash; STOPPED is an intentional, clean exit -- both leave the worker_id
# free to come back.
REPLACEABLE_WORKER_STATUSES = {WorkerStatus.FAILED, WorkerStatus.STOPPED}


class WorkerManager:
    def __init__(self):
        self._workers: dict[str, Worker] = {}

    def has_worker(self, worker_id: str) -> bool:
        return worker_id in self._workers

    def register_worker(self, worker_id: str, host: str, port: int) -> Worker:
        """Register worker_id, or -- if it already exists and is FAILED or
        STOPPED (see REPLACEABLE_WORKER_STATUSES) -- replace it in place
        with a new generation (see Worker.generation).

        worker_id is a stable LOGICAL identity a worker keeps across
        reconnects; generation identifies which physical connection
        currently speaks for it. Replacement is only allowed once the
        existing entry has no live connection representing it -- FAILED
        (crashed) or STOPPED (cleanly finished draining and disconnected,
        Phase 9.2.3). A worker_id that's REGISTERED/IDLE/BUSY/DRAINING
        still has a live connection, so a second registration attempt for
        it is a genuine conflict (still rejected as DuplicateWorkerError,
        unchanged from before) rather than a legitimate reconnect. Bumping
        generation rather than replacing the Worker object outright keeps
        every existing get_worker() reference (and the object identity
        tests may hold onto) pointing at the current state, matching how
        Task/Worker are mutated in place everywhere else in this codebase.
        """
        if not isinstance(worker_id, str) or not worker_id:
            raise ValueError("worker_id must be a non-empty string")
        if not isinstance(host, str) or not host:
            raise ValueError("host must be a non-empty string")
        if not isinstance(port, int) or isinstance(port, bool) or not (0 < port < 65536):
            raise ValueError("port must be a valid TCP port number")

        existing = self._workers.get(worker_id)
        if existing is not None:
            if existing.status not in REPLACEABLE_WORKER_STATUSES:
                raise DuplicateWorkerError(f"Worker already registered: {worker_id}")

            existing.host = host
            existing.port = port
            existing.status = WorkerStatus.IDLE
            existing.last_heartbeat = time.time()
            existing.generation += 1
            return existing

        worker = Worker(
            worker_id=worker_id,
            host=host,
            port=port,
            status=WorkerStatus.IDLE,
            last_heartbeat=time.time(),
            generation=1,
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

    def get_stale_workers(self, timeout: float) -> list[Worker]:
        """Mark FAILED and return every worker whose last heartbeat is older
        than `timeout` seconds. A worker that has never sent a heartbeat
        (last_heartbeat is None) is skipped, not treated as stale. So is
        one that's already STOPPED (Phase 9.2.3): it cleanly finished
        draining and disconnected on purpose, so its heartbeat naturally
        stopped too -- that's expected, not a failure, and must not be
        reclassified as one just because time passed."""
        now = time.time()
        stale_workers = []

        for worker in self._workers.values():
            if worker.last_heartbeat is None or worker.status == WorkerStatus.STOPPED:
                continue

            if now - worker.last_heartbeat > timeout:
                worker.status = WorkerStatus.FAILED
                stale_workers.append(worker)

        return stale_workers

    def remove_worker(self, worker_id: str) -> None:
        if worker_id not in self._workers:
            raise WorkerNotFoundError(f"Unknown worker: {worker_id}")
        del self._workers[worker_id]

    def clear(self) -> None:
        self._workers.clear()
