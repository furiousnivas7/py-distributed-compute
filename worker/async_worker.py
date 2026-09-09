"""Async TCP worker client: registers with the master, then executes TASK
requests and reports liveness, all on one persistent connection.

Unlike the threaded worker (which opens a second, short-lived connection
per heartbeat specifically to avoid two OS threads reading the same socket),
this worker sends HEARTBEAT as a fire-and-forget message on the SAME
connection its TASK/TASK_RESULT traffic uses. That's safe here because
serve_tasks() is the only coroutine that ever reads this connection -- the
heartbeat loop only ever writes to it, and asyncio guarantees only one
coroutine runs Python code at a time, so concurrent writes from two
coroutines can never interleave mid-message the way two OS threads could.
"""

import asyncio
import logging
import sys

from rpc import protocol
from rpc.async_connection import AsyncConnection
from rpc.async_rpc import new_request_id, receive_message, send_message, send_request
from rpc.protocol import build_message
from worker.backend import DirectBackend, ExecutionBackend
from worker.config import build_backend, resolve_backend_config, resolve_worker_runtime_config

logger = logging.getLogger(__name__)

MASTER_HOST = "127.0.0.1"
MASTER_PORT = 5000

WORKER_ID = "worker-1"
WORKER_HOST = "127.0.0.1"
WORKER_PORT = 6001

# Must stay well below master.async_server.HEARTBEAT_TIMEOUT (5.0s) -- see
# the same note on worker.worker.HEARTBEAT_INTERVAL_SECONDS.
HEARTBEAT_INTERVAL_SECONDS = 1.5


async def register(conn: AsyncConnection, worker_id: str, worker_host: str, worker_port: int) -> dict:
    return await send_request(
        conn, protocol.REGISTER, {"worker_id": worker_id, "host": worker_host, "port": worker_port}
    )


async def send_heartbeat(conn: AsyncConnection, worker_id: str) -> None:
    """Send a HEARTBEAT without waiting for its ack.

    serve_tasks()'s read loop owns this connection's only read; it'll see
    the HEARTBEAT_ACK show up as just another incoming message and ignore
    it (nothing here needs to correlate a heartbeat with its reply).
    """
    request = build_message(protocol.HEARTBEAT, new_request_id(), {"worker_id": worker_id})
    await send_message(conn, request)


async def send_shutdown(conn: AsyncConnection, worker_id: str) -> None:
    """Notify the master this worker intends to stop accepting new work.

    Fire-and-forget, like send_heartbeat -- the master decides when it's
    actually safe to close this connection (immediately if nothing is
    currently assigned, or once the in-flight task finishes) rather than
    this worker guessing at timing. See master/async_server.py's
    handle_worker_connection (the SHUTDOWN branch) and
    dispatch_assigned_task's DRAINING completion check.
    """
    request = build_message(protocol.SHUTDOWN, new_request_id(), {"worker_id": worker_id})
    await send_message(conn, request)


async def watch_for_shutdown(conn: AsyncConnection, worker_id: str, shutdown_event: asyncio.Event) -> None:
    """Wait for `shutdown_event`, then send SHUTDOWN once. Runs alongside
    serve_tasks() the same way the heartbeat loop does; serve_tasks()
    itself needs no changes; it just keeps running normally (able to
    finish serving a task already in flight, or receive one last
    legitimately-in-flight TASK that crossed with this SHUTDOWN on the
    wire) until the master closes the connection at the right moment."""
    await shutdown_event.wait()
    try:
        await send_shutdown(conn, worker_id)
    except (ConnectionError, OSError):
        pass


async def start_heartbeat_loop(
    conn: AsyncConnection,
    worker_id: str,
    stop_event: asyncio.Event,
    interval: float = HEARTBEAT_INTERVAL_SECONDS,
) -> None:
    """Send a HEARTBEAT every `interval` seconds until stop_event is set."""
    while True:
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
            return
        except asyncio.TimeoutError:
            pass

        try:
            await send_heartbeat(conn, worker_id)
        except (ConnectionError, OSError):
            return


async def serve_tasks(conn: AsyncConnection, backend: ExecutionBackend) -> None:
    """Read every message on this connection until it closes.

    Executes any TASK it receives (via `backend` -- Phase 11.1; see
    worker/backend.py) and replies with TASK_RESULT; silently ignores
    anything else it doesn't recognize (e.g. the HEARTBEAT_ACK for a
    heartbeat it sent, since those are fire-and-forget here).
    """
    while True:
        try:
            message = await receive_message(conn)
        except ConnectionError:
            return

        if message["type"] != protocol.TASK:
            continue

        task_payload = message["payload"]
        task_id = task_payload["task_id"]
        task_type = task_payload["task_type"]
        task_args = task_payload["task_payload"]
        attempt = task_payload.get("attempt", 1)

        result = await backend.execute(task_type, task_args)
        print(f"Executed task {task_id} (attempt {attempt}): {result}")

        response = build_message(
            protocol.TASK_RESULT,
            message["request_id"],
            {"task_id": task_id, "attempt": attempt, **result},
        )
        await send_message(conn, response)


async def run_worker(
    master_host: str,
    master_port: int,
    worker_id: str = WORKER_ID,
    worker_host: str = WORKER_HOST,
    worker_port: int = WORKER_PORT,
    heartbeat_interval: float = HEARTBEAT_INTERVAL_SECONDS,
    shutdown_event: asyncio.Event | None = None,
    backend: ExecutionBackend | None = None,
) -> None:
    """Run until the connection ends. `backend` (Phase 11.1; see
    worker/backend.py) selects HOW submitted tasks actually run --
    defaults to DirectBackend, an in-process call, identical to every
    worker's behavior before Phase 11. Passing a different backend (e.g.
    Phase 11.2's multiprocessing one) changes nothing about how tasks are
    submitted or what a result looks like; only how the worker internally
    carries the work out.

    Passing `shutdown_event` enables
    graceful shutdown (Phase 9.2.3): setting it requests that this worker
    stop accepting new work, finish whatever it's currently doing, and
    disconnect -- the master will close the connection at the right
    moment (see watch_for_shutdown/handle_worker_connection), so
    run_worker() itself doesn't need to guess when that is.

    API contract: this is fire-and-forget from the caller's side --
    setting `shutdown_event` does NOT wait for, or guarantee, that any
    task already submitted to the scheduler has actually been assigned to
    THIS worker yet. If the worker is still IDLE (nothing assigned) when
    the master processes SHUTDOWN, it stops immediately -- a task that
    was submitted just before but never reached this worker stays PENDING
    for someone else. A caller that needs a specific task to run on this
    worker before it drains must wait until that task has actually been
    assigned/dispatched (not merely submitted) before setting
    shutdown_event; there is deliberately no grace period here that would
    paper over that ordering with timing-dependent behavior instead.

    Without shutdown_event (the default), this worker behaves exactly as
    before -- serve_tasks() only ever ends via the connection dying, which
    run_worker()'s caller
    (e.g. a crash, or the process being killed) still controls."""
    if backend is None:
        backend = DirectBackend()
    elif not isinstance(backend, ExecutionBackend):
        # Phase 11.3: reject clearly and immediately -- before opening any
        # connection or touching the network at all -- rather than
        # failing confusingly later (an AttributeError from `backend.
        # start()`, or worse, silently doing nothing useful if the object
        # happens to have SOME but not all of the right method names).
        raise TypeError(
            f"backend must be an ExecutionBackend instance (or None for the "
            f"default DirectBackend), got {type(backend).__name__}"
        )

    # Phase 11.6: one INFO-level record before this worker touches the
    # network at all -- worker_id, where it's connecting to, and its
    # backend's full describe() (identity + capacity/capability metadata,
    # e.g. max_concurrency, max_workers/max_in_flight for a
    # MultiprocessingBackend) so a log reader can see how this worker is
    # configured without cross-referencing its startup code.
    logger.info(
        "worker_starting worker_id=%s master_host=%s master_port=%s backend=%s",
        worker_id,
        master_host,
        master_port,
        backend.describe(),
    )

    reader, writer = await asyncio.open_connection(master_host, master_port)
    conn = AsyncConnection(reader, writer)
    print("Connected to master")

    stop_heartbeat = asyncio.Event()
    heartbeat_task = None
    shutdown_watcher = None

    # Phase 11.3: backend.start()/stop() lifecycle is made deterministic
    # against failure at every stage, not just the happy path already
    # covered by Phase 11.1/11.2's tests:
    #   - start() failing must still close `conn` (the outer `finally`
    #     below) -- and gets a matching stop() attempt (see the inner
    #     except immediately below) in case start() partially initialized
    #     something before failing; every backend.stop() implementation
    #     so far tolerates being called on a backend that never fully
    #     started (DirectBackend has no state at all; MultiprocessingBackend
    #     checks `self._pool is None` and no-ops).
    #   - stop() failing (whether following a normal run or a failed
    #     start()) must not prevent `conn.close()` from still running.
    try:
        try:
            await backend.start()
        except Exception:
            await backend.stop()
            raise

        try:
            ping_response = await send_request(conn, protocol.PING)
            print(f"Status: {ping_response['payload'].get('status')}")

            register_response = await register(conn, worker_id, worker_host, worker_port)
            print(f"Status: {register_response['payload'].get('status')}")

            heartbeat_task = asyncio.create_task(
                start_heartbeat_loop(conn, worker_id, stop_heartbeat, heartbeat_interval)
            )
            if shutdown_event is not None:
                shutdown_watcher = asyncio.create_task(watch_for_shutdown(conn, worker_id, shutdown_event))

            await serve_tasks(conn, backend)
        finally:
            stop_heartbeat.set()
            if heartbeat_task is not None:
                await heartbeat_task
            if shutdown_watcher is not None:
                shutdown_watcher.cancel()
                try:
                    await shutdown_watcher
                except asyncio.CancelledError:
                    pass
            await backend.stop()
    finally:
        await conn.close()


def main() -> None:
    # Phase 11.8: only main() (the actual CLI entry point) configures
    # logging output -- a library caller (run_worker(), tests, anything
    # importing this module) must never have logging configuration
    # imposed on it just by importing worker.async_worker. Without this,
    # every INFO-level structured log added in Phase 11.6/11.7
    # (backend_created, backend_config_resolved, worker_starting, ...)
    # is silently dropped by logging's default WARNING threshold when run
    # as a plain script.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    # Phase 11.7: backend selection/configuration is a CLI-and-environment
    # concern, resolved here (the process entry point) -- run_worker()
    # itself still just takes an already-built `backend` argument
    # unchanged since Phase 11.1, so nothing about its own signature or
    # any of its callers (including every test that builds a backend
    # directly) needs to change.
    # Phase 11.8: a bad --backend/--max-workers/... value is a normal,
    # anticipated user mistake, not a bug -- reported as a clean one-line
    # message on stderr with exit code 1, not a raw traceback pointing
    # into resolve_backend_config's internals. resolve_backend_config's
    # ValueError message itself already says what was actually received
    # (Phase 11.6/11.7); this only changes how it's PRESENTED at the CLI
    # boundary, not its wording.
    try:
        config = resolve_backend_config()
        backend = build_backend(config)
        # Phase 12.3: WHERE/WHO this worker is (master host/port, its own
        # id/host/port) is now resolved the same CLI/env/defaults way as
        # the backend -- previously hardcoded module constants, meaning
        # two worker processes on the same machine (or one pointed at a
        # non-default master) required editing this file's source.
        runtime = resolve_worker_runtime_config()
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    # Phase 11.8: a plain print() here too, not just the structured log --
    # resolve_backend_config/backend.describe() already log this at INFO,
    # but this line guarantees an operator sees which backend got
    # selected even if they've redirected/filtered logging output,
    # matching this file's existing print()-for-human-visible-status-lines
    # convention (see the PING/REGISTER status prints in run_worker below).
    print(f"Selected backend: {backend.describe()}")
    print(
        f"Worker {runtime.worker_id} connecting to {runtime.master_host}:{runtime.master_port}"
    )
    asyncio.run(
        run_worker(
            runtime.master_host,
            runtime.master_port,
            worker_id=runtime.worker_id,
            worker_host=runtime.worker_host,
            worker_port=runtime.worker_port,
            backend=backend,
        )
    )


if __name__ == "__main__":
    main()
