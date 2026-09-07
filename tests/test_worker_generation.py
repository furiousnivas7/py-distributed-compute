"""Phase 9.2.1: worker registration + connection generation/epoch.

Establishes the foundation the rest of Phase 9.2 (heartbeat lifecycle,
worker replacement, graceful shutdown, race conditions) builds on: a
worker_id is a stable LOGICAL identity that can survive a crash and
reconnect, but each physical connection that ever speaks for it is a
distinct GENERATION (see common/models.py's Worker.generation and
master/worker_manager.py's register_worker). The key invariant:

    A worker connection event may only modify state if it belongs to
    the currently active generation for that worker_id.

WorkerManager.register_worker enforces this at the registration layer: a
worker_id that's still REGISTERED/IDLE/BUSY (a live connection actively
representing it) rejects a second registration as a genuine conflict
(DuplicateWorkerError, unchanged from before Phase 9.2); only once it's
FAILED can the SAME worker_id re-register, which bumps its generation.

master/async_server.py's handle_worker_connection enforces the connection
half: the moment a NEW connection's registration succeeds and supersedes
an OLD one in `connections`, that old connection is proactively closed --
not just superseded in bookkeeping -- so any message it might otherwise
still deliver (a heartbeat, a stray reply) never gets a chance to mutate
state on behalf of a generation that's no longer current. See
tests/test_async_fault_tolerance.py's
test_reconnection_with_same_worker_id_after_failure_succeeds_with_new_generation
for the direct registration-layer proof of this; the tests here focus on
the task-ownership interaction across a real reconnect.
"""

import asyncio
import time

import pytest

from common.models import TaskStatus, WorkerStatus
from master import async_server, rpc_handler
from rpc import protocol
from rpc.async_connection import AsyncConnection
from rpc.async_rpc import receive_message, send_message, send_request
from rpc.protocol import build_message
from worker import async_worker
from worker.executor import execute_task


@pytest.fixture(autouse=True)
def reset_async_master_state():
    rpc_handler.worker_manager.clear()
    async_server.scheduler.clear()
    async_server.connections.clear()
    yield


async def start_master_server():
    server = await asyncio.start_server(async_server.handle_worker_connection, "127.0.0.1", 0)
    host, port = server.sockets[0].getsockname()[:2]
    return server, host, port


async def stop_worker(task: asyncio.Task) -> None:
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


async def run_single_crash_worker(host: str, port: int, worker_id: str, ready: asyncio.Event) -> None:
    """Registers, receives exactly one TASK without replying, signals
    `ready`, then disconnects -- a genuine connection-death failure."""
    reader, writer = await asyncio.open_connection(host, port)
    conn = AsyncConnection(reader, writer)
    await send_request(conn, protocol.PING)
    await send_request(conn, protocol.REGISTER, {"worker_id": worker_id, "host": "127.0.0.1", "port": 6000})

    while True:
        message = await receive_message(conn)
        if message["type"] == protocol.TASK:
            break

    ready.set()
    await conn.close()


async def run_delayed_reply_worker(
    host: str, port: int, worker_id: str, ready_event: asyncio.Event, release_event: asyncio.Event
) -> None:
    """Registers, receives one TASK, signals `ready_event`, then waits on
    `release_event` before finally executing and replying -- goes stale on
    its own (connection stays open, no heartbeat) rather than crashing.

    Whether the eventual reply's send actually raises once the master has
    invalidated (closed) this connection out from under it is a matter of
    OS/TCP write-after-remote-close timing, not something to assert on --
    it can succeed into a local buffer, fail immediately, or fail on a
    later write, depending on platform. Either way is an acceptable
    outcome here (the caller verifies the actually-important thing: that
    the reply never corrupts any state), so a resulting connection error
    is swallowed rather than propagated.
    """
    reader, writer = await asyncio.open_connection(host, port)
    conn = AsyncConnection(reader, writer)
    await send_request(conn, protocol.PING)
    await send_request(conn, protocol.REGISTER, {"worker_id": worker_id, "host": "127.0.0.1", "port": 6000})

    message = await receive_message(conn)
    assert message["type"] == protocol.TASK
    ready_event.set()
    await release_event.wait()

    payload = message["payload"]
    result = execute_task(payload["task_type"], payload["task_payload"])
    response = build_message(
        protocol.TASK_RESULT,
        message["request_id"],
        {"task_id": payload["task_id"], "attempt": payload.get("attempt", 1), **result},
    )
    try:
        await send_message(conn, response)
        await conn.close()
    except (ConnectionError, OSError):
        pass


def test_reconnect_after_connection_death_gets_new_generation_and_takes_fresh_work():
    """worker-1 crashes mid-task (connection death); a fresh connection
    reconnects under the SAME worker_id afterward. It must get a bumped
    generation and be able to receive and complete a brand-new task
    normally -- reconnection is a full recovery, not just a registration
    formality."""

    async def scenario():
        server, host, port = await start_master_server()
        ready = asyncio.Event()
        crashing_task = asyncio.create_task(run_single_crash_worker(host, port, "worker-1", ready))
        await async_server.wait_for_workers(1)

        try:
            first_generation = async_server.worker_manager.get_worker("worker-1").generation
            assert first_generation == 1

            # Manual assign+dispatch, not wait_for_tasks: with only one
            # worker in play and MAX_TASK_ATTEMPTS=3, a single crash just
            # requeues this task to PENDING with no IDLE worker left to
            # retry it on -- it would never reach a terminal state for
            # wait_for_tasks to resolve on. This test only needs THIS
            # attempt's own WORKER_UNREACHABLE outcome.
            doomed = async_server.scheduler.submit_task("doomed", "ADD", {"a": 1, "b": 1})
            assigned = async_server.scheduler.assign_next_pending_task()
            assert assigned.assigned_worker_id == "worker-1"
            dispatch = asyncio.create_task(async_server.dispatch_assigned_task(doomed))
            await asyncio.wait_for(ready.wait(), timeout=5)
            doomed_response = await dispatch

            assert async_server.worker_manager.get_worker("worker-1").status == WorkerStatus.FAILED

            rescue_task = asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-1"))
            deadline = time.monotonic() + 5
            while (
                async_server.worker_manager.get_worker("worker-1").generation == first_generation
                and time.monotonic() < deadline
            ):
                await asyncio.sleep(0.02)

            worker = async_server.worker_manager.get_worker("worker-1")
            assert worker.generation == first_generation + 1
            assert worker.status == WorkerStatus.IDLE

            fresh = async_server.scheduler.submit_task("fresh", "MULTIPLY", {"a": 6, "b": 7})
            [fresh_response] = await async_server.wait_for_tasks({fresh.task_id})
            return doomed_response, fresh_response
        finally:
            await stop_worker(crashing_task)
            await stop_worker(rescue_task)
            server.close()
            await server.wait_closed()

    doomed_response, fresh_response = asyncio.run(scenario())
    assert doomed_response["payload"]["code"] == "WORKER_UNREACHABLE"
    assert fresh_response["payload"]["status"] == "success"
    assert fresh_response["payload"]["result"] == 42


def test_stale_reply_on_invalidated_connection_never_reaches_new_generation():
    """worker-1 holds a task and goes quiet (connection stays open, no
    heartbeat -- the classic staleness scenario). It's force-marked FAILED
    and its task reassigned to worker-2, which completes it. worker-1's
    worker_id THEN reconnects fresh (new generation) and takes new work.
    Only after all of that does the ORIGINAL, long-delayed connection's
    reply for the old attempt finally arrive -- it must not resolve or
    corrupt anything: the request_id it carries (attempt 1) can't match
    ANYTHING current, and the connection itself was already invalidated
    (closed) once the new generation registered, so this reply is being
    sent on a connection the master has already discarded."""

    async def scenario():
        server, host, port = await start_master_server()

        ready = asyncio.Event()
        release = asyncio.Event()
        worker1_task = asyncio.create_task(run_delayed_reply_worker(host, port, "worker-1", ready, release))
        await async_server.wait_for_workers(1)
        worker2_task = asyncio.create_task(
            async_worker.run_worker(host, port, worker_id="worker-2", heartbeat_interval=0.1)
        )
        await async_server.wait_for_workers(2)

        try:
            task = async_server.scheduler.submit_task("flaky", "ADD", {"a": 3, "b": 4})
            waiter = asyncio.create_task(async_server.wait_for_tasks({task.task_id}))
            await asyncio.wait_for(ready.wait(), timeout=5)

            async_server.worker_manager.get_worker("worker-1").last_heartbeat = time.time() - 10
            stale = async_server.worker_manager.get_stale_workers(async_server.HEARTBEAT_TIMEOUT)
            assert any(w.worker_id == "worker-1" for w in stale)
            async_server.scheduler.requeue_tasks_for_worker("worker-1")

            [response] = await asyncio.wait_for(waiter, timeout=5)
            assert response["payload"]["result"] == 7
            assert response["payload"]["attempt"] == 2

            # worker-1's worker_id reconnects fresh, well before its old
            # (still-open, still-pending-a-reply) connection ever replies.
            rescue_task = asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-1"))
            deadline = time.monotonic() + 5
            while (
                async_server.worker_manager.get_worker("worker-1").generation < 2 and time.monotonic() < deadline
            ):
                await asyncio.sleep(0.02)
            assert async_server.worker_manager.get_worker("worker-1").generation == 2

            fresh = async_server.scheduler.submit_task("fresh-after-reconnect", "MULTIPLY", {"a": 5, "b": 5})
            [fresh_response] = await async_server.wait_for_tasks({fresh.task_id})
            assert fresh_response["payload"]["result"] == 25

            # NOW let the ORIGINAL (long-invalidated) connection's reply
            # arrive. The master closed that connection outright once
            # worker-1's new generation took over, so this reply has
            # nowhere valid to land -- give it a moment to (fail to) do
            # anything, then verify neither the old task nor the new
            # generation's own work was disturbed.
            release.set()
            await asyncio.wait_for(worker1_task, timeout=5)
            await asyncio.sleep(0.1)

            return task, fresh_response
        finally:
            await stop_worker(worker1_task)
            await stop_worker(worker2_task)
            await stop_worker(rescue_task)
            server.close()
            await server.wait_closed()

    task, fresh_response = asyncio.run(scenario())
    assert task.status == TaskStatus.COMPLETED
    assert task.assigned_worker_id == "worker-2"
    assert task.attempt == 2
    assert fresh_response["payload"]["result"] == 25
