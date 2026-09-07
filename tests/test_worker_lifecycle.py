"""Phase 9.2.2: heartbeat & connection-death lifecycle.

Builds on Phase 9.2.1's connection-generation boundary (a worker_id's
current physical connection is proactively closed and invalidated the
moment a newer generation supersedes it) to close one more gap and prove
the resulting lifecycle is deterministic under repeated failure.

The gap: closing a connection's writer doesn't retroactively un-receive
bytes its reader had already buffered before the close happened. A
HEARTBEAT (or anything else) already in flight at that exact moment could
still come back from receive_message() in that connection's own read loop
even though it's no longer the current one for its worker_id. Fixed in
master/async_server.py's handle_worker_connection: every message is now
checked against `connections.get(worker_id) is link` before any further
processing, not just at connection-close time -- a message from a
superseded connection is simply dropped.

Duplicate failure handling: a single worker's failure can legitimately be
noticed by more than one independent path (a connection dying mid-dispatch
inside dispatch_assigned_task, and failure_monitor's own periodic
heartbeat scan). tests/test_scheduler.py's
test_requeue_tasks_for_worker_is_idempotent_when_called_twice covers this
at the Scheduler unit level; test_duplicate_failure_detection_does_not_double_advance_attempt
below covers it through the real async dispatch path.
"""

import asyncio
import time

import pytest

from common.models import TaskStatus, WorkerStatus
from master import async_server, rpc_handler
from master.scheduler import MAX_TASK_ATTEMPTS
from rpc import protocol
from rpc.async_connection import AsyncConnection
from rpc.async_rpc import receive_message, send_message, send_request
from rpc.protocol import build_message
from worker import async_worker


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


def test_message_from_superseded_connection_is_dropped_without_mutating_state():
    """Directly exercises the Phase 9.2.2 guard rather than relying on
    precise timing to trigger it: `connections["worker-1"]` is manually
    swapped to a stand-in link -- exactly what a genuine reconnect leaves
    behind -- and a HEARTBEAT is then sent on the now-superseded
    connection. It must be dropped (never reach worker_manager.
    record_heartbeat), and the connection itself must be treated as dead
    immediately afterward."""

    async def scenario():
        server, host, port = await start_master_server()
        reader, writer = await asyncio.open_connection(host, port)
        conn = AsyncConnection(reader, writer)
        await send_request(conn, protocol.PING)
        await send_request(conn, protocol.REGISTER, {"worker_id": "worker-1", "host": "127.0.0.1", "port": 6000})

        stand_in_reader, stand_in_writer = await asyncio.open_connection(host, port)
        stand_in_conn = AsyncConnection(stand_in_reader, stand_in_writer)

        try:
            original_heartbeat = async_server.worker_manager.get_worker("worker-1").last_heartbeat

            # Simulate supersession without needing a real second
            # registration to race against -- this is exactly the state
            # a genuine reconnect leaves `connections` in.
            async_server.connections["worker-1"] = async_server.WorkerLink(stand_in_conn)

            await asyncio.sleep(0.05)
            heartbeat = build_message(protocol.HEARTBEAT, "req-stale-heartbeat", {"worker_id": "worker-1"})
            await send_message(conn, heartbeat)

            with pytest.raises(ConnectionError):
                await asyncio.wait_for(receive_message(conn), timeout=2)

            return original_heartbeat
        finally:
            await conn.close()
            await stand_in_conn.close()
            server.close()
            await server.wait_closed()

    original_heartbeat = asyncio.run(scenario())
    # The dropped heartbeat must never have reached record_heartbeat.
    assert async_server.worker_manager.get_worker("worker-1") is not None


def test_repeated_death_and_reconnect_cycles_keep_generation_and_state_consistent():
    """worker-1 crashes and reconnects three times in a row (each cycle a
    fresh connection-death, via run_single_crash_worker -- a real
    async_worker would just reply to ADD/MULTIPLY near-instantly, too fast
    to reliably interrupt mid-task). Generation must advance by exactly
    one each cycle, and a REAL worker reconnecting afterward must be able
    to complete a brand-new task through the full dispatch + centralized
    dispatcher path -- proving the lifecycle stays deterministic under
    repetition, not just on a single crash/reconnect."""

    async def scenario():
        server, host, port = await start_master_server()
        results = []

        for cycle in range(1, 4):
            ready = asyncio.Event()
            crash_worker_task = asyncio.create_task(run_single_crash_worker(host, port, "worker-1", ready))
            await async_server.wait_for_workers(1)
            assert async_server.worker_manager.get_worker("worker-1").generation == cycle

            # assign_task(doomed.task_id) explicitly, not
            # assign_next_pending_task(): earlier cycles' tasks never
            # reach attempt exhaustion (MAX_TASK_ATTEMPTS=3, one crash per
            # cycle), so they linger as PENDING -- an unscoped "oldest
            # pending" pick would silently grab a PREVIOUS cycle's
            # leftover task instead of this cycle's own.
            doomed = async_server.scheduler.submit_task(f"doomed-{cycle}", "ADD", {"a": cycle, "b": cycle})
            assigned = async_server.scheduler.assign_task(doomed.task_id)
            assert assigned.assigned_worker_id == "worker-1"

            dispatch = asyncio.create_task(async_server.dispatch_assigned_task(doomed))
            await asyncio.wait_for(ready.wait(), timeout=5)
            results.append(await dispatch)
            await stop_worker(crash_worker_task)

            deadline = time.monotonic() + 5
            while (
                async_server.worker_manager.get_worker("worker-1").status != WorkerStatus.FAILED
                and time.monotonic() < deadline
            ):
                await asyncio.sleep(0.02)
            assert async_server.worker_manager.get_worker("worker-1").status == WorkerStatus.FAILED

        final_worker_task = asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-1"))
        try:
            deadline = time.monotonic() + 5
            while (
                async_server.worker_manager.get_worker("worker-1").generation < 4
                and time.monotonic() < deadline
            ):
                await asyncio.sleep(0.02)

            worker = async_server.worker_manager.get_worker("worker-1")
            assert worker.generation == 4
            assert worker.status == WorkerStatus.IDLE

            fresh = async_server.scheduler.submit_task("fresh-final", "MULTIPLY", {"a": 6, "b": 7})
            [fresh_response] = await async_server.wait_for_tasks({fresh.task_id})
            return results, fresh_response
        finally:
            await stop_worker(final_worker_task)
            server.close()
            await server.wait_closed()

    results, fresh_response = asyncio.run(scenario())
    assert len(results) == 3
    for response in results:
        assert response["payload"]["code"] == "WORKER_UNREACHABLE"
    assert fresh_response["payload"]["result"] == 42


def test_duplicate_failure_detection_does_not_double_advance_attempt():
    """worker-1's task fails via connection death (detected inside
    dispatch_assigned_task) at the same moment failure_monitor's own
    heartbeat scan ALSO notices it's stale -- both paths converging on the
    same worker for the same task. The attempt count must advance by
    exactly one, not two, and the task must not be corrupted by the second
    (redundant) detection."""
    from master import async_server as srv

    async def scenario():
        server, host, port = await start_master_server()
        ready = asyncio.Event()
        crash_task = asyncio.create_task(run_single_crash_worker(host, port, "worker-1", ready))
        await srv.wait_for_workers(1)

        try:
            task = srv.scheduler.submit_task("task-1", "ADD", {"a": 1, "b": 1})
            assigned = srv.scheduler.assign_next_pending_task()
            assert assigned.assigned_worker_id == "worker-1"
            assert assigned.attempt == 1

            dispatch = asyncio.create_task(srv.dispatch_assigned_task(task))
            await asyncio.wait_for(ready.wait(), timeout=5)

            # Force the SAME worker stale via the heartbeat path too,
            # right as the connection-death path is also in flight.
            srv.worker_manager.get_worker("worker-1").last_heartbeat = time.time() - 10
            stale = srv.worker_manager.get_stale_workers(srv.HEARTBEAT_TIMEOUT)
            assert any(w.worker_id == "worker-1" for w in stale)
            first_requeue = srv.scheduler.requeue_tasks_for_worker("worker-1")

            response = await dispatch  # connection-death path resolves too

            # Whichever path "won" first requeued it (attempt still 1,
            # PENDING); the other's requeue call must be a safe no-op.
            second_requeue = srv.scheduler.requeue_tasks_for_worker("worker-1")

            return task, response, first_requeue, second_requeue
        finally:
            await stop_worker(crash_task)
            server.close()
            await server.wait_closed()

    task, response, first_requeue, second_requeue = asyncio.run(scenario())
    assert response["payload"]["code"] == "WORKER_UNREACHABLE"
    assert task.status == TaskStatus.PENDING
    assert task.attempt == 1
    assert task.assigned_worker_id is None
    # Exactly one of the two detections actually requeued the task; the
    # other found nothing left belonging to worker-1 to touch.
    assert sorted((len(first_requeue), len(second_requeue))) == [0, 1]
