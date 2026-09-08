"""Phase 9.2.3: graceful worker shutdown.

Adds a fourth worker lifecycle path alongside register/heartbeat-timeout/
connection-death (Phases 9.2.1-9.2.2):

    REGISTERED/IDLE/BUSY --SHUTDOWN--> DRAINING --(task finishes,
                                                     connection closes)--> STOPPED

A DRAINING worker is excluded from new assignment for free -- Scheduler.
assign_task only ever picks IDLE workers, and Scheduler._release_worker
(called from complete_task/fail_task) deliberately leaves a DRAINING
worker DRAINING instead of handing it back to IDLE. The master closes a
DRAINING worker's connection once nothing is left in flight for it --
immediately if it was already IDLE when SHUTDOWN arrived, or once its
current task finishes otherwise (see master/async_server.py's SHUTDOWN
branch and _close_connection_if_draining). handle_worker_connection's
`finally` block then reclassifies DRAINING -> STOPPED on that resulting
clean disconnect.

STOPPED is deliberately a different terminal state from FAILED:
WorkerManager.get_stale_workers skips STOPPED workers (their heartbeat
naturally stops too, but that's expected, not a failure), while
REPLACEABLE_WORKER_STATUSES still allows the same worker_id to re-register
afterward with a bumped generation -- a clean exit and a crash both leave
a worker_id free to come back, they just mean different things along the
way.

The critical distinction this file tests directly: a worker that dies
WHILE draining, before its current task finishes, must still be treated
as a genuine failure (task requeued, worker FAILED) -- draining is not
immunity from crashing, only a request not to be given anything new.
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


async def run_manual_worker(host: str, port: int, worker_id: str, message_queue: asyncio.Queue) -> AsyncConnection:
    """Registers and forwards EVERY message it receives afterward onto
    `message_queue` -- the test drives replies (and SHUTDOWN/PING)
    explicitly, for full control over timing. Only one coroutine may ever
    read a given connection (see WorkerLink's own docstring for why); a
    single background reader here, with the test consuming from the
    queue instead of calling receive_message directly, is what keeps that
    true even when the test also wants to send its own requests and
    observe their replies."""
    reader, writer = await asyncio.open_connection(host, port)
    conn = AsyncConnection(reader, writer)
    await send_request(conn, protocol.PING)
    await send_request(conn, protocol.REGISTER, {"worker_id": worker_id, "host": "127.0.0.1", "port": 6000})

    async def forward_all():
        while True:
            try:
                message = await receive_message(conn)
            except ConnectionError:
                return
            await message_queue.put(message)

    asyncio.create_task(forward_all())
    return conn


async def send_and_await_reply(conn: AsyncConnection, message_queue: asyncio.Queue, msg_type: str, payload: dict) -> dict:
    """Send a request on `conn` and wait for its reply via `message_queue`
    (populated by run_manual_worker's single background reader) instead of
    reading `conn` directly -- see run_manual_worker's docstring."""
    from rpc.async_rpc import new_request_id

    await send_message(conn, build_message(msg_type, new_request_id(), payload))
    return await asyncio.wait_for(message_queue.get(), timeout=5)


def test_shutdown_while_idle_transitions_to_draining_then_stopped():
    """A worker with no assigned task that sends SHUTDOWN must move to
    DRAINING and then, since nothing is in flight, STOPPED almost
    immediately -- the master closes its connection right away."""

    async def scenario():
        server, host, port = await start_master_server()
        message_queue: asyncio.Queue = asyncio.Queue()
        conn = await run_manual_worker(host, port, "worker-1", message_queue)
        await async_server.wait_for_workers(1)

        try:
            response = await send_and_await_reply(conn, message_queue, protocol.SHUTDOWN, {"worker_id": "worker-1"})
            assert response["type"] == protocol.SHUTDOWN_ACK

            deadline = time.monotonic() + 5
            while (
                async_server.worker_manager.get_worker("worker-1").status != WorkerStatus.STOPPED
                and time.monotonic() < deadline
            ):
                await asyncio.sleep(0.02)

            return async_server.worker_manager.get_worker("worker-1").status
        finally:
            await conn.close()
            server.close()
            await server.wait_closed()

    status = asyncio.run(scenario())
    assert status == WorkerStatus.STOPPED


def test_shutdown_while_busy_finishes_current_task_before_stopping():
    """A worker with a task in flight when SHUTDOWN arrives must complete
    that task normally (correct result, no retry) and only THEN become
    STOPPED -- not before, and not skipping the reply."""

    async def scenario():
        server, host, port = await start_master_server()
        message_queue: asyncio.Queue = asyncio.Queue()
        conn = await run_manual_worker(host, port, "worker-1", message_queue)
        await async_server.wait_for_workers(1)

        try:
            task = async_server.scheduler.submit_task("t1", "ADD", {"a": 3, "b": 4})
            waiter = asyncio.create_task(async_server.wait_for_tasks({task.task_id}))
            message = await asyncio.wait_for(message_queue.get(), timeout=5)

            # SHUTDOWN arrives WHILE the task is still in flight.
            shutdown_response = await send_and_await_reply(conn, message_queue, protocol.SHUTDOWN, {"worker_id": "worker-1"})
            assert shutdown_response["type"] == protocol.SHUTDOWN_ACK
            assert async_server.worker_manager.get_worker("worker-1").status == WorkerStatus.DRAINING

            # The connection must NOT be closed yet -- the task hasn't
            # finished. A PING must still get a normal reply.
            ping = await send_and_await_reply(conn, message_queue, protocol.PING, {})
            assert ping["type"] == protocol.PONG

            payload = message["payload"]
            result = execute_task(payload["task_type"], payload["task_payload"])
            reply = build_message(
                protocol.TASK_RESULT,
                message["request_id"],
                {"task_id": payload["task_id"], "attempt": payload.get("attempt", 1), **result},
            )
            await send_message(conn, reply)

            [response] = await asyncio.wait_for(waiter, timeout=5)

            deadline = time.monotonic() + 5
            while (
                async_server.worker_manager.get_worker("worker-1").status != WorkerStatus.STOPPED
                and time.monotonic() < deadline
            ):
                await asyncio.sleep(0.02)

            return response, task, async_server.worker_manager.get_worker("worker-1").status
        finally:
            await conn.close()
            server.close()
            await server.wait_closed()

    response, task, final_status = asyncio.run(scenario())
    assert response["payload"]["result"] == 7
    assert task.status == TaskStatus.COMPLETED
    assert task.attempt == 1  # no retry
    assert final_status == WorkerStatus.STOPPED


def test_draining_worker_gets_no_new_tasks_other_worker_picks_up_the_slack():
    """Two workers; worker-1 drains (idle, no task in flight -- stops
    immediately). New work submitted afterward must all go to worker-2;
    the dispatcher must never try to hand anything to a DRAINING/STOPPED
    worker-1."""

    async def scenario():
        server, host, port = await start_master_server()
        message_queue: asyncio.Queue = asyncio.Queue()
        conn1 = await run_manual_worker(host, port, "worker-1", message_queue)
        worker2_task = asyncio.create_task(
            async_worker.run_worker(host, port, worker_id="worker-2", heartbeat_interval=0.1)
        )
        await async_server.wait_for_workers(2)

        try:
            await send_and_await_reply(conn1, message_queue, protocol.SHUTDOWN, {"worker_id": "worker-1"})
            deadline = time.monotonic() + 5
            while (
                async_server.worker_manager.get_worker("worker-1").status != WorkerStatus.STOPPED
                and time.monotonic() < deadline
            ):
                await asyncio.sleep(0.02)
            assert async_server.worker_manager.get_worker("worker-1").status == WorkerStatus.STOPPED

            task = async_server.scheduler.submit_task("after-drain", "MULTIPLY", {"a": 6, "b": 7})
            [response] = await async_server.wait_for_tasks({task.task_id})
            return response, task
        finally:
            await conn1.close()
            await stop_worker(worker2_task)
            server.close()
            await server.wait_closed()

    response, task = asyncio.run(scenario())
    assert response["payload"]["result"] == 42
    assert task.assigned_worker_id == "worker-2"


def test_worker_dying_while_draining_before_finishing_is_a_genuine_failure():
    """DRAINING is a request not to be given new work -- it is NOT
    immunity from crashing. If the connection dies WHILE a task assigned
    before the drain request is still in flight, that's a real failure:
    the task must be requeued/retried and the worker marked FAILED, not
    STOPPED."""

    async def scenario():
        server, host, port = await start_master_server()
        message_queue: asyncio.Queue = asyncio.Queue()
        conn = await run_manual_worker(host, port, "worker-1", message_queue)
        rescue_task = asyncio.create_task(
            async_worker.run_worker(host, port, worker_id="worker-2", heartbeat_interval=0.1)
        )
        await async_server.wait_for_workers(2)

        try:
            task = async_server.scheduler.submit_task("t1", "ADD", {"a": 1, "b": 1})
            assigned = async_server.scheduler.assign_task("t1")
            assert assigned.assigned_worker_id == "worker-1"
            dispatch = asyncio.create_task(async_server.dispatch_assigned_task(task))
            await asyncio.wait_for(message_queue.get(), timeout=5)

            shutdown_response = await send_and_await_reply(conn, message_queue, protocol.SHUTDOWN, {"worker_id": "worker-1"})
            assert shutdown_response["type"] == protocol.SHUTDOWN_ACK
            assert async_server.worker_manager.get_worker("worker-1").status == WorkerStatus.DRAINING

            # Now it crashes -- WITHOUT ever replying to its assigned task.
            await conn.close()

            response = await dispatch
            deadline = time.monotonic() + 5
            while task.status != TaskStatus.COMPLETED and time.monotonic() < deadline:
                await async_server.drain_pending_tasks()
                await asyncio.sleep(0.02)

            return response, task, async_server.worker_manager.get_worker("worker-1").status
        finally:
            await stop_worker(rescue_task)
            server.close()
            await server.wait_closed()

    response, task, worker1_status = asyncio.run(scenario())
    assert response["payload"]["code"] == "WORKER_UNREACHABLE"
    assert worker1_status == WorkerStatus.FAILED
    assert task.status == TaskStatus.COMPLETED
    assert task.assigned_worker_id == "worker-2"
    assert task.attempt == 2


def test_reconnect_after_clean_stop_gets_new_generation_and_takes_work():
    """After a clean STOPPED shutdown, the same worker_id may re-register
    exactly like after a FAILED crash -- generation bumps, and it can take
    fresh work normally."""

    async def scenario():
        server, host, port = await start_master_server()
        message_queue: asyncio.Queue = asyncio.Queue()
        conn = await run_manual_worker(host, port, "worker-1", message_queue)
        await async_server.wait_for_workers(1)
        first_generation = async_server.worker_manager.get_worker("worker-1").generation

        try:
            await send_and_await_reply(conn, message_queue, protocol.SHUTDOWN, {"worker_id": "worker-1"})
            deadline = time.monotonic() + 5
            while (
                async_server.worker_manager.get_worker("worker-1").status != WorkerStatus.STOPPED
                and time.monotonic() < deadline
            ):
                await asyncio.sleep(0.02)
            assert async_server.worker_manager.get_worker("worker-1").status == WorkerStatus.STOPPED

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

            task = async_server.scheduler.submit_task("fresh", "MULTIPLY", {"a": 6, "b": 7})
            [response] = await async_server.wait_for_tasks({task.task_id})
            return response
        finally:
            await conn.close()
            await stop_worker(rescue_task)
            server.close()
            await server.wait_closed()

    response = asyncio.run(scenario())
    assert response["payload"]["result"] == 42


def test_stopped_worker_is_never_reclassified_as_failed_by_heartbeat_monitor(monkeypatch):
    """A STOPPED worker's heartbeat naturally stops (it disconnected on
    purpose) -- failure_monitor must never reclassify that as a crash."""
    monkeypatch.setattr(async_server, "FAILURE_CHECK_INTERVAL", 0.05)
    monkeypatch.setattr(async_server, "HEARTBEAT_TIMEOUT", 0.2)

    async def scenario():
        server, host, port = await start_master_server()
        message_queue: asyncio.Queue = asyncio.Queue()
        conn = await run_manual_worker(host, port, "worker-1", message_queue)
        await async_server.wait_for_workers(1)

        stop_event = asyncio.Event()
        monitor_task = asyncio.create_task(async_server.failure_monitor(stop_event))
        try:
            await send_and_await_reply(conn, message_queue, protocol.SHUTDOWN, {"worker_id": "worker-1"})
            deadline = time.monotonic() + 5
            while (
                async_server.worker_manager.get_worker("worker-1").status != WorkerStatus.STOPPED
                and time.monotonic() < deadline
            ):
                await asyncio.sleep(0.02)
            assert async_server.worker_manager.get_worker("worker-1").status == WorkerStatus.STOPPED

            # Well past HEARTBEAT_TIMEOUT, checked many times over --
            # status must stay STOPPED throughout.
            for _ in range(10):
                await asyncio.sleep(0.05)
                assert async_server.worker_manager.get_worker("worker-1").status == WorkerStatus.STOPPED
        finally:
            stop_event.set()
            await monitor_task
            await conn.close()
            server.close()
            await server.wait_closed()

    asyncio.run(scenario())


def test_real_worker_shutdown_event_drains_and_stops_cleanly():
    """End-to-end through the production worker_async.run_worker() API
    (not a manual test connection): setting `shutdown_event` while a real
    worker is executing a task must let that task finish and reply
    normally, then have the worker cleanly disconnect and end up STOPPED
    -- exercising watch_for_shutdown/send_shutdown for real, not just the
    master-side protocol handling."""

    async def scenario():
        server, host, port = await start_master_server()
        shutdown_event = asyncio.Event()
        worker_task = asyncio.create_task(
            async_worker.run_worker(host, port, worker_id="worker-1", shutdown_event=shutdown_event)
        )
        await async_server.wait_for_workers(1)

        try:
            task = async_server.scheduler.submit_task("t1", "ADD", {"a": 10, "b": 5})
            assigned = async_server.scheduler.assign_task("t1")
            assert assigned.assigned_worker_id == "worker-1"
            dispatch = asyncio.create_task(async_server.dispatch_assigned_task(task))

            # Request shutdown right away -- the in-flight task must still
            # complete normally despite this.
            shutdown_event.set()
            response = await asyncio.wait_for(dispatch, timeout=5)

            deadline = time.monotonic() + 5
            while (
                async_server.worker_manager.get_worker("worker-1").status != WorkerStatus.STOPPED
                and time.monotonic() < deadline
            ):
                await asyncio.sleep(0.02)

            # The worker's own run_worker() coroutine must exit cleanly
            # (the master closing the connection is what ends serve_tasks()).
            await asyncio.wait_for(worker_task, timeout=5)

            return response, task, async_server.worker_manager.get_worker("worker-1").status
        finally:
            server.close()
            await server.wait_closed()

    response, task, final_status = asyncio.run(scenario())
    assert response["payload"]["result"] == 15
    assert task.status == TaskStatus.COMPLETED
    assert final_status == WorkerStatus.STOPPED
