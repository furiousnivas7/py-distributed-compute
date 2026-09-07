"""Phase 7.5: consolidation + hardening tests for async failure detection.

Everything here reuses master.async_server / worker.async_worker as they
already exist -- no new networking or dispatch logic. The goal is coverage
for scenarios the 7.3/7.4 test suites didn't specifically exercise:
heartbeat staleness with the connection still open (distinct from a dead
connection), a false-positive check, the current no-reconnection
limitation, multi-task requeue, completed-task protection, retry
exhaustion, and a second worker continuing after the first fails.

Each test drives its scenario with asyncio.run() (no pytest-asyncio
dependency, matching the rest of the async suite).
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


async def wait_for_worker_id(worker_id: str, poll_interval: float = 0.01, timeout: float = 5.0) -> None:
    """Wait for a SPECIFIC worker_id to be connected, rather than for a
    total count. async_server.wait_for_workers(N) counts currently-open
    connections -- when workers register and crash one at a time in
    sequence, the total can drop back down after each crash and never
    reach N again even though the worker we're waiting for is present."""
    deadline = asyncio.get_running_loop().time() + timeout
    while worker_id not in async_server.connections:
        if asyncio.get_running_loop().time() > deadline:
            raise TimeoutError(f"worker {worker_id!r} never connected")
        await asyncio.sleep(poll_interval)


async def run_multi_task_crashing_worker(host: str, port: int, worker_id: str, task_count: int, ready: asyncio.Event) -> None:
    """Registers, then reads `task_count` TASK messages without replying to
    any of them, signals `ready`, and disconnects -- simulating a worker
    that crashed with several tasks in flight at once."""
    reader, writer = await asyncio.open_connection(host, port)
    conn = AsyncConnection(reader, writer)
    await send_request(conn, protocol.PING)
    await send_request(conn, protocol.REGISTER, {"worker_id": worker_id, "host": "127.0.0.1", "port": 6000})

    seen = 0
    while seen < task_count:
        message = await receive_message(conn)
        if message["type"] == protocol.TASK:
            seen += 1

    ready.set()
    await conn.close()


async def run_worker_crashing_on_task(host: str, port: int, worker_id: str, crash_on_task_id: str) -> None:
    """Behaves like a normal worker for every task except `crash_on_task_id`,
    which it receives and then disconnects on without ever replying."""
    reader, writer = await asyncio.open_connection(host, port)
    conn = AsyncConnection(reader, writer)
    await send_request(conn, protocol.PING)
    await send_request(conn, protocol.REGISTER, {"worker_id": worker_id, "host": "127.0.0.1", "port": 6000})

    try:
        while True:
            try:
                message = await receive_message(conn)
            except ConnectionError:
                return

            if message["type"] != protocol.TASK:
                continue

            payload = message["payload"]
            if payload["task_id"] == crash_on_task_id:
                return  # closes via finally, without replying

            result = execute_task(payload["task_type"], payload["task_payload"])
            response = build_message(
                protocol.TASK_RESULT,
                message["request_id"],
                {"task_id": payload["task_id"], "attempt": payload.get("attempt", 1), **result},
            )
            await send_message(conn, response)
    finally:
        await conn.close()


def test_heartbeat_timeout_detected_while_connection_stays_open(monkeypatch):
    """Staleness is a heartbeat-clock signal, independent of the connection.
    A worker whose heartbeat loop has effectively stopped (huge interval)
    but whose TCP connection is still alive and being read must still be
    marked FAILED -- the connection itself doesn't need to die for this."""
    monkeypatch.setattr(async_server, "FAILURE_CHECK_INTERVAL", 0.05)
    monkeypatch.setattr(async_server, "HEARTBEAT_TIMEOUT", 0.3)

    async def scenario():
        server, host, port = await start_master_server()
        worker_task = asyncio.create_task(
            async_worker.run_worker(host, port, worker_id="worker-1", heartbeat_interval=9999)
        )
        stop_event = asyncio.Event()
        monitor_task = asyncio.create_task(async_server.failure_monitor(stop_event))
        try:
            await async_server.wait_for_workers(1)

            deadline = time.monotonic() + 3
            while (
                async_server.worker_manager.get_worker("worker-1").status != WorkerStatus.FAILED
                and time.monotonic() < deadline
            ):
                await asyncio.sleep(0.02)

            status = async_server.worker_manager.get_worker("worker-1").status
            still_connected = "worker-1" in async_server.connections
            return status, still_connected
        finally:
            stop_event.set()
            await monitor_task
            await stop_worker(worker_task)
            server.close()
            await server.wait_closed()

    status, still_connected = asyncio.run(scenario())
    assert status == WorkerStatus.FAILED
    # The connection wasn't touched by failure detection -- only the
    # worker's recorded status and its (nonexistent, here) tasks were.
    assert still_connected is True


def test_no_false_failure_when_heartbeats_continue(monkeypatch):
    """A worker heartbeating faster than the timeout must never be marked
    FAILED, even if the monitor checks well past HEARTBEAT_TIMEOUT.

    Margin note: this was originally HEARTBEAT_TIMEOUT=0.3 / heartbeat_
    interval=0.05 (a 6x margin) and was found to be genuinely flaky under
    real system load -- a repro loop of 150 back-to-back runs (see Phase
    9.1 follow-up) caught it failing ~1-2% of the time, because a single
    real heartbeat delayed past 250ms by ordinary OS/event-loop scheduling
    jitter is enough to trip a false-positive failure. Every other test in
    this file/test_async_regression.py uses a 10x margin (1.0s timeout /
    0.1s interval) and none of those have ever been observed to flake
    across 250+ cumulative runs, so this was widened to match rather than
    kept tight for speed."""
    monkeypatch.setattr(async_server, "FAILURE_CHECK_INTERVAL", 0.05)
    monkeypatch.setattr(async_server, "HEARTBEAT_TIMEOUT", 1.0)

    async def scenario():
        server, host, port = await start_master_server()
        worker_task = asyncio.create_task(
            async_worker.run_worker(host, port, worker_id="worker-1", heartbeat_interval=0.1)
        )
        stop_event = asyncio.Event()
        monitor_task = asyncio.create_task(async_server.failure_monitor(stop_event))
        try:
            await async_server.wait_for_workers(1)

            # Well past HEARTBEAT_TIMEOUT, checked many times over.
            for _ in range(15):
                await asyncio.sleep(0.15)
                assert async_server.worker_manager.get_worker("worker-1").status == WorkerStatus.IDLE
        finally:
            stop_event.set()
            await monitor_task
            await stop_worker(worker_task)
            server.close()
            await server.wait_closed()

    asyncio.run(scenario())


def test_reconnection_with_same_worker_id_while_active_is_still_rejected():
    """A worker_id that's still REGISTERED/IDLE/BUSY (a live connection is
    actually representing it) rejects a second registration attempt as a
    genuine conflict, not a reconnect -- unchanged by Phase 9.2.1, which
    only allows re-registration once the existing entry is FAILED (see
    test_reconnection_with_same_worker_id_after_failure_succeeds_with_new_generation)."""

    async def scenario():
        server, host, port = await start_master_server()
        worker_task = asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-1"))
        try:
            await async_server.wait_for_workers(1)
            assert async_server.worker_manager.get_worker("worker-1").status == WorkerStatus.IDLE

            reader, writer = await asyncio.open_connection(host, port)
            conn = AsyncConnection(reader, writer)
            try:
                await send_request(conn, protocol.PING)
                return await send_request(
                    conn, protocol.REGISTER, {"worker_id": "worker-1", "host": "127.0.0.1", "port": 6002}
                )
            finally:
                await conn.close()
        finally:
            await stop_worker(worker_task)
            server.close()
            await server.wait_closed()

    response = asyncio.run(scenario())
    assert response["type"] == protocol.ERROR
    assert response["payload"]["code"] == "DUPLICATE_WORKER"


def test_reconnection_with_same_worker_id_after_failure_succeeds_with_new_generation():
    """Phase 9.2.1: once a worker_id is marked FAILED, a new connection
    claiming that same worker_id may re-register -- this is worker
    recovery, not a duplicate-registration conflict. The new registration
    gets a bumped generation, and the OLD connection (still technically
    alive here -- forcing FAILED via update_status doesn't itself kill the
    real worker's TCP connection, exactly the scenario that matters) is
    proactively invalidated by the master: closed outright, so it can
    never again mutate state on behalf of what's now a superseded
    generation (a stale heartbeat refreshing the NEW generation's
    last_heartbeat, for instance)."""

    async def scenario():
        server, host, port = await start_master_server()
        worker_task = asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-1"))
        try:
            await async_server.wait_for_workers(1)
            old_link = async_server.connections["worker-1"]
            assert old_link.generation == 1

            async_server.worker_manager.update_status("worker-1", WorkerStatus.FAILED)

            reader, writer = await asyncio.open_connection(host, port)
            conn = AsyncConnection(reader, writer)
            try:
                await send_request(conn, protocol.PING)
                response = await send_request(
                    conn, protocol.REGISTER, {"worker_id": "worker-1", "host": "127.0.0.1", "port": 6002}
                )

                worker = async_server.worker_manager.get_worker("worker-1")
                assert worker.status == WorkerStatus.IDLE
                assert worker.generation == 2
                assert worker.port == 6002

                new_link = async_server.connections["worker-1"]
                assert new_link is not old_link
                assert new_link.generation == 2

                # The old connection must have been actually closed by the
                # master, not just superseded in bookkeeping -- proven by
                # its own read now failing.
                with pytest.raises(ConnectionError):
                    await receive_message(old_link.conn)

                return response
            finally:
                await conn.close()
        finally:
            await stop_worker(worker_task)
            server.close()
            await server.wait_closed()

    response = asyncio.run(scenario())
    assert response["type"] == protocol.REGISTER_ACK


def test_multiple_tasks_requeued_when_worker_fails(monkeypatch):
    """worker-1 has two tasks in flight when it crashes; both get requeued
    and both get completed by worker-2. Concurrent dispatch to one worker
    connection is safe here because AsyncConnection.send_bytes buffers a
    whole framed message in a single synchronous write() -- unlike the
    threaded suite, no special fire-and-forget helper is needed to avoid an
    unsafe interleaved write."""
    monkeypatch.setattr(async_server, "FAILURE_CHECK_INTERVAL", 0.05)
    monkeypatch.setattr(async_server, "HEARTBEAT_TIMEOUT", 1.0)

    async def scenario():
        server, host, port = await start_master_server()

        ready = asyncio.Event()
        worker1_task = asyncio.create_task(run_multi_task_crashing_worker(host, port, "worker-1", 2, ready))
        await async_server.wait_for_workers(1)

        worker2_task = asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-2"))
        await async_server.wait_for_workers(2)

        stop_event = asyncio.Event()
        monitor_task = asyncio.create_task(async_server.failure_monitor(stop_event))

        try:
            task1 = async_server.scheduler.submit_task("task-1", "ADD", {"a": 1, "b": 1})
            task2 = async_server.scheduler.submit_task("task-2", "ADD", {"a": 2, "b": 2})

            # Put both on worker-1: assign_task only ever picks an IDLE
            # worker, so toggling it back to IDLE between assignments is
            # what lets one worker end up holding two (matching the same
            # technique the scheduler-level tests use).
            async_server.scheduler.assign_task("task-1")
            async_server.worker_manager.update_status("worker-1", WorkerStatus.IDLE)
            async_server.scheduler.assign_task("task-2")

            assert task1.assigned_worker_id == "worker-1"
            assert task2.assigned_worker_id == "worker-1"

            dispatch1 = asyncio.create_task(async_server.dispatch_assigned_task(task1))
            dispatch2 = asyncio.create_task(async_server.dispatch_assigned_task(task2))

            await asyncio.wait_for(ready.wait(), timeout=5)

            deadline = time.monotonic() + 5
            while (
                async_server.worker_manager.get_worker("worker-1").status != WorkerStatus.FAILED
                and time.monotonic() < deadline
            ):
                await asyncio.sleep(0.02)

            assert async_server.worker_manager.get_worker("worker-1").status == WorkerStatus.FAILED
            assert task1.status == TaskStatus.PENDING
            assert task2.status == TaskStatus.PENDING

            await dispatch1
            await dispatch2

            responses = await async_server.drain_pending_tasks()
            assert all(r["payload"]["status"] == "success" for r in responses)

            assert task1.status == TaskStatus.COMPLETED
            assert task2.status == TaskStatus.COMPLETED
            assert task1.assigned_worker_id == "worker-2"
            assert task2.assigned_worker_id == "worker-2"
            assert task1.attempt == 2
            assert task2.attempt == 2
        finally:
            stop_event.set()
            await monitor_task
            await stop_worker(worker1_task)
            await stop_worker(worker2_task)
            server.close()
            await server.wait_closed()

    asyncio.run(scenario())


def test_completed_task_is_not_requeued_when_same_worker_later_fails(monkeypatch):
    """worker-1 completes task-1, then is given task-2 and crashes while
    running it. task-1 must stay COMPLETED; only task-2 gets requeued."""
    monkeypatch.setattr(async_server, "FAILURE_CHECK_INTERVAL", 0.05)
    monkeypatch.setattr(async_server, "HEARTBEAT_TIMEOUT", 1.0)

    async def scenario():
        server, host, port = await start_master_server()
        worker1_task = asyncio.create_task(
            run_worker_crashing_on_task(host, port, "worker-1", crash_on_task_id="task-2")
        )
        await async_server.wait_for_workers(1)
        worker2_task = asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-2"))
        await async_server.wait_for_workers(2)

        stop_event = asyncio.Event()
        monitor_task = asyncio.create_task(async_server.failure_monitor(stop_event))

        try:
            task1 = async_server.scheduler.submit_task("task-1", "ADD", {"a": 1, "b": 1})
            async_server.scheduler.assign_task("task-1")
            await async_server.dispatch_assigned_task(task1)
            assert task1.status == TaskStatus.COMPLETED
            assert task1.assigned_worker_id == "worker-1"

            # worker-1 is IDLE again; give it task-2, which it will crash on.
            task2 = async_server.scheduler.submit_task("task-2", "ADD", {"a": 2, "b": 2})
            assigned2 = async_server.scheduler.assign_task("task-2")
            assert assigned2.assigned_worker_id == "worker-1"

            dispatch2 = asyncio.create_task(async_server.dispatch_assigned_task(task2))

            deadline = time.monotonic() + 5
            while (
                async_server.worker_manager.get_worker("worker-1").status != WorkerStatus.FAILED
                and time.monotonic() < deadline
            ):
                async_server.worker_manager.get_worker("worker-1").last_heartbeat = time.time() - 10
                await asyncio.sleep(0.02)

            assert async_server.worker_manager.get_worker("worker-1").status == WorkerStatus.FAILED
            assert task1.status == TaskStatus.COMPLETED  # untouched
            assert task2.status == TaskStatus.PENDING

            await dispatch2

            responses = await async_server.drain_pending_tasks()
            assert responses[0]["payload"]["status"] == "success"
            assert task2.status == TaskStatus.COMPLETED
            assert task2.assigned_worker_id == "worker-2"
        finally:
            stop_event.set()
            await monitor_task
            await stop_worker(worker1_task)
            await stop_worker(worker2_task)
            server.close()
            await server.wait_closed()

    asyncio.run(scenario())


def test_retry_exhaustion_after_max_attempts(monkeypatch):
    """Three workers, each crashing in turn on the same task: attempt 1
    fails -> PENDING, attempt 2 fails -> PENDING, attempt 3 fails -> FAILED,
    with no attempt 4 -- proving MAX_TASK_ATTEMPTS is enforced through the
    real async dispatch + failure_monitor path, not just at the Scheduler
    unit level (already covered in tests/test_scheduler.py).

    Root cause of an intermittent failure found via a 150-run repro loop
    (Phase 9.1 follow-up): failure_monitor is a LIVE background task here
    (deliberately -- this test wants to prove ITS real requeue+redispatch
    path, not just manual orchestration), and on every FAILURE_CHECK_INTERVAL
    tick it re-scans for stale workers and, upon finding one, calls
    drain_pending_tasks() itself -- synchronously, with no `await` between
    detecting staleness and grabbing+dispatching the newly-PENDING task to
    whichever worker is currently IDLE. Once worker-(i-1) is forced stale,
    THE MOMENT worker-i registers and goes IDLE, failure_monitor's own next
    tick (as fast as 0.05s later) can grab and dispatch the requeued task
    to it before this test's own manual assign_next_pending_task() call
    gets a turn -- an inherent race between two independent callers of the
    same scheduler (the exact hazard Phase 8.8 addressed for concurrent
    MapReduce jobs, just showing up here between a test's manual control
    and failure_monitor's autonomous fallback). Under real system load the
    race could go either way, and a manual assign_next_pending_task() call
    that loses it returns None, crashing the test's own assertion.

    Fix: for i=1 no staleness exists yet, so nothing else could be racing
    -- manual assign+dispatch stays as the (only) way that attempt starts.
    For i=2 and i=3, this no longer calls assign_next_pending_task() /
    dispatch_assigned_task() at all -- that's the whole point being tested,
    so the test just registers the next worker and OBSERVES that
    failure_monitor's own pipeline reassigns and redispatches to it
    correctly (via ready.wait() firing and task.assigned_worker_id/attempt
    matching), rather than competing with it for who gets to perform the
    (re)assignment."""
    from master.scheduler import MAX_TASK_ATTEMPTS

    assert MAX_TASK_ATTEMPTS == 3, "test assumes the current default of 3"

    monkeypatch.setattr(async_server, "FAILURE_CHECK_INTERVAL", 0.05)
    monkeypatch.setattr(async_server, "HEARTBEAT_TIMEOUT", 1.0)

    async def force_stale_and_wait_failed(worker_id: str) -> None:
        async_server.worker_manager.get_worker(worker_id).last_heartbeat = time.time() - 10
        deadline = time.monotonic() + 5
        while (
            async_server.worker_manager.get_worker(worker_id).status != WorkerStatus.FAILED
            and time.monotonic() < deadline
        ):
            await asyncio.sleep(0.02)
        assert async_server.worker_manager.get_worker(worker_id).status == WorkerStatus.FAILED

    async def scenario():
        server, host, port = await start_master_server()

        worker_tasks = []
        stop_event = asyncio.Event()
        monitor_task = asyncio.create_task(async_server.failure_monitor(stop_event))

        try:
            task = async_server.scheduler.submit_task("task-1", "ADD", {"a": 1, "b": 1})

            # i=1: no staleness exists yet, so nothing else can grab this
            # task -- a manual assign+dispatch is required and safe.
            ready1 = asyncio.Event()
            worker_tasks.append(asyncio.create_task(run_multi_task_crashing_worker(host, port, "worker-1", 1, ready1)))
            await wait_for_worker_id("worker-1")

            assigned = async_server.scheduler.assign_next_pending_task()
            assert assigned.assigned_worker_id == "worker-1"
            assert assigned.attempt == 1

            dispatch1 = asyncio.create_task(async_server.dispatch_assigned_task(task))
            await asyncio.wait_for(ready1.wait(), timeout=5)
            await force_stale_and_wait_failed("worker-1")
            await dispatch1

            # i=2, i=3: worker-(i-1) is now stale and its task requeued --
            # failure_monitor's own background tick will reassign and
            # redispatch it to worker-i the moment it registers. Don't
            # compete with that; just observe it happen.
            for i in range(2, MAX_TASK_ATTEMPTS + 1):
                worker_id = f"worker-{i}"
                ready = asyncio.Event()
                worker_tasks.append(
                    asyncio.create_task(run_multi_task_crashing_worker(host, port, worker_id, 1, ready))
                )
                await wait_for_worker_id(worker_id)

                await asyncio.wait_for(ready.wait(), timeout=5)
                assert task.assigned_worker_id == worker_id
                assert task.attempt == i

                await force_stale_and_wait_failed(worker_id)

            deadline = time.monotonic() + 5
            while task.status != TaskStatus.FAILED and time.monotonic() < deadline:
                await asyncio.sleep(0.02)

            assert task.status == TaskStatus.FAILED
            assert task.attempt == MAX_TASK_ATTEMPTS
            assert async_server.scheduler.assign_next_pending_task() is None
        finally:
            stop_event.set()
            await monitor_task
            for t in worker_tasks:
                await stop_worker(t)
            server.close()
            await server.wait_closed()

    asyncio.run(scenario())


def test_second_worker_keeps_working_after_first_worker_fails(monkeypatch):
    """2 workers, each already given a task; worker-1 crashes on its task
    while worker-2's own task completes normally and independently, then
    worker-2 also picks up worker-1's requeued task."""
    monkeypatch.setattr(async_server, "FAILURE_CHECK_INTERVAL", 0.05)
    monkeypatch.setattr(async_server, "HEARTBEAT_TIMEOUT", 1.0)

    async def scenario():
        server, host, port = await start_master_server()

        ready = asyncio.Event()
        worker1_task = asyncio.create_task(run_multi_task_crashing_worker(host, port, "worker-1", 1, ready))
        await async_server.wait_for_workers(1)
        worker2_task = asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-2"))
        await async_server.wait_for_workers(2)

        stop_event = asyncio.Event()
        monitor_task = asyncio.create_task(async_server.failure_monitor(stop_event))

        try:
            task1 = async_server.scheduler.submit_task("task-1", "ADD", {"a": 1, "b": 1})
            task2 = async_server.scheduler.submit_task("task-2", "ADD", {"a": 2, "b": 2})

            assigned1 = async_server.scheduler.assign_task("task-1")
            assigned2 = async_server.scheduler.assign_task("task-2")
            assert assigned1.assigned_worker_id == "worker-1"
            assert assigned2.assigned_worker_id == "worker-2"

            dispatch1 = asyncio.create_task(async_server.dispatch_assigned_task(task1))
            dispatch2_response = await async_server.dispatch_assigned_task(task2)

            # worker-2's own task succeeded independently of worker-1's fate.
            assert dispatch2_response["payload"]["status"] == "success"
            assert task2.status == TaskStatus.COMPLETED

            await asyncio.wait_for(ready.wait(), timeout=5)
            async_server.worker_manager.get_worker("worker-1").last_heartbeat = time.time() - 10

            deadline = time.monotonic() + 5
            while (
                async_server.worker_manager.get_worker("worker-1").status != WorkerStatus.FAILED
                and time.monotonic() < deadline
            ):
                await asyncio.sleep(0.02)

            assert task1.status == TaskStatus.PENDING
            await dispatch1

            responses = await async_server.drain_pending_tasks()
            assert responses[0]["payload"]["status"] == "success"
            assert task1.status == TaskStatus.COMPLETED
            assert task1.assigned_worker_id == "worker-2"
            assert task1.attempt == 2
        finally:
            stop_event.set()
            await monitor_task
            await stop_worker(worker1_task)
            await stop_worker(worker2_task)
            server.close()
            await server.wait_closed()

    asyncio.run(scenario())
