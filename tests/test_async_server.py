"""Integration tests for master.async_server.

Each test drives its scenario with asyncio.run() (no pytest-asyncio
dependency, matching test_async_rpc.py / test_async_connection.py). A small
fake async worker stands in for the real async worker, which is Phase 7.4 --
it speaks the same protocol a real one will, including sending HEARTBEAT on
the SAME connection it also receives TASK on, which is exactly the
simplification this phase's module docstring calls out.
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
from worker.executor import execute_task


@pytest.fixture(autouse=True)
def reset_async_master_state():
    """worker_manager and scheduler are module-level singletons in
    master.async_server (worker_manager is even shared with the threaded
    implementation, via rpc_handler) -- clear them in place per test rather
    than replacing the objects, for the same reason test_task_flow.py does:
    replacing rpc_handler.worker_manager would desync it from
    async_server.scheduler.worker_manager, which still points at the old one.
    """
    rpc_handler.worker_manager.clear()
    async_server.scheduler.clear()
    async_server.connections.clear()
    yield


async def start_master_server():
    server = await asyncio.start_server(async_server.handle_worker_connection, "127.0.0.1", 0)
    host, port = server.sockets[0].getsockname()[:2]
    return server, host, port


async def run_async_fake_worker(
    host: str,
    port: int,
    worker_id: str,
    *,
    worker_host: str = "127.0.0.1",
    worker_port: int = 6000,
    ready_event: asyncio.Event | None = None,
    release_event: asyncio.Event | None = None,
    crash_after_task: bool = False,
) -> None:
    """Registers like a real worker would, then serves TASK requests until
    the connection closes. Ignores any other unsolicited message (e.g. a
    HEARTBEAT_ACK for a heartbeat it sent) -- it isn't correlating those,
    just like a fire-and-forget heartbeat sender wouldn't need to.

    `ready_event`/`release_event` let a test pause the worker right after it
    receives a TASK (mirroring the threaded suite's run_gated_worker), and
    `crash_after_task` makes it disconnect without ever replying, simulating
    a crash mid-task.
    """
    reader, writer = await asyncio.open_connection(host, port)
    conn = AsyncConnection(reader, writer)

    await send_request(conn, protocol.PING)
    await send_request(conn, protocol.REGISTER, {"worker_id": worker_id, "host": worker_host, "port": worker_port})

    try:
        while True:
            try:
                message = await receive_message(conn)
            except ConnectionError:
                return

            if message["type"] != protocol.TASK:
                continue

            payload = message["payload"]

            if ready_event is not None:
                ready_event.set()
            if release_event is not None:
                await release_event.wait()

            if crash_after_task:
                return

            result = execute_task(payload["task_type"], payload["task_payload"])
            response = build_message(
                protocol.TASK_RESULT,
                message["request_id"],
                {"task_id": payload["task_id"], "attempt": payload.get("attempt", 1), **result},
            )
            await send_message(conn, response)
    finally:
        await conn.close()


def test_task_dispatched_and_executed():
    async def scenario():
        server, host, port = await start_master_server()
        worker_task = asyncio.create_task(run_async_fake_worker(host, port, "worker-1"))
        try:
            await async_server.wait_for_workers(1)

            async_server.scheduler.submit_task("task-1", "ADD", {"a": 10, "b": 20})
            return await async_server.drain_pending_tasks()
        finally:
            server.close()
            await server.wait_closed()
            worker_task.cancel()

    responses = asyncio.run(scenario())
    assert len(responses) == 1
    assert responses[0]["payload"] == {"task_id": "task-1", "attempt": 1, "status": "success", "result": 30}


def test_multiply_task_dispatched_and_executed():
    async def scenario():
        server, host, port = await start_master_server()
        worker_task = asyncio.create_task(run_async_fake_worker(host, port, "worker-1"))
        try:
            await async_server.wait_for_workers(1)
            async_server.scheduler.submit_task("task-1", "MULTIPLY", {"a": 6, "b": 7})
            return await async_server.drain_pending_tasks()
        finally:
            server.close()
            await server.wait_closed()
            worker_task.cancel()

    responses = asyncio.run(scenario())
    assert responses[0]["payload"]["status"] == "success"
    assert responses[0]["payload"]["result"] == 42


def test_task_execution_failure_is_reported():
    async def scenario():
        server, host, port = await start_master_server()
        worker_task = asyncio.create_task(run_async_fake_worker(host, port, "worker-1"))
        try:
            await async_server.wait_for_workers(1)
            async_server.scheduler.submit_task("task-1", "ADD", {"a": "x", "b": 1})
            return await async_server.drain_pending_tasks()
        finally:
            server.close()
            await server.wait_closed()
            worker_task.cancel()

    responses = asyncio.run(scenario())
    assert responses[0]["payload"]["status"] == "error"
    assert "message" in responses[0]["payload"]


def test_scheduler_queues_tasks_across_two_workers():
    """2 workers + 4 tasks: everyone gets used, none left starved."""

    async def scenario():
        server, host, port = await start_master_server()
        worker_tasks = [
            asyncio.create_task(run_async_fake_worker(host, port, "worker-1")),
            asyncio.create_task(run_async_fake_worker(host, port, "worker-2")),
        ]
        try:
            await async_server.wait_for_workers(2)

            for task_id, task_type, payload in [
                ("task-1", "ADD", {"a": 1, "b": 1}),
                ("task-2", "ADD", {"a": 2, "b": 2}),
                ("task-3", "ADD", {"a": 3, "b": 3}),
                ("task-4", "ADD", {"a": 4, "b": 4}),
            ]:
                async_server.scheduler.submit_task(task_id, task_type, payload)

            return await async_server.drain_pending_tasks()
        finally:
            server.close()
            await server.wait_closed()
            for t in worker_tasks:
                t.cancel()

    responses = asyncio.run(scenario())
    assert len(responses) == 4
    assert all(r["payload"]["status"] == "success" for r in responses)

    statuses = {t.task_id: t.status for t in async_server.scheduler.get_all_tasks()}
    assert all(status == TaskStatus.COMPLETED for status in statuses.values())
    workers_used = {t.assigned_worker_id for t in async_server.scheduler.get_all_tasks()}
    assert workers_used == {"worker-1", "worker-2"}


def test_dispatch_runs_two_workers_concurrently():
    """Deterministic proof of concurrency: both tasks reach RUNNING and sit
    there simultaneously, gated on asyncio.Events rather than sleeps/timing."""

    async def scenario():
        server, host, port = await start_master_server()

        ready = {"worker-1": asyncio.Event(), "worker-2": asyncio.Event()}
        release = {"worker-1": asyncio.Event(), "worker-2": asyncio.Event()}

        worker_tasks = [
            asyncio.create_task(
                run_async_fake_worker(host, port, wid, ready_event=ready[wid], release_event=release[wid])
            )
            for wid in ("worker-1", "worker-2")
        ]
        try:
            await async_server.wait_for_workers(2)

            async_server.scheduler.submit_task("task-1", "ADD", {"a": 1, "b": 1})
            async_server.scheduler.submit_task("task-2", "ADD", {"a": 2, "b": 2})

            drain_task = asyncio.create_task(async_server.drain_pending_tasks())

            await asyncio.wait_for(ready["worker-1"].wait(), timeout=5)
            await asyncio.wait_for(ready["worker-2"].wait(), timeout=5)

            statuses = {t.task_id: t.status for t in async_server.scheduler.get_all_tasks()}
            assert statuses == {"task-1": TaskStatus.RUNNING, "task-2": TaskStatus.RUNNING}

            release["worker-1"].set()
            release["worker-2"].set()
            return await drain_task
        finally:
            server.close()
            await server.wait_closed()
            for t in worker_tasks:
                t.cancel()

    responses = asyncio.run(scenario())
    assert all(r["payload"]["status"] == "success" for r in responses)


def test_heartbeat_and_task_share_one_connection():
    """The architectural simplification this phase enables: HEARTBEAT and
    TASK/TASK_RESULT multiplex over the SAME persistent connection, with no
    separate heartbeat port or listener needed."""

    async def scenario():
        server, host, port = await start_master_server()
        reader, writer = await asyncio.open_connection(host, port)
        conn = AsyncConnection(reader, writer)

        await send_request(conn, protocol.PING)
        await send_request(conn, protocol.REGISTER, {"worker_id": "worker-1", "host": "127.0.0.1", "port": 6001})

        before = async_server.worker_manager.get_worker("worker-1").last_heartbeat
        heartbeat_response = await send_request(conn, protocol.HEARTBEAT, {"worker_id": "worker-1"})
        after = async_server.worker_manager.get_worker("worker-1").last_heartbeat

        # Now dispatch a TASK over the SAME connection and reply to it, all
        # driven by this one coroutine (matching how a real async worker
        # would multiplex both roles on its single connection).
        async_server.scheduler.submit_task("task-1", "ADD", {"a": 10, "b": 20})
        task = async_server.scheduler.assign_next_pending_task()
        assert task.assigned_worker_id == "worker-1"

        dispatch_coro = asyncio.create_task(async_server.dispatch_assigned_task(task))

        task_request = await receive_message(conn)
        assert task_request["type"] == protocol.TASK
        result = execute_task(task_request["payload"]["task_type"], task_request["payload"]["task_payload"])
        task_response = build_message(
            protocol.TASK_RESULT,
            task_request["request_id"],
            {"task_id": "task-1", "attempt": task_request["payload"]["attempt"], **result},
        )
        await send_message(conn, task_response)

        dispatch_response = await dispatch_coro

        server.close()
        await server.wait_closed()
        await conn.close()

        return heartbeat_response, before, after, dispatch_response

    heartbeat_response, before, after, dispatch_response = asyncio.run(scenario())
    assert heartbeat_response["type"] == protocol.HEARTBEAT_ACK
    assert after >= before
    assert dispatch_response["payload"]["status"] == "success"
    assert dispatch_response["payload"]["result"] == 30


def test_heartbeat_for_unknown_worker_is_rejected():
    async def scenario():
        server, host, port = await start_master_server()
        reader, writer = await asyncio.open_connection(host, port)
        conn = AsyncConnection(reader, writer)
        try:
            return await send_request(conn, protocol.HEARTBEAT, {"worker_id": "ghost"})
        finally:
            await conn.close()
            server.close()
            await server.wait_closed()

    response = asyncio.run(scenario())
    assert response["type"] == protocol.ERROR
    assert response["payload"]["code"] == "UNKNOWN_WORKER"


def test_failed_worker_task_is_requeued_and_completed_by_another_worker(monkeypatch):
    """The 6.6.5-equivalent real-TCP fault-tolerance scenario, over the
    async transport: worker-1 gets the task and crashes before replying;
    the failure monitor detects it and requeues; worker-2 completes it."""
    monkeypatch.setattr(async_server, "FAILURE_CHECK_INTERVAL", 0.05)
    monkeypatch.setattr(async_server, "HEARTBEAT_TIMEOUT", 1.0)

    async def scenario():
        server, host, port = await start_master_server()

        received_event = asyncio.Event()
        worker1_task = asyncio.create_task(
            run_async_fake_worker(host, port, "worker-1", ready_event=received_event, crash_after_task=True)
        )
        await async_server.wait_for_workers(1)

        worker2_task = asyncio.create_task(run_async_fake_worker(host, port, "worker-2"))
        await async_server.wait_for_workers(2)

        stop_event = asyncio.Event()
        monitor_task = asyncio.create_task(async_server.failure_monitor(stop_event))

        try:
            task = async_server.scheduler.submit_task("task-1", "ADD", {"a": 10, "b": 20})
            assigned = async_server.scheduler.assign_task("task-1")
            assert assigned.assigned_worker_id == "worker-1"
            assert assigned.attempt == 1

            dispatch_task_coro = asyncio.create_task(async_server.dispatch_assigned_task(task))

            await asyncio.wait_for(received_event.wait(), timeout=5)

            # Force staleness directly instead of waiting out a real heartbeat cycle.
            async_server.worker_manager.get_worker("worker-1").last_heartbeat = time.time() - 10

            deadline = time.monotonic() + 5
            while (
                async_server.worker_manager.get_worker("worker-1").status != WorkerStatus.FAILED
                and time.monotonic() < deadline
            ):
                await asyncio.sleep(0.02)

            assert async_server.worker_manager.get_worker("worker-1").status == WorkerStatus.FAILED
            assert task.status == TaskStatus.PENDING
            assert task.assigned_worker_id is None
            assert task.attempt == 1

            reassigned = async_server.scheduler.assign_next_pending_task()
            assert reassigned is task
            assert task.assigned_worker_id == "worker-2"
            assert task.attempt == 2

            response = await async_server.dispatch_assigned_task(task)

            assert response["type"] == protocol.TASK_RESULT
            assert response["payload"]["attempt"] == 2
            assert response["payload"]["status"] == "success"
            assert response["payload"]["result"] == 30

            assert task.status == TaskStatus.COMPLETED
            assert task.assigned_worker_id == "worker-2"

            # The stale worker-1 dispatch, still unwinding from the dead
            # connection, must not clobber attempt 2's COMPLETED outcome.
            stale_dispatch_response = await dispatch_task_coro
            assert stale_dispatch_response["payload"]["code"] == "WORKER_UNREACHABLE"
            assert task.status == TaskStatus.COMPLETED
            assert task.assigned_worker_id == "worker-2"

            return True
        finally:
            stop_event.set()
            await monitor_task
            server.close()
            await server.wait_closed()
            worker1_task.cancel()
            worker2_task.cancel()

    assert asyncio.run(scenario()) is True
