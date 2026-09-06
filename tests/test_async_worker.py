"""Integration tests for worker.async_worker, driven against the real
master.async_server (proven in Phase 7.3). Replaces the fake async worker
used by the 7.3 test suite with the genuine article.

Each test drives its scenario with asyncio.run() (no pytest-asyncio
dependency, matching the rest of the async test suite).
"""

import asyncio
import time

import pytest

from common.models import TaskStatus, WorkerStatus
from master import async_server, rpc_handler
from rpc import protocol
from worker import async_worker


@pytest.fixture(autouse=True)
def reset_async_master_state():
    """Same singleton-reset reasoning as test_async_server.py: worker_manager
    and scheduler are module-level state shared across the whole test run."""
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


def test_worker_registers_successfully():
    async def scenario():
        server, host, port = await start_master_server()
        worker_task = asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-1"))
        try:
            await async_server.wait_for_workers(1)
            worker = async_server.worker_manager.get_worker("worker-1")
            assert worker is not None
            assert worker.status == WorkerStatus.IDLE
        finally:
            await stop_worker(worker_task)
            server.close()
            await server.wait_closed()

    asyncio.run(scenario())


def test_worker_executes_add_task():
    async def scenario():
        server, host, port = await start_master_server()
        worker_task = asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-1"))
        try:
            await async_server.wait_for_workers(1)
            async_server.scheduler.submit_task("task-1", "ADD", {"a": 10, "b": 20})
            return await async_server.drain_pending_tasks()
        finally:
            await stop_worker(worker_task)
            server.close()
            await server.wait_closed()

    responses = asyncio.run(scenario())
    assert responses[0]["payload"] == {"task_id": "task-1", "attempt": 1, "status": "success", "result": 30}


def test_worker_executes_multiply_task():
    async def scenario():
        server, host, port = await start_master_server()
        worker_task = asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-1"))
        try:
            await async_server.wait_for_workers(1)
            async_server.scheduler.submit_task("task-1", "MULTIPLY", {"a": 6, "b": 7})
            return await async_server.drain_pending_tasks()
        finally:
            await stop_worker(worker_task)
            server.close()
            await server.wait_closed()

    responses = asyncio.run(scenario())
    assert responses[0]["payload"]["status"] == "success"
    assert responses[0]["payload"]["result"] == 42


def test_worker_reports_execution_failure():
    async def scenario():
        server, host, port = await start_master_server()
        worker_task = asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-1"))
        try:
            await async_server.wait_for_workers(1)
            async_server.scheduler.submit_task("task-1", "ADD", {"a": "x", "b": 1})
            return await async_server.drain_pending_tasks()
        finally:
            await stop_worker(worker_task)
            server.close()
            await server.wait_closed()

    responses = asyncio.run(scenario())
    assert responses[0]["payload"]["status"] == "error"
    assert "message" in responses[0]["payload"]


def test_worker_executes_multiple_tasks_sequentially():
    async def scenario():
        server, host, port = await start_master_server()
        worker_task = asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-1"))
        try:
            await async_server.wait_for_workers(1)
            async_server.scheduler.submit_task("task-1", "ADD", {"a": 1, "b": 1})
            async_server.scheduler.submit_task("task-2", "MULTIPLY", {"a": 3, "b": 3})
            async_server.scheduler.submit_task("task-3", "ADD", {"a": 5, "b": 5})
            return await async_server.drain_pending_tasks()
        finally:
            await stop_worker(worker_task)
            server.close()
            await server.wait_closed()

    responses = asyncio.run(scenario())
    assert len(responses) == 3
    assert [r["payload"]["result"] for r in responses] == [2, 9, 10]
    statuses = {t.task_id: t.status for t in async_server.scheduler.get_all_tasks()}
    assert all(status == TaskStatus.COMPLETED for status in statuses.values())


def test_worker_sends_heartbeats_periodically():
    async def scenario():
        server, host, port = await start_master_server()
        worker_task = asyncio.create_task(
            async_worker.run_worker(host, port, worker_id="worker-1", heartbeat_interval=0.05)
        )
        try:
            await async_server.wait_for_workers(1)
            first = async_server.worker_manager.get_worker("worker-1").last_heartbeat

            deadline = time.monotonic() + 2
            second = first
            while second == first and time.monotonic() < deadline:
                await asyncio.sleep(0.02)
                second = async_server.worker_manager.get_worker("worker-1").last_heartbeat

            return first, second
        finally:
            await stop_worker(worker_task)
            server.close()
            await server.wait_closed()

    first, second = asyncio.run(scenario())
    assert second > first


def test_heartbeat_and_task_share_same_connection():
    """The whole point of Phase 7.4/7.3: a real worker's heartbeats keep
    flowing on the same connection a task is dispatched and executed on."""

    async def scenario():
        server, host, port = await start_master_server()
        worker_task = asyncio.create_task(
            async_worker.run_worker(host, port, worker_id="worker-1", heartbeat_interval=0.05)
        )
        try:
            await async_server.wait_for_workers(1)

            before = async_server.worker_manager.get_worker("worker-1").last_heartbeat
            await asyncio.sleep(0.2)  # let a few heartbeats land

            async_server.scheduler.submit_task("task-1", "ADD", {"a": 10, "b": 20})
            responses = await async_server.drain_pending_tasks()

            await asyncio.sleep(0.2)  # confirm heartbeats keep flowing afterward
            after = async_server.worker_manager.get_worker("worker-1").last_heartbeat

            return before, after, responses
        finally:
            await stop_worker(worker_task)
            server.close()
            await server.wait_closed()

    before, after, responses = asyncio.run(scenario())
    assert after > before
    assert responses[0]["payload"]["status"] == "success"
    assert responses[0]["payload"]["result"] == 30


def test_worker_disconnect_removes_it_from_connections():
    async def scenario():
        server, host, port = await start_master_server()
        worker_task = asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-1"))
        try:
            await async_server.wait_for_workers(1)
            assert "worker-1" in async_server.connections

            await stop_worker(worker_task)
            worker_task = None

            deadline = time.monotonic() + 2
            while "worker-1" in async_server.connections and time.monotonic() < deadline:
                await asyncio.sleep(0.02)

            assert "worker-1" not in async_server.connections
        finally:
            if worker_task is not None:
                await stop_worker(worker_task)
            server.close()
            await server.wait_closed()

    asyncio.run(scenario())


def test_attempt_is_echoed_across_retries():
    async def scenario():
        server, host, port = await start_master_server()
        worker_task = asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-1"))
        try:
            await async_server.wait_for_workers(1)

            task = async_server.scheduler.submit_task("task-1", "ADD", {"a": 1, "b": 1})
            async_server.scheduler.assign_task("task-1")
            assert task.attempt == 1
            first_response = await async_server.dispatch_assigned_task(task)
            assert first_response["payload"]["attempt"] == 1

            # Simulate a retry (task-1 already COMPLETED from attempt 1):
            # force it back to PENDING the way a real requeue would leave
            # it, then assign again and confirm attempt 2 is echoed.
            task.status = TaskStatus.PENDING
            task.assigned_worker_id = None
            reassigned = async_server.scheduler.assign_task("task-1")
            assert reassigned.attempt == 2

            second_response = await async_server.dispatch_assigned_task(task)
            return first_response, second_response
        finally:
            await stop_worker(worker_task)
            server.close()
            await server.wait_closed()

    first_response, second_response = asyncio.run(scenario())
    assert first_response["payload"]["attempt"] == 1
    assert second_response["payload"]["attempt"] == 2


def test_two_real_async_workers_share_the_load():
    async def scenario():
        server, host, port = await start_master_server()
        worker_tasks = [
            asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-1")),
            asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-2")),
        ]
        try:
            await async_server.wait_for_workers(2)

            for task_id, payload in [
                ("task-1", {"a": 1, "b": 1}),
                ("task-2", {"a": 2, "b": 2}),
                ("task-3", {"a": 3, "b": 3}),
                ("task-4", {"a": 4, "b": 4}),
            ]:
                async_server.scheduler.submit_task(task_id, "ADD", payload)

            return await async_server.drain_pending_tasks()
        finally:
            for t in worker_tasks:
                await stop_worker(t)
            server.close()
            await server.wait_closed()

    responses = asyncio.run(scenario())
    assert len(responses) == 4
    assert all(r["payload"]["status"] == "success" for r in responses)
    workers_used = {t.assigned_worker_id for t in async_server.scheduler.get_all_tasks()}
    assert workers_used == {"worker-1", "worker-2"}
