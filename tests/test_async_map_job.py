"""Integration tests proving MAP jobs work end-to-end over the real async
transport, reusing the existing task execution infrastructure entirely --
no MapReduce-specific dispatch, scheduler, or worker code. A MAP task is
just another task_type flowing through master.async_server's existing
Scheduler / dispatch_assigned_task / drain_pending_tasks / failure_monitor.
"""

import asyncio
import time

import pytest

from common.models import TaskStatus, WorkerStatus
from jobs.map import build_map_job, collect_map_results
from master import async_server, rpc_handler
from rpc import protocol
from rpc.async_connection import AsyncConnection
from rpc.async_rpc import receive_message, send_request
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


async def run_multi_task_crashing_worker(
    host: str, port: int, worker_id: str, task_count: int, ready: asyncio.Event
) -> None:
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


def test_single_worker_map_execution():
    async def scenario():
        server, host, port = await start_master_server()
        worker_task = asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-1"))
        try:
            await async_server.wait_for_workers(1)

            tasks = build_map_job(async_server.scheduler, "job-1", "SQUARE", [1, 2, 3, 4], num_partitions=1)
            responses = await async_server.drain_pending_tasks()

            return tasks, responses
        finally:
            await stop_worker(worker_task)
            server.close()
            await server.wait_closed()

    tasks, responses = asyncio.run(scenario())
    assert collect_map_results(tasks, responses) == [1, 4, 9, 16]
    assert all(t.status == TaskStatus.COMPLETED for t in tasks)


def test_multi_worker_map_execution():
    """5 partitions across 2 real workers (more partitions than workers, so
    drain_pending_tasks needs multiple rounds) -- proves distribution and
    collection both work, matching values regardless of which worker
    handled which partition. The reordering logic in collect_map_results
    itself (responses arriving out of partition order) is proven directly,
    without real-TCP timing dependencies, by
    test_jobs_map.py::test_collect_map_results_in_partition_order_regardless_of_response_order.
    """

    async def scenario():
        server, host, port = await start_master_server()
        worker_tasks = [
            asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-1")),
            asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-2")),
        ]
        try:
            await async_server.wait_for_workers(2)

            tasks = build_map_job(
                async_server.scheduler, "job-1", "DOUBLE", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], num_partitions=5
            )
            responses = await async_server.drain_pending_tasks()
            return tasks, responses
        finally:
            server.close()
            await server.wait_closed()
            for t in worker_tasks:
                await stop_worker(t)

    tasks, responses = asyncio.run(scenario())
    assert len(tasks) == 5
    assert all(t.status == TaskStatus.COMPLETED for t in tasks)
    workers_used = {t.assigned_worker_id for t in tasks}
    assert workers_used == {"worker-1", "worker-2"}
    assert collect_map_results(tasks, responses) == [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]


def test_map_job_with_invalid_operation_reports_task_failure():
    async def scenario():
        server, host, port = await start_master_server()
        worker_task = asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-1"))
        try:
            await async_server.wait_for_workers(1)
            tasks = build_map_job(async_server.scheduler, "job-1", "CUBE", [1, 2, 3], num_partitions=1)
            responses = await async_server.drain_pending_tasks()
            return tasks, responses
        finally:
            await stop_worker(worker_task)
            server.close()
            await server.wait_closed()

    tasks, responses = asyncio.run(scenario())
    assert tasks[0].status == TaskStatus.FAILED
    with pytest.raises(ValueError):
        collect_map_results(tasks, responses)


def test_map_task_is_retried_after_worker_failure(monkeypatch):
    """A MAP task's worker crashes mid-task; the existing failure-detection
    and requeue mechanism recovers it exactly like an ADD/MULTIPLY task,
    with attempt incrementing to 2."""
    monkeypatch.setattr(async_server, "FAILURE_CHECK_INTERVAL", 0.05)
    monkeypatch.setattr(async_server, "HEARTBEAT_TIMEOUT", 1.0)

    async def scenario():
        server, host, port = await start_master_server()

        ready = asyncio.Event()
        worker1_task = asyncio.create_task(run_multi_task_crashing_worker(host, port, "worker-1", 1, ready))
        await async_server.wait_for_workers(1)
        worker2_task = asyncio.create_task(
            async_worker.run_worker(host, port, worker_id="worker-2", heartbeat_interval=0.1)
        )
        await async_server.wait_for_workers(2)

        stop_event = asyncio.Event()
        monitor_task = asyncio.create_task(async_server.failure_monitor(stop_event))

        try:
            tasks = build_map_job(async_server.scheduler, "job-1", "SQUARE", [2, 3], num_partitions=1)
            task = tasks[0]
            assigned = async_server.scheduler.assign_task(task.task_id)
            assert assigned.assigned_worker_id == "worker-1"
            assert assigned.attempt == 1

            dispatch1 = asyncio.create_task(async_server.dispatch_assigned_task(task))
            await asyncio.wait_for(ready.wait(), timeout=5)

            deadline = time.monotonic() + 5
            while task.status != TaskStatus.COMPLETED and time.monotonic() < deadline:
                await asyncio.sleep(0.02)

            assert async_server.worker_manager.get_worker("worker-1").status == WorkerStatus.FAILED
            assert task.status == TaskStatus.COMPLETED
            assert task.assigned_worker_id == "worker-2"
            assert task.attempt == 2

            await dispatch1  # unwinds the dead first attempt without corrupting the above
            return tasks
        finally:
            stop_event.set()
            await monitor_task
            await stop_worker(worker1_task)
            await stop_worker(worker2_task)
            server.close()
            await server.wait_closed()

    tasks = asyncio.run(scenario())
    assert tasks[0].status == TaskStatus.COMPLETED
    assert tasks[0].attempt == 2
