"""Phase 8.2: proves the EXISTING scheduler correctly distributes many Map
partitions across a worker pool -- no Map-specific scheduling logic is
added anywhere here. jobs/map.py's build_map_job() just calls
scheduler.submit_task() per partition; everything below is the same
Scheduler/dispatch/failure_monitor machinery already proven in Phases 4-7,
now exercised specifically through Map jobs.
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


async def run_crashing_worker_after_n_tasks(
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


def test_more_partitions_than_workers():
    """6 partitions, 2 workers: each worker must handle multiple partitions
    over several rounds, and everything still completes correctly."""

    async def scenario():
        server, host, port = await start_master_server()
        worker_tasks = [
            asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-1")),
            asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-2")),
        ]
        try:
            await async_server.wait_for_workers(2)
            tasks = build_map_job(async_server.scheduler, "job", "SQUARE", list(range(1, 13)), num_partitions=6)
            responses = await async_server.drain_pending_tasks()
            return tasks, responses
        finally:
            server.close()
            await server.wait_closed()
            for t in worker_tasks:
                await stop_worker(t)

    tasks, responses = asyncio.run(scenario())
    assert len(tasks) == 6
    assert all(t.status == TaskStatus.COMPLETED for t in tasks)

    per_worker = {}
    for t in tasks:
        per_worker[t.assigned_worker_id] = per_worker.get(t.assigned_worker_id, 0) + 1
    assert set(per_worker) == {"worker-1", "worker-2"}
    assert all(count > 1 for count in per_worker.values())  # each did more than one round

    assert collect_map_results(tasks, responses) == [x * x for x in range(1, 13)]


def test_fewer_partitions_than_workers():
    """2 partitions, 4 registered workers: only 2 should ever be used --
    the other 2 stay IDLE and untouched, exactly like non-Map task
    scheduling already guarantees (Scheduler doesn't know or care that
    these are Map tasks)."""

    async def scenario():
        server, host, port = await start_master_server()
        worker_tasks = [
            asyncio.create_task(async_worker.run_worker(host, port, worker_id=f"worker-{i}"))
            for i in range(1, 5)
        ]
        try:
            await async_server.wait_for_workers(4)
            tasks = build_map_job(async_server.scheduler, "job", "DOUBLE", [1, 2, 3, 4], num_partitions=2)
            responses = await async_server.drain_pending_tasks()
            idle_workers = {
                w.worker_id for w in async_server.worker_manager.get_all_workers() if w.status == WorkerStatus.IDLE
            }
            return tasks, responses, idle_workers
        finally:
            server.close()
            await server.wait_closed()
            for t in worker_tasks:
                await stop_worker(t)

    tasks, responses, idle_workers_after = asyncio.run(scenario())
    assert len(tasks) == 2
    assert all(t.status == TaskStatus.COMPLETED for t in tasks)

    used_workers = {t.assigned_worker_id for t in tasks}
    assert len(used_workers) == 2
    # All 4 are IDLE again once the (only) 2 partitions finish, but only 2
    # of them ever actually did anything.
    assert idle_workers_after == {"worker-1", "worker-2", "worker-3", "worker-4"}
    assert collect_map_results(tasks, responses) == [2, 4, 6, 8]


def test_map_partitions_execute_concurrently():
    """Deterministic (event-gated, not timing-based) proof that two
    partitions assigned to two different workers are genuinely RUNNING at
    the same time, not merely completing one after another."""

    async def scenario():
        server, host, port = await start_master_server()

        ready = {"worker-1": asyncio.Event(), "worker-2": asyncio.Event()}
        release = {"worker-1": asyncio.Event(), "worker-2": asyncio.Event()}

        async def gated_worker(worker_id: str) -> None:
            from rpc.async_rpc import send_message
            from rpc.protocol import build_message
            from worker.executor import execute_task

            reader, writer = await asyncio.open_connection(host, port)
            conn = AsyncConnection(reader, writer)
            await send_request(conn, protocol.PING)
            await send_request(conn, protocol.REGISTER, {"worker_id": worker_id, "host": "127.0.0.1", "port": 6000})

            message = await receive_message(conn)
            assert message["type"] == protocol.TASK
            ready[worker_id].set()
            await release[worker_id].wait()

            payload = message["payload"]
            result = execute_task(payload["task_type"], payload["task_payload"])
            response = build_message(
                protocol.TASK_RESULT,
                message["request_id"],
                {"task_id": payload["task_id"], "attempt": payload.get("attempt", 1), **result},
            )
            await send_message(conn, response)
            await conn.close()

        worker_tasks = [asyncio.create_task(gated_worker(wid)) for wid in ("worker-1", "worker-2")]
        try:
            await async_server.wait_for_workers(2)
            tasks = build_map_job(async_server.scheduler, "job", "SQUARE", [1, 2, 3, 4], num_partitions=2)

            drain_task = asyncio.create_task(async_server.drain_pending_tasks())

            await asyncio.wait_for(ready["worker-1"].wait(), timeout=5)
            await asyncio.wait_for(ready["worker-2"].wait(), timeout=5)

            assert all(t.status == TaskStatus.RUNNING for t in tasks)

            release["worker-1"].set()
            release["worker-2"].set()

            responses = await drain_task
            return tasks, responses
        finally:
            server.close()
            await server.wait_closed()
            for t in worker_tasks:
                await stop_worker(t)

    tasks, responses = asyncio.run(scenario())
    assert collect_map_results(tasks, responses) == [1, 4, 9, 16]


def test_empty_dataset_end_to_end():
    """An empty dataset submits zero tasks and needs no worker at all --
    proven over the real transport, not just the pure unit test."""

    async def scenario():
        server, host, port = await start_master_server()
        worker_task = asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-1"))
        try:
            await async_server.wait_for_workers(1)
            tasks = build_map_job(async_server.scheduler, "job", "SQUARE", [], num_partitions=4)
            responses = await async_server.drain_pending_tasks()
            return tasks, responses
        finally:
            await stop_worker(worker_task)
            server.close()
            await server.wait_closed()

    tasks, responses = asyncio.run(scenario())
    assert tasks == []
    assert responses == []
    assert collect_map_results(tasks, responses) == []


def test_worker_failure_during_map_distribution_with_survivor(monkeypatch):
    """5 partitions, 2 workers. worker-1 is deliberately handed 2 of them
    (via the same IDLE-toggle trick used elsewhere to stack multiple tasks
    on one worker) and crashes holding both; the other 3 partitions are
    still PENDING at that point. Once worker-1 is marked FAILED and its 2
    are requeued, a single drain_pending_tasks() call hands worker-2 (the
    only survivor) all 5 partitions across several rounds. All 5 end
    COMPLETED with the correct combined result."""
    monkeypatch.setattr(async_server, "FAILURE_CHECK_INTERVAL", 0.05)
    monkeypatch.setattr(async_server, "HEARTBEAT_TIMEOUT", 1.0)

    async def scenario():
        server, host, port = await start_master_server()

        ready = asyncio.Event()
        worker1_task = asyncio.create_task(
            run_crashing_worker_after_n_tasks(host, port, "worker-1", 2, ready)
        )
        await async_server.wait_for_workers(1)
        worker2_task = asyncio.create_task(
            async_worker.run_worker(host, port, worker_id="worker-2", heartbeat_interval=0.1)
        )
        await async_server.wait_for_workers(2)

        stop_event = asyncio.Event()
        monitor_task = asyncio.create_task(async_server.failure_monitor(stop_event))

        try:
            tasks = build_map_job(async_server.scheduler, "job", "INCREMENT", list(range(1, 11)), num_partitions=5)

            # Put 2 partitions on worker-1 (it will crash holding both) via
            # the same IDLE-toggle technique used elsewhere for stacking
            # multiple tasks on one worker -- assign_task only ever picks
            # an IDLE worker, so toggling status between calls is what lets
            # one worker end up holding more than one.
            async_server.scheduler.assign_task(tasks[0].task_id)
            async_server.worker_manager.update_status("worker-1", WorkerStatus.IDLE)
            async_server.scheduler.assign_task(tasks[1].task_id)
            assert tasks[0].assigned_worker_id == "worker-1"
            assert tasks[1].assigned_worker_id == "worker-1"

            dispatch0 = asyncio.create_task(async_server.dispatch_assigned_task(tasks[0]))
            dispatch1 = asyncio.create_task(async_server.dispatch_assigned_task(tasks[1]))

            await asyncio.wait_for(ready.wait(), timeout=5)

            deadline = time.monotonic() + 5
            while (
                async_server.worker_manager.get_worker("worker-1").status != WorkerStatus.FAILED
                and time.monotonic() < deadline
            ):
                await asyncio.sleep(0.02)
            assert async_server.worker_manager.get_worker("worker-1").status == WorkerStatus.FAILED

            await dispatch0
            await dispatch1

            responses = await async_server.drain_pending_tasks()
            return tasks, responses
        finally:
            stop_event.set()
            await monitor_task
            await stop_worker(worker1_task)
            await stop_worker(worker2_task)
            server.close()
            await server.wait_closed()

    tasks, responses = asyncio.run(scenario())
    assert all(t.status == TaskStatus.COMPLETED for t in tasks)
    assert all(t.assigned_worker_id == "worker-2" for t in tasks[:2])  # requeued survivors
    assert collect_map_results(tasks, responses) == [x + 1 for x in range(1, 11)]
