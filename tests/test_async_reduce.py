"""Phase 8.5: real worker execution of REDUCE tasks, and the full
Map -> Shuffle -> Reduce pipeline end-to-end over the actual async
transport. No changes to the scheduler, transport, or worker dispatch
logic -- REDUCE is just another task_type, dispatched exactly like
ADD/MULTIPLY/MAP.
"""

import asyncio

import pytest

from jobs.map import build_intermediate_results, build_map_job
from jobs.reduce import build_reduce_job, collect_reduce_results, reduce_grouped
from jobs.shuffle import shuffle
from master import async_server, rpc_handler
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


def test_real_worker_executes_reduce_tasks():
    """One REDUCE task per key, dispatched to real workers -- proving
    REDUCE is a genuine distributed task type, not just a local function."""

    async def scenario():
        server, host, port = await start_master_server()
        worker_tasks = [
            asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-1")),
            asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-2")),
        ]
        try:
            await async_server.wait_for_workers(2)

            grouped = {"apple": [1, 1, 1, 1], "banana": [1, 1, 1], "orange": [1, 1]}
            tasks = build_reduce_job(async_server.scheduler, "job-1", grouped, "SUM")
            responses = await async_server.drain_pending_tasks()

            return tasks, responses
        finally:
            server.close()
            await server.wait_closed()
            for t in worker_tasks:
                await stop_worker(t)

    tasks, responses = asyncio.run(scenario())
    reduced = collect_reduce_results(tasks, responses)
    assert reduced == {"apple": 4, "banana": 3, "orange": 2}

    workers_used = {t.assigned_worker_id for t in tasks.values()}
    assert workers_used == {"worker-1", "worker-2"}


def test_dispatched_reduce_matches_local_reduce_grouped():
    """Local reduce_grouped() and a real distributed REDUCE dispatch must
    compute the identical answer for the same (operation, grouped) input --
    they share the same worker.executor.execute_reduce underneath."""

    async def scenario():
        server, host, port = await start_master_server()
        worker_task = asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-1"))
        try:
            await async_server.wait_for_workers(1)
            grouped = {"apple": [3, 7, 2], "banana": [10, 5]}
            tasks = build_reduce_job(async_server.scheduler, "job-1", grouped, "MAX")
            responses = await async_server.drain_pending_tasks()
            return grouped, tasks, responses
        finally:
            await stop_worker(worker_task)
            server.close()
            await server.wait_closed()

    grouped, tasks, responses = asyncio.run(scenario())
    dispatched_result = collect_reduce_results(tasks, responses)
    local_result = reduce_grouped(grouped, "MAX")
    assert dispatched_result == local_result == {"apple": 7, "banana": 10}


def test_full_map_shuffle_reduce_pipeline_word_count():
    """The complete pipeline, end-to-end: real Map (WORD_COUNT) -> Shuffle
    -> real Reduce (SUM), across real async workers."""
    words = [
        "apple", "banana", "apple",
        "orange", "banana", "apple",
        "banana", "orange", "apple",
    ]

    async def scenario():
        server, host, port = await start_master_server()
        worker_tasks = [
            asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-1")),
            asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-2")),
        ]
        try:
            await async_server.wait_for_workers(2)

            map_tasks = build_map_job(async_server.scheduler, "wc-job", "WORD_COUNT", words, num_partitions=3)
            map_responses = await async_server.drain_pending_tasks()
            intermediate = build_intermediate_results("wc-job", map_tasks, map_responses)

            grouped = shuffle(intermediate)

            reduce_tasks = build_reduce_job(async_server.scheduler, "wc-job", grouped, "SUM")
            reduce_responses = await async_server.drain_pending_tasks()

            return reduce_tasks, reduce_responses
        finally:
            server.close()
            await server.wait_closed()
            for t in worker_tasks:
                await stop_worker(t)

    reduce_tasks, reduce_responses = asyncio.run(scenario())
    final = collect_reduce_results(reduce_tasks, reduce_responses)
    assert final == {"apple": 4, "banana": 3, "orange": 2}


def test_collect_reduce_results_raises_on_real_worker_failure():
    """A REDUCE task that genuinely fails at the executor (bad values) must
    surface explicitly, never be silently absorbed into the final dict."""

    async def scenario():
        server, host, port = await start_master_server()
        worker_task = asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-1"))
        try:
            await async_server.wait_for_workers(1)
            # An empty values list is rejected by execute_reduce.
            tasks = build_reduce_job(async_server.scheduler, "job-1", {"apple": []}, "SUM")
            responses = await async_server.drain_pending_tasks()
            return tasks, responses
        finally:
            await stop_worker(worker_task)
            server.close()
            await server.wait_closed()

    tasks, responses = asyncio.run(scenario())
    assert tasks["apple"].status == "FAILED"
    with pytest.raises(ValueError):
        collect_reduce_results(tasks, responses)
