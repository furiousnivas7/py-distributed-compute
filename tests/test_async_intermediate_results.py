"""Phase 8.3: real async integration for the intermediate-result layer.

No changes to the scheduler, transport, or worker -- build_intermediate_results
and IntermediateResultStore are pure post-processing over the same
TASK_RESULT responses drain_pending_tasks() already produces.
"""

import asyncio

import pytest

from common.models import TaskStatus
from jobs.map import build_intermediate_results, build_map_job
from jobs.models import IntermediateResultStore, ResultStatus
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


def test_map_job_produces_stored_intermediate_results():
    async def scenario():
        server, host, port = await start_master_server()
        worker_tasks = [
            asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-1")),
            asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-2")),
        ]
        try:
            await async_server.wait_for_workers(2)

            tasks = build_map_job(async_server.scheduler, "job-1", "SQUARE", [1, 2, 3, 4, 5, 6], num_partitions=3)
            responses = await async_server.drain_pending_tasks()

            results = build_intermediate_results("job-1", tasks, responses)

            store = IntermediateResultStore()
            store.store("job-1", results)
            return store
        finally:
            server.close()
            await server.wait_closed()
            for t in worker_tasks:
                await stop_worker(t)

    store = asyncio.run(scenario())
    results = store.get("job-1")

    assert len(results) == 3
    assert [r.partition_id for r in results] == [0, 1, 2]
    assert all(r.status == ResultStatus.SUCCESS for r in results)
    assert all(r.job_id == "job-1" for r in results)
    assert all(r.worker_id in ("worker-1", "worker-2") for r in results)
    # Reassembling the partitions' data in order reproduces the full mapped list.
    flattened = [x for r in results for x in r.data]
    assert flattened == [1, 4, 9, 16, 25, 36]


def test_two_map_jobs_do_not_mix_results_in_the_store():
    async def scenario():
        server, host, port = await start_master_server()
        worker_task = asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-1"))
        try:
            await async_server.wait_for_workers(1)

            tasks_a = build_map_job(async_server.scheduler, "job-a", "SQUARE", [1, 2], num_partitions=1)
            responses_a = await async_server.drain_pending_tasks()
            results_a = build_intermediate_results("job-a", tasks_a, responses_a)

            tasks_b = build_map_job(async_server.scheduler, "job-b", "DOUBLE", [10, 20], num_partitions=1)
            responses_b = await async_server.drain_pending_tasks()
            results_b = build_intermediate_results("job-b", tasks_b, responses_b)

            store = IntermediateResultStore()
            store.store("job-a", results_a)
            store.store("job-b", results_b)
            return store
        finally:
            await stop_worker(worker_task)
            server.close()
            await server.wait_closed()

    store = asyncio.run(scenario())

    job_a = store.get("job-a")
    job_b = store.get("job-b")
    assert [r.data for r in job_a] == [[1, 4]]
    assert [r.data for r in job_b] == [[20, 40]]
    assert all(r.job_id == "job-a" for r in job_a)
    assert all(r.job_id == "job-b" for r in job_b)


def test_intermediate_results_capture_worker_failure_without_raising(monkeypatch):
    """A Map task that fails at the executor level (bad operation) should
    show up as an ERROR IntermediateResult, not blow up the whole job."""

    async def scenario():
        server, host, port = await start_master_server()
        worker_task = asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-1"))
        try:
            await async_server.wait_for_workers(1)

            tasks = build_map_job(async_server.scheduler, "job-1", "CUBE", [1, 2, 3], num_partitions=1)
            responses = await async_server.drain_pending_tasks()
            return build_intermediate_results("job-1", tasks, responses)
        finally:
            await stop_worker(worker_task)
            server.close()
            await server.wait_closed()

    results = asyncio.run(scenario())
    assert len(results) == 1
    assert results[0].status == ResultStatus.ERROR
    assert results[0].data is None
    assert "CUBE" in results[0].message
