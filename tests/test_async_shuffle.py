"""Phase 8.4: real end-to-end Map -> IntermediateResultStore -> Shuffle,
over the actual async transport. No changes to the scheduler, transport,
or worker dispatch logic -- shuffle() is pure post-processing over the same
IntermediateResult objects Phase 8.3 already produces.
"""

import asyncio

import pytest

from jobs.map import build_intermediate_results, build_map_job
from jobs.models import IntermediateResultStore
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


def test_word_count_map_job_shuffles_correctly_across_two_workers():
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

            tasks = build_map_job(async_server.scheduler, "wordcount-job", "WORD_COUNT", words, num_partitions=3)
            responses = await async_server.drain_pending_tasks()
            results = build_intermediate_results("wordcount-job", tasks, responses)

            store = IntermediateResultStore()
            store.store("wordcount-job", results)
            return store
        finally:
            server.close()
            await server.wait_closed()
            for t in worker_tasks:
                await stop_worker(t)

    store = asyncio.run(scenario())
    results = store.get("wordcount-job")

    assert len(results) == 3
    assert all(r.status == "success" for r in results)

    grouped = shuffle(results)

    assert grouped["apple"] == [1, 1, 1, 1]
    assert grouped["banana"] == [1, 1, 1]
    assert grouped["orange"] == [1, 1]
    assert sum(len(v) for v in grouped.values()) == len(words)


def test_shuffle_skips_failed_partition_in_real_job():
    """One partition uses an invalid operation and fails at the executor;
    shuffle must still group whatever the successful partitions produced."""

    async def scenario():
        server, host, port = await start_master_server()
        worker_task = asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-1"))
        try:
            await async_server.wait_for_workers(1)

            # Submit two separate single-partition jobs: one valid, one that
            # will fail -- simpler and more explicit than trying to force a
            # specific partition of one job to use a different operation.
            good_tasks = build_map_job(
                async_server.scheduler, "job-good", "WORD_COUNT", ["apple", "apple"], num_partitions=1
            )
            good_responses = await async_server.drain_pending_tasks()
            good_results = build_intermediate_results("job-good", good_tasks, good_responses)

            bad_tasks = build_map_job(
                async_server.scheduler, "job-bad", "WORD_COUNT", [1, 2], num_partitions=1
            )
            bad_responses = await async_server.drain_pending_tasks()
            bad_results = build_intermediate_results("job-bad", bad_tasks, bad_responses)

            return good_results, bad_results
        finally:
            await stop_worker(worker_task)
            server.close()
            await server.wait_closed()

    good_results, bad_results = asyncio.run(scenario())

    assert bad_results[0].status == "error"
    assert shuffle(bad_results) == {}
    assert shuffle(good_results) == {"apple": [1, 1]}
