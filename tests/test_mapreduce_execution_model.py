"""Phase 10.5: MapReduce integration with the function-execution model
(registered functions and serialized callables), plus explicit backward
compatibility confirmation for the pre-existing built-in-operation path.

worker.executor.execute_map/execute_reduce fall back to worker.registry
for whatever operation name isn't a built-in (SQUARE, WORD_COUNT, SUM,
...), and accept an execution_mode: "serialized_callable" payload for an
arbitrary callable -- see jobs.map.build_map_job_serialized and
jobs.reduce.build_reduce_job_serialized. jobs.map_reduce.run_map_reduce
itself needed NO changes for the "registered" case: map_operation/
reduce_operation were always just strings passed through to
build_map_job/build_reduce_job, so a registered function's name works
exactly like ADD or SUM already did. The "serialized_callable" case isn't
wired into run_map_reduce's fixed built-in/registered string-operation
signature, so those tests compose the same Map -> Shuffle -> Reduce
pipeline manually, calling build_map_job_serialized/build_reduce_job_serialized
in run_map_reduce's place.
"""

import asyncio

import pytest

from jobs.map import build_intermediate_results, build_map_job, build_map_job_serialized, collect_map_results
from jobs.map_reduce import run_map_reduce
from jobs.reduce import build_reduce_job, build_reduce_job_serialized, collect_reduce_results
from jobs.shuffle import shuffle
from master import async_server, rpc_handler
from worker import async_worker, registry


@pytest.fixture(autouse=True)
def reset_async_master_state():
    rpc_handler.worker_manager.clear()
    async_server.scheduler.clear()
    async_server.connections.clear()
    registry.clear()
    yield
    registry.clear()


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


def test_mapreduce_word_count_using_a_registered_map_function():
    """A more representative case: a registered function replacing
    WORD_COUNT's exact behavior, reduced with the built-in SUM -- proving
    a registered MAP function and a built-in REDUCE operation compose."""
    registry.register_function("count_word", lambda word: [word, 1])

    async def scenario():
        server, host, port = await start_master_server()
        worker_tasks = [
            asyncio.create_task(async_worker.run_worker(host, port, worker_id=f"worker-{i}")) for i in range(1, 3)
        ]
        await async_server.wait_for_workers(2)

        try:
            return await run_map_reduce(
                async_server.scheduler,
                async_server.wait_for_tasks,
                "job-mixed-1",
                ["apple", "banana", "apple", "cherry", "banana", "apple"],
                "count_word",
                "SUM",
                num_partitions=2,
            )
        finally:
            for t in worker_tasks:
                await stop_worker(t)
            server.close()
            await server.wait_closed()

    result = asyncio.run(scenario())
    assert result == {"apple": 3, "banana": 2, "cherry": 1}


def test_mapreduce_word_count_using_a_registered_reduce_function():
    """The inverse composition: built-in WORD_COUNT for MAP, a registered
    function for REDUCE."""
    registry.register_function("total", sum)

    async def scenario():
        server, host, port = await start_master_server()
        worker_tasks = [
            asyncio.create_task(async_worker.run_worker(host, port, worker_id=f"worker-{i}")) for i in range(1, 3)
        ]
        await async_server.wait_for_workers(2)

        try:
            return await run_map_reduce(
                async_server.scheduler,
                async_server.wait_for_tasks,
                "job-mixed-2",
                ["apple", "banana", "apple"],
                "WORD_COUNT",
                "total",
                num_partitions=2,
            )
        finally:
            for t in worker_tasks:
                await stop_worker(t)
            server.close()
            await server.wait_closed()

    result = asyncio.run(scenario())
    assert result == {"apple": 2, "banana": 1}


def test_mapreduce_using_serialized_callables_for_both_map_and_reduce():
    """Manually composed Map -> Shuffle -> Reduce using
    build_map_job_serialized/build_reduce_job_serialized in place of
    run_map_reduce's built-in/registered string-operation call --
    proving the same pipeline (Scheduler, dispatch, Shuffle) works with
    arbitrary serialized callables end to end."""

    def count_word(word):
        return [word, 1]

    def total(values):
        return sum(values)

    async def scenario():
        server, host, port = await start_master_server()
        worker_tasks = [
            asyncio.create_task(async_worker.run_worker(host, port, worker_id=f"worker-{i}")) for i in range(1, 3)
        ]
        await async_server.wait_for_workers(2)

        try:
            job_id = "job-serialized"
            words = ["apple", "banana", "apple", "cherry", "banana", "apple"]

            map_tasks = build_map_job_serialized(async_server.scheduler, job_id, count_word, words, num_partitions=2)
            map_responses = await async_server.wait_for_tasks({t.task_id for t in map_tasks})
            intermediate = build_intermediate_results(job_id, map_tasks, map_responses)

            grouped = shuffle(intermediate)

            reduce_tasks = build_reduce_job_serialized(async_server.scheduler, job_id, grouped, total)
            reduce_responses = await async_server.wait_for_tasks({t.task_id for t in reduce_tasks.values()})
            return collect_reduce_results(reduce_tasks, reduce_responses)
        finally:
            for t in worker_tasks:
                await stop_worker(t)
            server.close()
            await server.wait_closed()

    result = asyncio.run(scenario())
    assert result == {"apple": 3, "banana": 2, "cherry": 1}


def test_mapreduce_mixing_serialized_map_with_registered_reduce():
    """A serialized MAP callable feeding a REGISTERED reduce function --
    proving the two Phase 10 execution modes compose within one job, not
    just independently."""
    registry.register_function("total", sum)

    def count_word(word):
        return [word, 1]

    async def scenario():
        server, host, port = await start_master_server()
        worker_tasks = [
            asyncio.create_task(async_worker.run_worker(host, port, worker_id=f"worker-{i}")) for i in range(1, 3)
        ]
        await async_server.wait_for_workers(2)

        try:
            job_id = "job-mixed-modes"
            words = ["x", "y", "x"]

            map_tasks = build_map_job_serialized(async_server.scheduler, job_id, count_word, words, num_partitions=2)
            map_responses = await async_server.wait_for_tasks({t.task_id for t in map_tasks})
            intermediate = build_intermediate_results(job_id, map_tasks, map_responses)
            grouped = shuffle(intermediate)

            reduce_tasks = build_reduce_job(async_server.scheduler, job_id, grouped, "total")
            reduce_responses = await async_server.wait_for_tasks({t.task_id for t in reduce_tasks.values()})
            return collect_reduce_results(reduce_tasks, reduce_responses)
        finally:
            for t in worker_tasks:
                await stop_worker(t)
            server.close()
            await server.wait_closed()

    result = asyncio.run(scenario())
    assert result == {"x": 2, "y": 1}


def test_mapreduce_backward_compatibility_with_existing_builtin_word_count():
    """Explicit confirmation, alongside the pre-existing MapReduce test
    suite: the original built-in WORD_COUNT/SUM path is byte-for-byte
    unaffected by the registry/serialization fallback added in Phase
    10.5 -- MAP_OPERATIONS/REDUCE_OPERATIONS are still checked first,
    before any registry lookup is even attempted."""

    async def scenario():
        server, host, port = await start_master_server()
        worker_tasks = [
            asyncio.create_task(async_worker.run_worker(host, port, worker_id=f"worker-{i}")) for i in range(1, 3)
        ]
        await async_server.wait_for_workers(2)

        try:
            return await run_map_reduce(
                async_server.scheduler,
                async_server.wait_for_tasks,
                "job-builtin",
                ["apple", "banana", "apple", "cherry", "banana", "apple"],
                "WORD_COUNT",
                "SUM",
                num_partitions=3,
            )
        finally:
            for t in worker_tasks:
                await stop_worker(t)
            server.close()
            await server.wait_closed()

    result = asyncio.run(scenario())
    assert result == {"apple": 3, "banana": 2, "cherry": 1}
