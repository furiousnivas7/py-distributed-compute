"""Phase 10.5: MapReduce integration with the function-execution model
(registered functions and serialized callables), plus explicit backward
compatibility confirmation for the pre-existing built-in-operation path.

worker.executor.execute_map/execute_reduce fall back to worker.registry
for whatever operation name isn't a built-in (SQUARE, WORD_COUNT, SUM,
...), and accept an execution_mode: "serialized_callable" payload for an
arbitrary callable. jobs.models.ExecutionSpec is what lets ONE public API
-- build_map_job, build_reduce_job, and run_map_reduce itself -- accept
either: a plain string (shorthand for ExecutionSpec.registered(string),
so every pre-10.5 caller keeps working completely unchanged) or
ExecutionSpec.serialized(fn) for an arbitrary callable. There is no
separate "_serialized" pipeline -- these tests prove run_map_reduce
itself, not a manually-composed Map/Shuffle/Reduce, supports all four
map/reduce execution-mode combinations.
"""

import asyncio

import pytest

from jobs.map_reduce import run_map_reduce
from jobs.models import ExecutionSpec
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


async def _run_word_count_job(job_id, words, map_operation, reduce_operation, num_partitions=2):
    server, host, port = await start_master_server()
    worker_tasks = [
        asyncio.create_task(async_worker.run_worker(host, port, worker_id=f"worker-{i}")) for i in range(1, 3)
    ]
    await async_server.wait_for_workers(2)

    try:
        return await run_map_reduce(
            async_server.scheduler,
            async_server.wait_for_tasks,
            job_id,
            words,
            map_operation,
            reduce_operation,
            num_partitions=num_partitions,
        )
    finally:
        for t in worker_tasks:
            await stop_worker(t)
        server.close()
        await server.wait_closed()


WORDS = ["apple", "banana", "apple", "cherry", "banana", "apple"]
EXPECTED = {"apple": 3, "banana": 2, "cherry": 1}


def test_run_map_reduce_with_registered_map_and_registered_reduce():
    registry.register_function("count_word", lambda word: [word, 1])
    registry.register_function("total", sum)

    result = asyncio.run(_run_word_count_job("job-rr", WORDS, "count_word", "total"))
    assert result == EXPECTED


def test_run_map_reduce_with_serialized_map_and_serialized_reduce():
    def count_word(word):
        return [word, 1]

    def total(values):
        return sum(values)

    result = asyncio.run(
        _run_word_count_job(
            "job-ss", WORDS, ExecutionSpec.serialized(count_word), ExecutionSpec.serialized(total)
        )
    )
    assert result == EXPECTED


def test_run_map_reduce_with_registered_map_and_serialized_reduce():
    registry.register_function("count_word", lambda word: [word, 1])

    def total(values):
        return sum(values)

    result = asyncio.run(_run_word_count_job("job-rs", WORDS, "count_word", ExecutionSpec.serialized(total)))
    assert result == EXPECTED


def test_run_map_reduce_with_serialized_map_and_registered_reduce():
    def count_word(word):
        return [word, 1]

    registry.register_function("total", sum)

    result = asyncio.run(_run_word_count_job("job-sr", WORDS, ExecutionSpec.serialized(count_word), "total"))
    assert result == EXPECTED


def test_run_map_reduce_with_builtin_map_and_registered_reduce():
    """Mixing a THIRD variety in -- a built-in -- alongside registered,
    proving all three operation kinds (built-in, registered, serialized)
    compose freely within run_map_reduce, not just registered/serialized
    against each other."""
    registry.register_function("total", sum)

    result = asyncio.run(_run_word_count_job("job-br", WORDS, "WORD_COUNT", "total"))
    assert result == EXPECTED


def test_run_map_reduce_with_registered_map_and_builtin_reduce():
    registry.register_function("count_word", lambda word: [word, 1])

    result = asyncio.run(_run_word_count_job("job-rb", WORDS, "count_word", "SUM"))
    assert result == EXPECTED


def test_run_map_reduce_backward_compatibility_with_existing_builtin_word_count():
    """Explicit confirmation, alongside the pre-existing MapReduce test
    suite: the original built-in WORD_COUNT/SUM path -- plain strings,
    the exact same call shape used since Phase 8 -- is byte-for-byte
    unaffected by ExecutionSpec.coerce's involvement."""
    result = asyncio.run(_run_word_count_job("job-builtin", WORDS, "WORD_COUNT", "SUM", num_partitions=3))
    assert result == EXPECTED


def test_execution_spec_coerce_rejects_an_unsupported_type():
    with pytest.raises(TypeError):
        ExecutionSpec.coerce(42)


def test_execution_spec_coerce_is_idempotent_on_an_existing_spec():
    spec = ExecutionSpec.registered("SUM")
    assert ExecutionSpec.coerce(spec) is spec
