"""Phase 8.6: end-to-end MapReduce orchestration, over real async workers.

run_map_reduce() adds no new scheduling/transport/execution logic -- every
test here is really re-proving that Map, Shuffle, and Reduce (each already
proven independently in Phases 8.1-8.5) compose correctly when wired
together, including the deliberate asymmetry in failure handling:
a failed Map partition is tolerated, a failed Reduce key is not.
"""

import asyncio

import pytest

from common.models import TaskStatus
from jobs.map_reduce import run_map_reduce
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


WORDS = ["apple", "banana", "apple", "orange", "banana", "apple"]


def test_full_word_count_map_reduce_job():
    """The user-facing example, verbatim."""

    async def scenario():
        server, host, port = await start_master_server()
        worker_tasks = [
            asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-1")),
            asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-2")),
        ]
        try:
            await async_server.wait_for_workers(2)
            return await run_map_reduce(
                async_server.scheduler,
                async_server.wait_for_tasks,
                "job-1",
                WORDS,
                map_operation="WORD_COUNT",
                reduce_operation="SUM",
                num_partitions=2,
            )
        finally:
            server.close()
            await server.wait_closed()
            for t in worker_tasks:
                await stop_worker(t)

    result = asyncio.run(scenario())
    assert result == {"apple": 3, "banana": 2, "orange": 1}


@pytest.mark.parametrize("num_partitions", [1, 2, 3, 6, 10])
def test_result_is_independent_of_partition_count(num_partitions):
    async def scenario():
        server, host, port = await start_master_server()
        worker_tasks = [
            asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-1")),
            asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-2")),
        ]
        try:
            await async_server.wait_for_workers(2)
            return await run_map_reduce(
                async_server.scheduler,
                async_server.wait_for_tasks,
                "job-1",
                WORDS,
                "WORD_COUNT",
                "SUM",
                num_partitions=num_partitions,
            )
        finally:
            server.close()
            await server.wait_closed()
            for t in worker_tasks:
                await stop_worker(t)

    assert asyncio.run(scenario()) == {"apple": 3, "banana": 2, "orange": 1}


def test_three_workers_share_the_job():
    async def scenario():
        server, host, port = await start_master_server()
        worker_tasks = [
            asyncio.create_task(async_worker.run_worker(host, port, worker_id=f"worker-{i}"))
            for i in range(1, 4)
        ]
        try:
            await async_server.wait_for_workers(3)
            result = await run_map_reduce(
                async_server.scheduler,
                async_server.wait_for_tasks,
                "job-1",
                WORDS,
                "WORD_COUNT",
                "SUM",
                num_partitions=3,
            )
            workers_used = {
                t.assigned_worker_id
                for t in async_server.scheduler.get_all_tasks()
                if t.assigned_worker_id is not None
            }
            return result, workers_used
        finally:
            server.close()
            await server.wait_closed()
            for t in worker_tasks:
                await stop_worker(t)

    result, workers_used = asyncio.run(scenario())
    assert result == {"apple": 3, "banana": 2, "orange": 1}
    assert workers_used == {"worker-1", "worker-2", "worker-3"}


def test_empty_input_produces_empty_result():
    async def scenario():
        server, host, port = await start_master_server()
        worker_task = asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-1"))
        try:
            await async_server.wait_for_workers(1)
            return await run_map_reduce(
                async_server.scheduler,
                async_server.wait_for_tasks,
                "job-1",
                [],
                "WORD_COUNT",
                "SUM",
                num_partitions=4,
            )
        finally:
            await stop_worker(worker_task)
            server.close()
            await server.wait_closed()

    assert asyncio.run(scenario()) == {}


def test_count_reduce_operation():
    """COUNT is degenerate on WORD_COUNT data (every value is already 1,
    so COUNT and SUM agree), but this proves reduce_operation is correctly
    plumbed through the whole pipeline, not hardcoded to SUM."""

    async def scenario():
        server, host, port = await start_master_server()
        worker_task = asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-1"))
        try:
            await async_server.wait_for_workers(1)
            return await run_map_reduce(
                async_server.scheduler,
                async_server.wait_for_tasks,
                "job-1",
                WORDS,
                "WORD_COUNT",
                "COUNT",
                num_partitions=2,
            )
        finally:
            await stop_worker(worker_task)
            server.close()
            await server.wait_closed()

    assert asyncio.run(scenario()) == {"apple": 3, "banana": 2, "orange": 1}


def test_map_failure_excludes_partition_but_job_still_completes():
    """One partition's words include an empty string, which WORD_COUNT
    rejects -- that whole partition fails at the executor, but the job
    still produces a correct (partial) result from the surviving words,
    exactly like jobs.shuffle.shuffle()'s documented tolerance."""

    async def scenario():
        server, host, port = await start_master_server()
        worker_tasks = [
            asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-1")),
            asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-2")),
        ]
        try:
            await async_server.wait_for_workers(2)
            # 2 partitions: ["apple", "banana", ""] (fails) and ["apple", "orange"] (succeeds).
            data = ["apple", "banana", "", "apple", "orange"]
            return await run_map_reduce(
                async_server.scheduler,
                async_server.wait_for_tasks,
                "job-1",
                data,
                "WORD_COUNT",
                "SUM",
                num_partitions=2,
            )
        finally:
            server.close()
            await server.wait_closed()
            for t in worker_tasks:
                await stop_worker(t)

    result = asyncio.run(scenario())
    # Only the surviving partition's words appear; nothing raised.
    assert result == {"apple": 1, "orange": 1}


def test_reduce_failure_propagates_explicitly_not_swallowed():
    """An unsupported reduce_operation makes every Reduce task fail --
    run_map_reduce must let that raise, never return a partial/empty dict
    in its place."""

    async def scenario():
        server, host, port = await start_master_server()
        worker_task = asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-1"))
        try:
            await async_server.wait_for_workers(1)
            with pytest.raises(ValueError):
                await run_map_reduce(
                    async_server.scheduler,
                    async_server.wait_for_tasks,
                    "job-1",
                    WORDS,
                    "WORD_COUNT",
                    "AVERAGE",
                    num_partitions=2,
                )
        finally:
            await stop_worker(worker_task)
            server.close()
            await server.wait_closed()

    asyncio.run(scenario())


def test_job_isolation_two_sequential_jobs_do_not_mix():
    """Two MapReduce jobs run one after another against the SAME scheduler
    and worker pool (state is not reset in between) -- job_id-prefixed
    task_ids must keep them from colliding or mixing results.

    Deliberately sequential, not concurrent: drain_pending_tasks() keeps
    its own local `responses` list per call, but assign_next_pending_task()
    mutates the shared Scheduler -- so two overlapping drain_pending_tasks()
    calls can race and steal each other's tasks, with a task's response
    landing in the wrong caller's returned list. Running two run_map_reduce()
    jobs concurrently via asyncio.gather() hits exactly that: this is a real
    limitation of the current (non-reentrant) dispatch loop, not something
    Phase 8.6 orchestration should paper over. Fixing drain_pending_tasks()
    to be concurrent-caller-safe is out of scope here; sequential execution
    is the supported pattern today.
    """

    async def scenario():
        server, host, port = await start_master_server()
        worker_tasks = [
            asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-1")),
            asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-2")),
        ]
        try:
            await async_server.wait_for_workers(2)

            result_a = await run_map_reduce(
                async_server.scheduler,
                async_server.wait_for_tasks,
                "job-a",
                ["apple", "apple", "banana"],
                "WORD_COUNT",
                "SUM",
                num_partitions=1,
            )
            result_b = await run_map_reduce(
                async_server.scheduler,
                async_server.wait_for_tasks,
                "job-b",
                ["cherry", "cherry", "cherry"],
                "WORD_COUNT",
                "SUM",
                num_partitions=1,
            )

            return result_a, result_b
        finally:
            server.close()
            await server.wait_closed()
            for t in worker_tasks:
                await stop_worker(t)

    result_a, result_b = asyncio.run(scenario())
    assert result_a == {"apple": 2, "banana": 1}
    assert result_b == {"cherry": 3}


def test_deterministic_final_result_across_repeated_runs():
    async def run_once(job_id: str):
        server, host, port = await start_master_server()
        worker_tasks = [
            asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-1")),
            asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-2")),
        ]
        try:
            await async_server.wait_for_workers(2)
            return await run_map_reduce(
                async_server.scheduler,
                async_server.wait_for_tasks,
                job_id,
                WORDS,
                "WORD_COUNT",
                "SUM",
                num_partitions=2,
            )
        finally:
            server.close()
            await server.wait_closed()
            for t in worker_tasks:
                await stop_worker(t)

    async def scenario():
        first = await run_once("job-1")
        rpc_handler.worker_manager.clear()
        async_server.scheduler.clear()
        async_server.connections.clear()
        second = await run_once("job-2")
        return first, second

    first, second = asyncio.run(scenario())
    assert first == second == {"apple": 3, "banana": 2, "orange": 1}


def test_concurrent_map_tasks_during_pipeline():
    """Event-gated proof that the Map phase genuinely dispatches to both
    workers at once, not sequentially, while run_map_reduce is in flight."""

    async def scenario():
        server, host, port = await start_master_server()

        ready = {"worker-1": asyncio.Event(), "worker-2": asyncio.Event()}
        release = {"worker-1": asyncio.Event(), "worker-2": asyncio.Event()}

        async def gated_worker(worker_id: str) -> None:
            reader, writer = await asyncio.open_connection(host, port)
            conn = AsyncConnection(reader, writer)
            await send_request(conn, protocol.PING)
            await send_request(conn, protocol.REGISTER, {"worker_id": worker_id, "host": "127.0.0.1", "port": 6000})

            gated_once = False
            while True:
                try:
                    message = await receive_message(conn)
                except ConnectionError:
                    return
                if message["type"] != protocol.TASK:
                    continue
                if not gated_once:
                    gated_once = True
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

        worker_tasks = [asyncio.create_task(gated_worker(wid)) for wid in ("worker-1", "worker-2")]
        try:
            await async_server.wait_for_workers(2)

            mr_task = asyncio.create_task(
                run_map_reduce(
                    async_server.scheduler,
                    async_server.wait_for_tasks,
                    "job-1",
                    WORDS,
                    "WORD_COUNT",
                    "SUM",
                    num_partitions=2,
                )
            )

            await asyncio.wait_for(ready["worker-1"].wait(), timeout=5)
            await asyncio.wait_for(ready["worker-2"].wait(), timeout=5)

            running = [t for t in async_server.scheduler.get_all_tasks() if t.status == TaskStatus.RUNNING]
            assert len(running) == 2
            assert all(t.task_type == "MAP" for t in running)

            release["worker-1"].set()
            release["worker-2"].set()

            return await mr_task
        finally:
            server.close()
            await server.wait_closed()
            for t in worker_tasks:
                await stop_worker(t)

    assert asyncio.run(scenario()) == {"apple": 3, "banana": 2, "orange": 1}


def test_concurrent_reduce_tasks_during_pipeline():
    """Event-gated proof that the Reduce phase also genuinely overlaps
    across workers -- gates each worker's SECOND task (its first Reduce
    task; the first task overall is its Map partition, let through freely)."""

    async def scenario():
        server, host, port = await start_master_server()

        ready = {"worker-1": asyncio.Event(), "worker-2": asyncio.Event()}
        release = {"worker-1": asyncio.Event(), "worker-2": asyncio.Event()}

        async def gated_worker(worker_id: str) -> None:
            reader, writer = await asyncio.open_connection(host, port)
            conn = AsyncConnection(reader, writer)
            await send_request(conn, protocol.PING)
            await send_request(conn, protocol.REGISTER, {"worker_id": worker_id, "host": "127.0.0.1", "port": 6000})

            task_count = 0
            while True:
                try:
                    message = await receive_message(conn)
                except ConnectionError:
                    return
                if message["type"] != protocol.TASK:
                    continue

                task_count += 1
                if task_count == 2:
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

        worker_tasks = [asyncio.create_task(gated_worker(wid)) for wid in ("worker-1", "worker-2")]
        try:
            await async_server.wait_for_workers(2)

            # 2 partitions -> each worker does exactly 1 Map task, then the
            # 3 distinct words (apple/banana/orange) mean 2 Reduce tasks
            # land in the first round -- one per worker, gated as each
            # worker's 2nd task overall.
            mr_task = asyncio.create_task(
                run_map_reduce(
                    async_server.scheduler,
                    async_server.wait_for_tasks,
                    "job-1",
                    WORDS,
                    "WORD_COUNT",
                    "SUM",
                    num_partitions=2,
                )
            )

            await asyncio.wait_for(ready["worker-1"].wait(), timeout=5)
            await asyncio.wait_for(ready["worker-2"].wait(), timeout=5)

            running_reduce = [
                t
                for t in async_server.scheduler.get_all_tasks()
                if t.status == TaskStatus.RUNNING and t.task_type == "REDUCE"
            ]
            assert len(running_reduce) == 2

            release["worker-1"].set()
            release["worker-2"].set()

            return await mr_task
        finally:
            server.close()
            await server.wait_closed()
            for t in worker_tasks:
                await stop_worker(t)

    assert asyncio.run(scenario()) == {"apple": 3, "banana": 2, "orange": 1}
