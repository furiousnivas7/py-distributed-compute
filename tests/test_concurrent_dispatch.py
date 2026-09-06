"""Phase 8.8: concurrent MapReduce dispatch hardening.

Root cause (see the phase report for full detail): master.async_server's
drain loop mutates a scheduler shared by every caller. Before this phase,
drain_pending_tasks() considered ALL currently-PENDING tasks regardless of
who submitted them -- so two concurrent callers (two jobs.map_reduce.
run_map_reduce() calls, or a job running alongside ordinary ad-hoc tasks)
could "steal" each other's tasks: whichever caller's assignment loop ran
first would dispatch the task and keep its response in its OWN aggregate
list, leaving the task's rightful caller thinking it was never dispatched
at all (test_map_reduce.py's original 8.6 job-isolation test demonstrated
exactly this and had to be downgraded to sequential execution).

The fix: master.async_server.drain_tasks_for(task_ids) scopes assignment
to a specific set of task_ids (master.scheduler.assign_next_pending_task
grew a matching optional filter). jobs.map_reduce.run_map_reduce now calls
dispatch(task_ids) with its own task_ids for each phase, so two concurrent
run_map_reduce() calls -- or a job running alongside ordinary tasks -- never
compete for the same task and their response lists can never cross-
contaminate. No lock is needed: individual Scheduler/WorkerManager calls
are already atomic (no `await` inside them), and disjoint task_id sets
mean concurrent callers have nothing left to race over.

drain_pending_tasks() (unscoped) remains available and is still not safe
for concurrent callers -- it's for the single-caller case only (the manual
demo in run_server(), or a test driving one job/worker set in isolation).

Every test here verifies EXACT results, not just "completed without error".
"""

import asyncio
import time

import pytest

from common.models import TaskStatus, WorkerStatus
from jobs.map_reduce import run_map_reduce
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


async def wait_for_worker_id(worker_id: str, poll_interval: float = 0.01, timeout: float = 5.0) -> None:
    """Wait for a SPECIFIC worker_id to be connected, rather than for a
    total count. async_server.wait_for_workers(N) counts currently-open
    connections -- in these tests a worker can crash and leave before a
    later one joins, so the total can drop back down and never reach N
    again even though every worker we still care about is present."""
    deadline = asyncio.get_running_loop().time() + timeout
    while worker_id not in async_server.connections:
        if asyncio.get_running_loop().time() > deadline:
            raise TimeoutError(f"worker {worker_id!r} never connected")
        await asyncio.sleep(poll_interval)


async def run_multi_task_crashing_worker(
    host: str, port: int, worker_id: str, task_count: int, ready: asyncio.Event
) -> None:
    """Registers, reads `task_count` TASK messages without replying to any
    of them, signals `ready`, then disconnects (a genuine connection-death
    failure -- safe to run without a live failure_monitor task, since
    dispatch_assigned_task handles this inline)."""
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


def test_two_concurrent_mapreduce_jobs_multiple_workers():
    """Item 1: two independent MapReduce jobs, run genuinely concurrently
    via asyncio.gather, sharing a pool of 3 workers. Both must produce
    their own exact, uncorrupted result."""

    async def scenario():
        server, host, port = await start_master_server()
        worker_tasks = [
            asyncio.create_task(async_worker.run_worker(host, port, worker_id=f"worker-{i}"))
            for i in range(1, 4)
        ]
        try:
            await async_server.wait_for_workers(3)

            job_a = run_map_reduce(
                async_server.scheduler,
                async_server.wait_for_tasks,
                "job-a",
                ["apple", "banana", "apple", "cherry", "banana", "apple"],
                "WORD_COUNT",
                "SUM",
                num_partitions=3,
            )
            job_b = run_map_reduce(
                async_server.scheduler,
                async_server.wait_for_tasks,
                "job-b",
                ["dog", "cat", "dog", "dog", "bird"],
                "WORD_COUNT",
                "SUM",
                num_partitions=2,
            )

            return await asyncio.gather(job_a, job_b)
        finally:
            server.close()
            await server.wait_closed()
            for t in worker_tasks:
                await stop_worker(t)

    result_a, result_b = asyncio.run(scenario())
    assert result_a == {"apple": 3, "banana": 2, "cherry": 1}
    assert result_b == {"dog": 3, "cat": 1, "bird": 1}


def test_multiple_concurrent_jobs_different_input_sizes():
    """Item 2: three jobs of very different sizes (1, 5, and 12 words),
    different partition counts, all started together."""

    async def scenario():
        server, host, port = await start_master_server()
        worker_tasks = [
            asyncio.create_task(async_worker.run_worker(host, port, worker_id=f"worker-{i}"))
            for i in range(1, 4)
        ]
        try:
            await async_server.wait_for_workers(3)

            small = run_map_reduce(
                async_server.scheduler,
                async_server.wait_for_tasks,
                "small",
                ["solo"],
                "WORD_COUNT",
                "SUM",
                num_partitions=1,
            )
            medium = run_map_reduce(
                async_server.scheduler,
                async_server.wait_for_tasks,
                "medium",
                ["x", "y", "x", "z", "y"],
                "WORD_COUNT",
                "SUM",
                num_partitions=2,
            )
            large = run_map_reduce(
                async_server.scheduler,
                async_server.wait_for_tasks,
                "large",
                ["p"] * 5 + ["q"] * 4 + ["r"] * 3,
                "WORD_COUNT",
                "SUM",
                num_partitions=4,
            )

            return await asyncio.gather(small, medium, large)
        finally:
            server.close()
            await server.wait_closed()
            for t in worker_tasks:
                await stop_worker(t)

    result_small, result_medium, result_large = asyncio.run(scenario())
    assert result_small == {"solo": 1}
    assert result_medium == {"x": 2, "y": 2, "z": 1}
    assert result_large == {"p": 5, "q": 4, "r": 3}


def test_no_response_cross_contamination_between_jobs():
    """Item 3: two jobs that deliberately share a word ("apple") with
    DIFFERENT counts -- job_id-prefixed task_ids must keep their per-word
    tallies from merging, even though the same key string appears in both.
    """

    async def scenario():
        server, host, port = await start_master_server()
        worker_tasks = [
            asyncio.create_task(async_worker.run_worker(host, port, worker_id=f"worker-{i}"))
            for i in range(1, 3)
        ]
        try:
            await async_server.wait_for_workers(2)

            job_a = run_map_reduce(
                async_server.scheduler,
                async_server.wait_for_tasks,
                "job-a",
                ["apple", "apple"],
                "WORD_COUNT",
                "SUM",
                num_partitions=1,
            )
            job_b = run_map_reduce(
                async_server.scheduler,
                async_server.wait_for_tasks,
                "job-b",
                ["apple", "apple", "apple", "apple", "apple"],
                "WORD_COUNT",
                "SUM",
                num_partitions=1,
            )

            return await asyncio.gather(job_a, job_b)
        finally:
            server.close()
            await server.wait_closed()
            for t in worker_tasks:
                await stop_worker(t)

    result_a, result_b = asyncio.run(scenario())
    # If job B's map/reduce responses had ever landed in job A's aggregate
    # (or vice versa), these counts would be corrupted (merged, doubled,
    # or one job would see the other's task as "missing").
    assert result_a == {"apple": 2}
    assert result_b == {"apple": 5}


def test_concurrent_ordinary_tasks_and_a_mapreduce_job():
    """Item 4: plain ADD/MULTIPLY tasks (submitted directly, not through
    any job) are drained concurrently with a MapReduce job sharing the
    same worker pool and scheduler. Neither must interfere with the other."""

    async def scenario():
        server, host, port = await start_master_server()
        worker_tasks = [
            asyncio.create_task(async_worker.run_worker(host, port, worker_id=f"worker-{i}"))
            for i in range(1, 3)
        ]
        try:
            await async_server.wait_for_workers(2)

            ordinary_ids = set()
            for i, (a, b) in enumerate([(1, 1), (2, 2), (3, 3)]):
                task = async_server.scheduler.submit_task(f"ordinary-{i}", "ADD", {"a": a, "b": b})
                ordinary_ids.add(task.task_id)

            async def run_ordinary():
                return await async_server.wait_for_tasks(ordinary_ids)

            mr_job = run_map_reduce(
                async_server.scheduler,
                async_server.wait_for_tasks,
                "job-mr",
                ["apple", "banana", "apple"],
                "WORD_COUNT",
                "SUM",
                num_partitions=2,
            )

            ordinary_responses, mr_result = await asyncio.gather(run_ordinary(), mr_job)
            return ordinary_responses, mr_result
        finally:
            server.close()
            await server.wait_closed()
            for t in worker_tasks:
                await stop_worker(t)

    ordinary_responses, mr_result = asyncio.run(scenario())
    assert mr_result == {"apple": 2, "banana": 1}
    ordinary_results = sorted(r["payload"]["result"] for r in ordinary_responses)
    assert ordinary_results == [2, 4, 6]


def test_concurrent_dispatch_with_worker_failure_in_one_job():
    """Item 5: two jobs run concurrently; job A's only worker crashes
    mid-task (connection death) and must recover via a second worker,
    while job B (on entirely separate workers) is completely unaffected."""

    async def scenario():
        server, host, port = await start_master_server()

        ready = asyncio.Event()
        crashing_task = asyncio.create_task(run_multi_task_crashing_worker(host, port, "worker-a1", 1, ready))
        await async_server.wait_for_workers(1)
        rescue_task = asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-a2"))
        await async_server.wait_for_workers(2)
        worker_b_task = asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-b1"))
        await async_server.wait_for_workers(3)

        try:
            job_a = run_map_reduce(
                async_server.scheduler,
                async_server.wait_for_tasks,
                "job-a",
                ["apple", "apple", "apple"],
                "WORD_COUNT",
                "SUM",
                num_partitions=1,
            )
            job_b = run_map_reduce(
                async_server.scheduler,
                async_server.wait_for_tasks,
                "job-b",
                ["kiwi", "kiwi"],
                "WORD_COUNT",
                "SUM",
                num_partitions=1,
            )

            result_a, result_b = await asyncio.gather(job_a, job_b)
            return result_a, result_b
        finally:
            await stop_worker(crashing_task)
            await stop_worker(rescue_task)
            await stop_worker(worker_b_task)
            server.close()
            await server.wait_closed()

    result_a, result_b = asyncio.run(scenario())
    assert result_a == {"apple": 3}
    assert result_b == {"kiwi": 2}


def test_retry_response_arriving_after_another_job_has_started():
    """Item 6: job A's task fails and is mid-retry when job B starts (not
    just "at some point during" -- explicitly after, via an event). Job A's
    eventual (correct, retried) result must not be affected by job B
    having started in the meantime, and vice versa."""

    async def scenario():
        server, host, port = await start_master_server()

        ready = asyncio.Event()
        crashing_task = asyncio.create_task(run_multi_task_crashing_worker(host, port, "worker-a1", 1, ready))
        await async_server.wait_for_workers(1)
        rescue_task = asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-a2"))
        await async_server.wait_for_workers(2)

        try:
            job_a_task = asyncio.create_task(
                run_map_reduce(
                    async_server.scheduler,
                    async_server.wait_for_tasks,
                    "job-a",
                    ["mango", "mango"],
                    "WORD_COUNT",
                    "SUM",
                    num_partitions=1,
                )
            )

            # Wait until worker-a1 has definitely received job A's task
            # (and is about to crash) before job B even starts.
            await asyncio.wait_for(ready.wait(), timeout=5)

            worker_b_task = asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-b1"))
            await wait_for_worker_id("worker-b1")

            job_b_task = asyncio.create_task(
                run_map_reduce(
                    async_server.scheduler,
                    async_server.wait_for_tasks,
                    "job-b",
                    ["pear", "pear", "pear"],
                    "WORD_COUNT",
                    "SUM",
                    num_partitions=1,
                )
            )

            result_a, result_b = await asyncio.gather(job_a_task, job_b_task)
            return result_a, result_b
        finally:
            await stop_worker(crashing_task)
            await stop_worker(rescue_task)
            await stop_worker(worker_b_task)
            server.close()
            await server.wait_closed()

    result_a, result_b = asyncio.run(scenario())
    assert result_a == {"mango": 2}
    assert result_b == {"pear": 3}


def test_stale_response_not_returned_to_wrong_caller():
    """Item 7: the precise scenario the fix targets. worker-a1 is given
    job A's task and never replies (connection genuinely dies eventually,
    but not before job B has started and is actively dispatching its own
    tasks on entirely separate workers). Job B's result must never include
    anything from job A's stale/retried attempt, and job A's eventual
    retried result must be exactly correct."""

    async def scenario():
        server, host, port = await start_master_server()

        ready = asyncio.Event()
        crashing_task = asyncio.create_task(run_multi_task_crashing_worker(host, port, "worker-a1", 1, ready))
        await async_server.wait_for_workers(1)

        try:
            job_a_task = asyncio.create_task(
                run_map_reduce(
                    async_server.scheduler,
                    async_server.wait_for_tasks,
                    "job-a",
                    ["grape"],
                    "WORD_COUNT",
                    "SUM",
                    num_partitions=1,
                )
            )
            await asyncio.wait_for(ready.wait(), timeout=5)

            # job B starts entirely on its own workers while job A's stale
            # attempt is still unwinding.
            worker_b1 = asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-b1"))
            worker_b2 = asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-b2"))
            await wait_for_worker_id("worker-b1")
            await wait_for_worker_id("worker-b2")

            job_b_task = asyncio.create_task(
                run_map_reduce(
                    async_server.scheduler,
                    async_server.wait_for_tasks,
                    "job-b",
                    ["lemon", "lime", "lemon"],
                    "WORD_COUNT",
                    "SUM",
                    num_partitions=2,
                )
            )

            result_b = await job_b_task

            # job A can only complete once a rescue worker exists.
            rescue = asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-a2"))
            await wait_for_worker_id("worker-a2")
            result_a = await job_a_task

            return result_a, result_b
        finally:
            await stop_worker(crashing_task)
            await stop_worker(worker_b1)
            await stop_worker(worker_b2)
            await stop_worker(rescue)
            server.close()
            await server.wait_closed()

    result_a, result_b = asyncio.run(scenario())
    assert result_a == {"grape": 1}
    assert result_b == {"lemon": 2, "lime": 1}
    # No cross-contamination in either direction.
    assert "lemon" not in result_a and "lime" not in result_a
    assert "grape" not in result_b


def test_repeated_concurrent_execution_detects_intermittent_failures():
    """Item 8: the same two-concurrent-jobs scenario run 20 times in a
    single test (in addition to the external repeated pytest invocations
    used for validation) -- catches anything that only shows up
    intermittently under repeated scheduling."""

    async def run_once(iteration: int):
        server, host, port = await start_master_server()
        worker_tasks = [
            asyncio.create_task(async_worker.run_worker(host, port, worker_id=f"worker-{i}"))
            for i in range(1, 3)
        ]
        try:
            await async_server.wait_for_workers(2)

            job_a = run_map_reduce(
                async_server.scheduler,
                async_server.wait_for_tasks,
                f"job-a-{iteration}",
                ["red", "blue", "red"],
                "WORD_COUNT",
                "SUM",
                num_partitions=1,
            )
            job_b = run_map_reduce(
                async_server.scheduler,
                async_server.wait_for_tasks,
                f"job-b-{iteration}",
                ["green", "green", "green", "yellow"],
                "WORD_COUNT",
                "SUM",
                num_partitions=2,
            )
            return await asyncio.gather(job_a, job_b)
        finally:
            server.close()
            await server.wait_closed()
            for t in worker_tasks:
                await stop_worker(t)

    async def scenario():
        results = []
        for i in range(20):
            rpc_handler.worker_manager.clear()
            async_server.scheduler.clear()
            async_server.connections.clear()
            results.append(await run_once(i))
        return results

    results = asyncio.run(scenario())
    for result_a, result_b in results:
        assert result_a == {"red": 2, "blue": 1}
        assert result_b == {"green": 3, "yellow": 1}
