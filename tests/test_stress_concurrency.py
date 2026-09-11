"""Phase 12.5.3: stress and concurrency testing -- this project is
fundamentally concurrent (asyncio master, multiple workers, retries,
heartbeats all running at once), so its correctness under real
concurrent load is a first-class property, not something unit tests
alone can prove. Each test here specifically looks for: race conditions,
duplicate task execution, lost results, stuck tasks, deadlocks, leaked
asyncio tasks, executor shutdown problems, semaphore corruption, and
stale worker state -- see each test's docstring for which.
"""

import asyncio
import time

from common import TaskStatus
from jobs import collect_call_result, submit_call
from master import async_server, rpc_handler, scheduler, start_master, stop_master, wait_for_tasks
from worker import DirectBackend, MultiprocessingBackend, register_function, registry, run_worker


async def _start_master():
    server = await start_master("127.0.0.1", 0)
    host, port = server.sockets[0].getsockname()[:2]
    return server, host, port


async def _stop_worker(task: asyncio.Task) -> None:
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):
        pass


def setup_function(_):
    registry.clear()
    rpc_handler.worker_manager.clear()
    scheduler.clear()
    async_server.connections.clear()
    async_server.clear_dispatch_registry()


def teardown_function(_):
    registry.clear()


# -- 100 tasks, 10 workers ------------------------------------------------


def test_100_tasks_across_10_workers_no_duplicates_no_lost_results():
    """The core stress scenario from the phase spec. Checks, precisely:
    every task completes exactly once (no duplicate execution), every
    result is correct (nothing lost or corrupted), and no task is left
    stuck PENDING/ASSIGNED/RUNNING once wait_for_tasks returns."""
    execution_counts: dict[str, int] = {}

    def counted_square(x):
        execution_counts[x] = execution_counts.get(x, 0) + 1
        return x * x

    register_function("counted_square", counted_square)

    async def scenario():
        server, host, port = await _start_master()
        worker_tasks = [
            asyncio.create_task(run_worker(host, port, worker_id=f"worker-{i}", backend=DirectBackend()))
            for i in range(10)
        ]
        try:
            await async_server.wait_for_workers(10)
            start = time.monotonic()
            tasks = [submit_call(scheduler, f"sq-{i}", "counted_square", x=i) for i in range(100)]
            responses = await wait_for_tasks({t.task_id for t in tasks}, timeout=30.0)
            elapsed = time.monotonic() - start
            return responses, elapsed
        finally:
            for t in worker_tasks:
                await _stop_worker(t)
            server.close()
            await server.wait_closed()
            await stop_master()

    responses, elapsed = asyncio.run(scenario())

    assert len(responses) == 100
    results_by_task = {r["payload"]["task_id"]: collect_call_result(r) for r in responses}
    assert results_by_task == {f"sq-{i}": i * i for i in range(100)}

    # No duplicate execution: each x in [0, 100) executed exactly once,
    # even though 10 workers were racing to pick up work concurrently.
    assert execution_counts == {i: 1 for i in range(100)}

    # No stuck tasks left behind.
    for i in range(100):
        assert scheduler.get_task(f"sq-{i}").status == TaskStatus.COMPLETED

    print(f"\n100 tasks / 10 workers: {elapsed:.2f}s ({100 / elapsed:.1f} tasks/sec)")


def test_100_tasks_across_10_workers_with_multiprocessing_backend():
    """The same stress scenario, but with real process-pool execution and
    a max_in_flight cap -- specifically looking for semaphore corruption
    (a stuck or over-released permit) and executor shutdown problems
    under load."""
    register_function("stress_add", lambda x: x + 1)

    async def scenario():
        server, host, port = await _start_master()
        backends = [MultiprocessingBackend(max_workers=2, max_in_flight=4) for _ in range(3)]
        worker_tasks = [
            asyncio.create_task(run_worker(host, port, worker_id=f"worker-{i}", backend=b))
            for i, b in enumerate(backends)
        ]
        try:
            await async_server.wait_for_workers(3)
            tasks = [submit_call(scheduler, f"add-{i}", "stress_add", x=i) for i in range(100)]
            responses = await wait_for_tasks({t.task_id for t in tasks}, timeout=60.0)
            return responses
        finally:
            for t in worker_tasks:
                await _stop_worker(t)
            server.close()
            await server.wait_closed()
            await stop_master()

    responses = asyncio.run(scenario())
    assert len(responses) == 100
    results = {r["payload"]["task_id"]: collect_call_result(r) for r in responses}
    assert results == {f"add-{i}": i + 1 for i in range(100)}


# -- concurrent submitters ------------------------------------------------


def test_many_concurrent_wait_for_tasks_callers_never_cross_results():
    """Several coroutines calling wait_for_tasks() concurrently, each for
    its OWN disjoint set of tasks -- a race condition here would show up
    as one caller getting another's result. Specifically exercises the
    per-task-id future/response registry (master/async_server.py) under
    real concurrent access."""
    register_function("tag", lambda n: f"result-{n}")

    async def scenario():
        server, host, port = await _start_master()
        worker_tasks = [
            asyncio.create_task(run_worker(host, port, worker_id=f"worker-{i}", backend=DirectBackend()))
            for i in range(4)
        ]
        try:
            await async_server.wait_for_workers(4)

            async def submit_and_wait(caller_id: int):
                tasks = [submit_call(scheduler, f"c{caller_id}-{i}", "tag", n=f"{caller_id}-{i}") for i in range(10)]
                responses = await wait_for_tasks({t.task_id for t in tasks})
                return caller_id, {r["payload"]["task_id"]: collect_call_result(r) for r in responses}

            all_results = await asyncio.gather(*(submit_and_wait(c) for c in range(8)))
            return dict(all_results)
        finally:
            for t in worker_tasks:
                await _stop_worker(t)
            server.close()
            await server.wait_closed()
            await stop_master()

    all_results = asyncio.run(scenario())
    assert len(all_results) == 8
    for caller_id, results in all_results.items():
        expected = {f"c{caller_id}-{i}": f"result-{caller_id}-{i}" for i in range(10)}
        assert results == expected, f"caller {caller_id} got contaminated results"


# -- repeated failures/reconnections under concurrent load -----------------


def test_repeated_worker_failures_and_reconnections_under_load():
    """Multiple worker failures interleaved with ongoing task submission
    -- looking for stale worker state (a FAILED worker still considered
    assignable) and lost/duplicated results across several
    failure/retry/reconnect cycles happening close together."""
    register_function("stable_op", lambda x: x * 10)

    async def scenario():
        server, host, port = await _start_master()

        healthy_task = asyncio.create_task(
            run_worker(host, port, worker_id="healthy", backend=DirectBackend())
        )
        flaky_ids = [f"flaky-{i}" for i in range(3)]
        flaky_tasks = {
            wid: asyncio.create_task(run_worker(host, port, worker_id=wid, backend=DirectBackend()))
            for wid in flaky_ids
        }
        try:
            await async_server.wait_for_workers(4)

            # Kill all three "flaky" workers' connections directly, then
            # submit a batch of tasks -- only "healthy" (and later,
            # reconnected flaky workers) can service them.
            for wid in flaky_ids:
                link = async_server.connections.get(wid)
                if link is not None:
                    await link.conn.close()
            await asyncio.sleep(0.1)

            batch1 = [submit_call(scheduler, f"b1-{i}", "stable_op", x=i) for i in range(20)]
            responses1 = await wait_for_tasks({t.task_id for t in batch1}, timeout=20.0)

            # Reconnect the flaky workers (simulating a process restart)
            # and submit a second batch -- now everyone should share it.
            reconnected = {
                wid: asyncio.create_task(run_worker(host, port, worker_id=wid, backend=DirectBackend()))
                for wid in flaky_ids
            }
            await async_server.wait_for_workers(4)

            batch2 = [submit_call(scheduler, f"b2-{i}", "stable_op", x=i) for i in range(20)]
            responses2 = await wait_for_tasks({t.task_id for t in batch2}, timeout=20.0)

            for t in reconnected.values():
                await _stop_worker(t)

            return responses1, responses2
        finally:
            await _stop_worker(healthy_task)
            for t in flaky_tasks.values():
                await _stop_worker(t)
            server.close()
            await server.wait_closed()
            await stop_master()

    responses1, responses2 = asyncio.run(scenario())

    results1 = {r["payload"]["task_id"]: collect_call_result(r) for r in responses1}
    assert results1 == {f"b1-{i}": i * 10 for i in range(20)}

    results2 = {r["payload"]["task_id"]: collect_call_result(r) for r in responses2}
    assert results2 == {f"b2-{i}": i * 10 for i in range(20)}


# -- no leaked asyncio tasks after a stress run ------------------------


def test_no_leaked_asyncio_tasks_after_a_stress_run_and_shutdown():
    """After every worker/master is torn down, no asyncio Task from this
    run should still be alive on the loop -- a leaked heartbeat loop,
    dispatch task, or shutdown watcher would show up here."""
    register_function("noop", lambda: "ok")

    async def scenario():
        tasks_before = {t for t in asyncio.all_tasks() if not t.done()}

        server, host, port = await _start_master()
        worker_tasks = [
            asyncio.create_task(run_worker(host, port, worker_id=f"worker-{i}", backend=DirectBackend()))
            for i in range(5)
        ]
        await async_server.wait_for_workers(5)
        submitted = [submit_call(scheduler, f"n-{i}", "noop") for i in range(30)]
        await wait_for_tasks({t.task_id for t in submitted})

        for t in worker_tasks:
            await _stop_worker(t)
        server.close()
        await server.wait_closed()
        await stop_master()

        # Let anything still unwinding (a done-callback-scheduled cleanup)
        # get a turn before checking.
        await asyncio.sleep(0.05)
        tasks_after = {t for t in asyncio.all_tasks() if not t.done()}

        leaked = tasks_after - tasks_before
        return leaked

    leaked = asyncio.run(scenario())
    assert not leaked, f"leaked asyncio tasks: {[t.get_name() for t in leaked]}"
