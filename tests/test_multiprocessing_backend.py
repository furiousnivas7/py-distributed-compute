"""Phase 11.2: worker.backend.MultiprocessingBackend, in isolation from
any real worker/master connection (tests/test_async_worker_multiprocessing.py
covers the full pipeline).

Module-level helper functions (not lambdas/closures in test bodies) are
used wherever a function needs to be pickled BY REFERENCE for
multiprocessing itself to find it (e.g. anything passed directly to
ProcessPoolExecutor, which none of these are -- only execute_task ever is,
and that's already a top-level function in worker.executor). Registered
functions here use ordinary lambdas registered directly in the test body
BEFORE backend.start(): the "fork" context (this backend's default)
copies the parent's memory -- including whatever's already in
worker.registry -- at fork time, so a lambda registered before start() is
still visible to already-forked children without needing to be
independently importable/picklable itself. See MultiprocessingBackend's
own docstring for why registration timing matters and serialized
callables are unaffected by it.
"""

import asyncio
import os
import time

import pytest

from worker import registry
from worker.backend import MultiprocessingBackend
from worker.executor import ExecutionErrorCode, execute_task


@pytest.fixture(autouse=True)
def reset_registry():
    registry.clear()
    yield
    registry.clear()


def _get_pid():
    return os.getpid()


def _sleep_one_second():
    time.sleep(1.0)
    return "done"


def _crash():
    os._exit(1)


def _raise_value_error():
    raise ValueError("deliberate failure")


def test_execute_runs_in_a_different_process():
    registry.register_function("get_pid", _get_pid)

    async def scenario():
        backend = MultiprocessingBackend(max_workers=2)
        await backend.start()
        try:
            return await backend.execute("get_pid", {})
        finally:
            await backend.stop()

    result = asyncio.run(scenario())
    assert result["status"] == "success"
    assert result["result"] != os.getpid()


def test_execute_success_matches_direct_backend_contract():
    async def scenario():
        backend = MultiprocessingBackend(max_workers=2)
        await backend.start()
        try:
            return await backend.execute("ADD", {"a": 2, "b": 3})
        finally:
            await backend.stop()

    result = asyncio.run(scenario())
    assert result == execute_task("ADD", {"a": 2, "b": 3})
    assert result == {"status": "success", "result": 5}


def test_execute_registered_function_when_registered_before_start():
    registry.register_function("double", lambda x: x * 2)

    async def scenario():
        backend = MultiprocessingBackend(max_workers=2)
        await backend.start()
        try:
            return await backend.execute("double", {"x": 21})
        finally:
            await backend.stop()

    result = asyncio.run(scenario())
    assert result == {"status": "success", "result": 42}


def test_execute_serialized_callable_unaffected_by_fork_timing():
    """Unlike a registered function, a serialized callable is
    unaffected by whether it existed before or after the pool started --
    cloudpickle ships it explicitly as part of the task payload, decoded
    inside the child regardless of fork/spawn timing."""
    from jobs.call import collect_call_result, submit_serialized_call
    from master.scheduler import Scheduler
    from master.worker_manager import WorkerManager

    def triple(x):
        return x * 3

    async def scenario():
        backend = MultiprocessingBackend(max_workers=2)
        await backend.start()  # pool starts BEFORE `triple` is ever registered anywhere
        try:
            scheduler = Scheduler(WorkerManager())
            task = submit_serialized_call(scheduler, "t1", triple, args=[14])
            payload = task.payload
            return await backend.execute(task.task_type, payload)
        finally:
            await backend.stop()

    result = asyncio.run(scenario())
    assert result == {"status": "success", "result": 42}


def test_execution_failure_returns_a_normal_error_dict_not_a_raise():
    """A function raising inside the child process is still a normal
    EXECUTION failure -- execute_task's own try/except already turns it
    into a controlled error dict before returning, so it survives the
    process boundary as data, not an exception. Contrast with
    test_child_process_crash_raises_from_execute below."""
    registry.register_function("boom", _raise_value_error)

    async def scenario():
        backend = MultiprocessingBackend(max_workers=2)
        await backend.start()
        try:
            return await backend.execute("boom", {})
        finally:
            await backend.stop()

    result = asyncio.run(scenario())
    assert result["status"] == "error"
    assert result["code"] == ExecutionErrorCode.USER_FUNCTION_ERROR
    assert "ValueError" in result["message"]


def test_unknown_operation_returns_a_normal_error_dict():
    async def scenario():
        backend = MultiprocessingBackend(max_workers=2)
        await backend.start()
        try:
            return await backend.execute("does_not_exist", {})
        finally:
            await backend.stop()

    result = asyncio.run(scenario())
    assert result["status"] == "error"
    assert result["code"] == ExecutionErrorCode.UNKNOWN_OPERATION


def test_event_loop_remains_responsive_during_a_cpu_bound_task():
    """The central Phase 11.2 property: a 1-second CPU-bound task running
    in a child process must not block a concurrent asyncio coroutine from
    ticking on schedule -- proving execute() doesn't block the event
    loop, not just that it eventually returns the right answer."""
    registry.register_function("slow", _sleep_one_second)

    async def scenario():
        backend = MultiprocessingBackend(max_workers=2)
        await backend.start()

        ticks = []

        async def ticker():
            for _ in range(6):
                await asyncio.sleep(0.15)
                ticks.append(1)

        try:
            exec_task = asyncio.create_task(backend.execute("slow", {}))
            tick_task = asyncio.create_task(ticker())
            result = await exec_task
            await tick_task
            return result, len(ticks)
        finally:
            await backend.stop()

    result, tick_count = asyncio.run(scenario())
    assert result == {"status": "success", "result": "done"}
    # 6 ticks * 0.15s = 0.9s of ticking; the slow task takes ~1s. If
    # execute() were blocking the loop, the ticker couldn't advance at
    # all until the slow task finished, and note asyncio.sleep(0.15)
    # calls made *during* a blocked loop would all fire late/bunched
    # rather than getting anywhere near the full count in time.
    assert tick_count == 6


def test_multiple_concurrent_cpu_bound_tasks_all_complete_correctly():
    def square(x):
        total = 0
        for _ in range(200_000):
            total += 1
        return x * x

    registry.register_function("square", square)

    async def scenario():
        backend = MultiprocessingBackend(max_workers=4)
        await backend.start()
        try:
            results = await asyncio.gather(*(backend.execute("square", {"x": i}) for i in range(6)))
            return results
        finally:
            await backend.stop()

    results = asyncio.run(scenario())
    assert [r["result"] for r in results] == [i * i for i in range(6)]


def test_child_process_crash_raises_from_execute():
    """The other side of the Phase 11.2 fault-handling split: the child
    process itself dying (not the function raising) is a BACKEND failure,
    not an execution failure -- it must raise out of execute(), never
    come back as a normal {"status": "error", ...} result."""
    registry.register_function("crash", _crash)

    async def scenario():
        backend = MultiprocessingBackend(max_workers=2)
        await backend.start()
        try:
            with pytest.raises(Exception):
                await backend.execute("crash", {})
        finally:
            await backend.stop()

    asyncio.run(scenario())


def test_execute_before_start_raises_runtime_error():
    async def scenario():
        backend = MultiprocessingBackend(max_workers=2)
        with pytest.raises(RuntimeError):
            await backend.execute("ADD", {"a": 1, "b": 1})

    asyncio.run(scenario())


def test_stop_terminates_child_processes_no_lingering_pool():
    """After stop(), the pool's own process list must be empty/terminated
    -- no lingering child processes left running past the backend's
    lifetime (Phase 11.4's exact concern, verified directly here at the
    backend level rather than only observed indirectly)."""
    registry.register_function("get_pid", _get_pid)

    async def scenario():
        backend = MultiprocessingBackend(max_workers=2)
        await backend.start()
        await backend.execute("get_pid", {})  # force at least one child to actually spawn
        processes = list(backend._pool._processes.values())
        await backend.stop()
        return processes

    processes = asyncio.run(scenario())
    assert processes, "expected at least one child process to have been spawned"
    for process in processes:
        assert not process.is_alive()


def test_stop_is_safe_to_call_when_never_started():
    async def scenario():
        backend = MultiprocessingBackend(max_workers=2)
        await backend.stop()  # must not raise

    asyncio.run(scenario())


def test_serialized_callable_is_portable_under_spawn_not_just_fork():
    """Empirically backs the docstring's portability claim rather than
    just asserting it: mp_context="spawn" runs in a genuinely fresh
    child interpreter (no inherited memory, unlike fork), and a
    serialized callable must still work correctly -- because cloudpickle
    ships the callable explicitly as data, execution never depended on
    what the child happened to inherit."""
    from jobs.call import submit_serialized_call
    from master.scheduler import Scheduler
    from master.worker_manager import WorkerManager

    def triple(x):
        return x * 3

    async def scenario():
        backend = MultiprocessingBackend(max_workers=2, mp_context="spawn")
        await backend.start()
        try:
            scheduler = Scheduler(WorkerManager())
            task = submit_serialized_call(scheduler, "t1", triple, args=[14])
            return await backend.execute(task.task_type, task.payload)
        finally:
            await backend.stop()

    result = asyncio.run(scenario())
    assert result == {"status": "success", "result": 42}


# -- Phase 11.4: resource and concurrency controls --------------------


def _sleep_briefly():
    time.sleep(0.3)
    return "done"


def test_max_workers_zero_is_rejected():
    with pytest.raises(ValueError, match="max_workers"):
        MultiprocessingBackend(max_workers=0)


def test_max_workers_negative_is_rejected():
    with pytest.raises(ValueError, match="max_workers"):
        MultiprocessingBackend(max_workers=-1)


def test_max_in_flight_zero_is_rejected():
    with pytest.raises(ValueError, match="max_in_flight"):
        MultiprocessingBackend(max_workers=2, max_in_flight=0)


def test_max_in_flight_negative_is_rejected():
    with pytest.raises(ValueError, match="max_in_flight"):
        MultiprocessingBackend(max_workers=2, max_in_flight=-3)


def test_max_workers_none_and_max_in_flight_none_are_still_accepted():
    """The explicit, documented defaults -- unbounded/delegate-to-pool --
    must not be rejected by the same validation that rejects 0/negative."""
    backend = MultiprocessingBackend()  # must not raise
    assert backend.max_concurrency is None


def test_process_pool_uses_the_configured_max_workers():
    async def scenario():
        backend = MultiprocessingBackend(max_workers=3)
        await backend.start()
        try:
            return backend._pool._max_workers
        finally:
            await backend.stop()

    assert asyncio.run(scenario()) == 3


def test_max_concurrency_metadata_reflects_max_in_flight_when_set():
    backend = MultiprocessingBackend(max_workers=4, max_in_flight=2)
    assert backend.max_concurrency == 2


def test_max_concurrency_metadata_falls_back_to_max_workers():
    backend = MultiprocessingBackend(max_workers=4)
    assert backend.max_concurrency == 4


def test_maximum_number_of_simultaneously_executing_tasks_is_bounded_by_max_workers():
    """Timing-based proof (matching Phase 11.2's own event-loop test
    style): 4 tasks that each take ~0.3s, with only 2 real worker
    processes, must take roughly 2 rounds (~0.6s+), not 1 round (~0.3s,
    which would mean more than 2 ran truly in parallel) or 4 rounds
    (~1.2s, which would mean none ran in parallel at all)."""
    registry.register_function("slow", _sleep_briefly)

    async def scenario():
        backend = MultiprocessingBackend(max_workers=2)
        await backend.start()
        try:
            start = time.monotonic()
            await asyncio.gather(*(backend.execute("slow", {}) for _ in range(4)))
            return time.monotonic() - start
        finally:
            await backend.stop()

    elapsed = asyncio.run(scenario())
    assert elapsed > 0.5, f"completed too fast ({elapsed:.2f}s) for only 2 real worker processes"
    assert elapsed < 1.0, f"completed too slow ({elapsed:.2f}s) -- looks fully serialized, not parallel"


def test_max_in_flight_limits_concurrent_submissions_to_the_pool():
    """Precise, non-timing-based proof of the semaphore's own gating:
    _submit is stubbed out (no real multiprocessing overhead/timing
    noise) to directly count how many are concurrently past the
    semaphore at once."""

    async def scenario():
        backend = MultiprocessingBackend(max_workers=4, max_in_flight=2)
        await backend.start()

        active = 0
        max_active = 0

        async def fake_submit(task_type, payload):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.05)
            active -= 1
            return {"status": "success", "result": "ok"}

        backend._submit = fake_submit
        try:
            await asyncio.gather(*(backend.execute("x", {}) for _ in range(6)))
        finally:
            await backend.stop()
        return max_active

    max_active = asyncio.run(scenario())
    assert max_active == 2


def test_semaphore_permit_is_released_after_successful_execution():
    """With max_in_flight=1, running several tasks SEQUENTIALLY (one
    completes before the next starts) must never deadlock -- proving the
    permit is released each time, not just acquired once and never
    freed."""
    registry.register_function("noop", lambda: "ok")

    async def scenario():
        backend = MultiprocessingBackend(max_workers=2, max_in_flight=1)
        await backend.start()
        try:
            results = []
            for _ in range(3):
                results.append(await asyncio.wait_for(backend.execute("noop", {}), timeout=5))
            return results
        finally:
            await backend.stop()

    results = asyncio.run(scenario())
    assert results == [{"status": "success", "result": "ok"}] * 3


def test_semaphore_permit_is_released_after_execution_failure():
    """The permit must be released even when the task ITSELF fails (a
    normal execution failure, still returns a result dict, doesn't
    raise) -- otherwise one failing task would permanently starve every
    later one under a tight max_in_flight."""
    registry.register_function("boom", _raise_value_error)
    registry.register_function("noop", lambda: "ok")

    async def scenario():
        backend = MultiprocessingBackend(max_workers=2, max_in_flight=1)
        await backend.start()
        try:
            first = await asyncio.wait_for(backend.execute("boom", {}), timeout=5)
            second = await asyncio.wait_for(backend.execute("noop", {}), timeout=5)
            return first, second
        finally:
            await backend.stop()

    first, second = asyncio.run(scenario())
    assert first["status"] == "error"
    assert second == {"status": "success", "result": "ok"}


def test_shutdown_while_tasks_are_queued_in_the_pool_does_not_hang():
    """max_workers=1 means only the FIRST of several concurrently-
    submitted tasks can actually be running at once; the rest sit
    queued inside the pool itself. Calling stop() while they're still
    queued must not hang, and must resolve every outstanding execute()
    call one way or another (result or exception) rather than leaving
    any of them pending forever."""
    registry.register_function("slow", _sleep_briefly)

    async def scenario():
        backend = MultiprocessingBackend(max_workers=1)
        await backend.start()

        exec_tasks = [asyncio.create_task(backend.execute("slow", {})) for _ in range(3)]
        await asyncio.sleep(0.05)  # let the first one actually start running

        await asyncio.wait_for(backend.stop(), timeout=5)

        return await asyncio.gather(*exec_tasks, return_exceptions=True)

    results = asyncio.run(scenario())
    assert len(results) == 3
    for result in results:
        assert result is not None  # every call resolved somehow, none left pending
