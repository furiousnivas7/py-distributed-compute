"""Phase 11.5: backend execution metrics (worker.backend.BackendMetrics),
covering both DirectBackend and MultiprocessingBackend through the shared
ExecutionBackend.get_metrics() interface. Purely observational -- nothing
here touches scheduling, retries, or the wire protocol; these tests only
verify the numbers themselves are correct.
"""

import asyncio
import time

import pytest

from worker import registry
from worker.backend import BackendMetrics, DirectBackend, ExecutionBackend, MultiprocessingBackend
from worker.executor import ExecutionErrorCode


@pytest.fixture(autouse=True)
def reset_registry():
    registry.clear()
    yield
    registry.clear()


def _sleep_briefly():
    time.sleep(0.2)
    return "done"


def _raise_value_error():
    raise ValueError("deliberate failure")


def _crash():
    import os

    os._exit(1)


# -- 1. initial metrics are zero --------------------------------------


def test_direct_backend_initial_metrics_are_zero():
    metrics = DirectBackend().get_metrics()
    assert metrics == BackendMetrics()


def test_multiprocessing_backend_initial_metrics_are_zero():
    metrics = MultiprocessingBackend().get_metrics()
    assert metrics == BackendMetrics()


def test_base_execution_backend_default_metrics_are_zero():
    class Minimal(ExecutionBackend):
        async def execute(self, task_type: str, payload: dict) -> dict:
            return {"status": "success", "result": None}

    assert Minimal().get_metrics() == BackendMetrics()


# -- 2. successful execution increments submitted and completed -------


def test_direct_backend_successful_execution_increments_submitted_and_completed():
    async def scenario():
        backend = DirectBackend()
        await backend.execute("ADD", {"a": 1, "b": 1})
        return backend.get_metrics()

    metrics = asyncio.run(scenario())
    assert metrics.submitted == 1
    assert metrics.completed == 1
    assert metrics.failed == 0


def test_multiprocessing_backend_successful_execution_increments_submitted_and_completed():
    async def scenario():
        backend = MultiprocessingBackend(max_workers=2)
        await backend.start()
        try:
            await backend.execute("ADD", {"a": 1, "b": 1})
        finally:
            await backend.stop()
        return backend.get_metrics()

    metrics = asyncio.run(scenario())
    assert metrics.submitted == 1
    assert metrics.completed == 1
    assert metrics.failed == 0


# -- 3. structured execution failure increments failed -----------------


def test_direct_backend_structured_failure_increments_failed_not_backend_errors():
    async def scenario():
        backend = DirectBackend()
        result = await backend.execute("does_not_exist", {})
        return result, backend.get_metrics()

    result, metrics = asyncio.run(scenario())
    assert result["code"] == ExecutionErrorCode.UNKNOWN_OPERATION
    assert metrics.failed == 1
    assert metrics.completed == 0
    assert metrics.backend_errors == 0


def test_multiprocessing_backend_structured_failure_increments_failed_not_backend_errors():
    registry.register_function("boom", _raise_value_error)

    async def scenario():
        backend = MultiprocessingBackend(max_workers=2)
        await backend.start()
        try:
            result = await backend.execute("boom", {})
        finally:
            await backend.stop()
        return result, backend.get_metrics()

    result, metrics = asyncio.run(scenario())
    assert result["status"] == "error"
    assert metrics.failed == 1
    assert metrics.completed == 0
    assert metrics.backend_errors == 0


# -- 4. backend exception increments backend_errors ---------------------


def test_multiprocessing_backend_child_process_crash_increments_backend_errors():
    registry.register_function("crash", _crash)

    async def scenario():
        backend = MultiprocessingBackend(max_workers=2)
        await backend.start()
        try:
            with pytest.raises(Exception):
                await backend.execute("crash", {})
        finally:
            await backend.stop()
        return backend.get_metrics()

    metrics = asyncio.run(scenario())
    assert metrics.backend_errors == 1
    assert metrics.failed == 0
    assert metrics.completed == 0


def test_multiprocessing_backend_execute_after_stop_does_not_touch_metrics():
    """A call rejected before the backend even accepts it (never started,
    or already stopped) is a precondition violation, not a submitted-then-
    failed execution -- it never reaches the metrics-tracking code at all
    (see execute()'s self._pool is None check, which runs before
    `submitted` is incremented)."""

    async def scenario():
        backend = MultiprocessingBackend(max_workers=2)
        await backend.start()
        await backend.stop()
        with pytest.raises(RuntimeError):
            await backend.execute("ADD", {"a": 1, "b": 1})
        return backend.get_metrics()

    metrics = asyncio.run(scenario())
    assert metrics == BackendMetrics()


# -- 5. running count returns to zero after completion -------------------


def test_running_count_returns_to_zero_after_completion():
    registry.register_function("noop", lambda: "ok")

    async def scenario():
        backend = MultiprocessingBackend(max_workers=2)
        await backend.start()
        try:
            await backend.execute("noop", {})
        finally:
            await backend.stop()
        return backend.get_metrics()

    metrics = asyncio.run(scenario())
    assert metrics.running == 0


def test_running_count_returns_to_zero_after_a_backend_failure():
    registry.register_function("crash", _crash)

    async def scenario():
        backend = MultiprocessingBackend(max_workers=2)
        await backend.start()
        try:
            with pytest.raises(Exception):
                await backend.execute("crash", {})
        finally:
            await backend.stop()
        return backend.get_metrics()

    metrics = asyncio.run(scenario())
    assert metrics.running == 0


def test_running_count_is_nonzero_while_a_task_is_actually_executing():
    registry.register_function("slow", _sleep_briefly)

    async def scenario():
        backend = MultiprocessingBackend(max_workers=2)
        await backend.start()
        try:
            exec_task = asyncio.create_task(backend.execute("slow", {}))
            await asyncio.sleep(0.05)
            mid_flight_running = backend.get_metrics().running
            await exec_task
        finally:
            await backend.stop()
        return mid_flight_running, backend.get_metrics().running

    mid_flight_running, final_running = asyncio.run(scenario())
    assert mid_flight_running == 1
    assert final_running == 0


# -- 6. execution duration is recorded -----------------------------------


def test_execution_duration_is_recorded():
    registry.register_function("slow", _sleep_briefly)

    async def scenario():
        backend = MultiprocessingBackend(max_workers=2)
        await backend.start()
        try:
            await backend.execute("slow", {})
        finally:
            await backend.stop()
        return backend.get_metrics()

    metrics = asyncio.run(scenario())
    # _sleep_briefly takes ~0.2s -- generous lower bound to avoid timing
    # flakiness while still proving a real duration was recorded, not 0.
    assert metrics.total_execution_time > 0.1


# -- 7. semaphore wait duration is recorded ------------------------------


def test_semaphore_wait_duration_is_recorded():
    registry.register_function("slow", _sleep_briefly)

    async def scenario():
        backend = MultiprocessingBackend(max_workers=1, max_in_flight=1)
        await backend.start()
        try:
            # Two tasks, one in-flight slot -- the second must wait
            # roughly as long as the first takes to run.
            await asyncio.gather(
                backend.execute("slow", {}),
                backend.execute("slow", {}),
            )
        finally:
            await backend.stop()
        return backend.get_metrics()

    metrics = asyncio.run(scenario())
    assert metrics.total_wait_time > 0.1


def test_no_semaphore_wait_time_when_max_in_flight_is_unset():
    registry.register_function("noop", lambda: "ok")

    async def scenario():
        backend = MultiprocessingBackend(max_workers=2)
        await backend.start()
        try:
            await backend.execute("noop", {})
        finally:
            await backend.stop()
        return backend.get_metrics()

    metrics = asyncio.run(scenario())
    assert metrics.total_wait_time == 0.0


# -- 8. cancelled execution does not leave `running` permanently incremented -


def test_cancelled_execution_does_not_leave_running_permanently_incremented():
    registry.register_function("slow", _sleep_briefly)

    async def scenario():
        backend = MultiprocessingBackend(max_workers=2)
        await backend.start()
        try:
            exec_task = asyncio.create_task(backend.execute("slow", {}))
            await asyncio.sleep(0.02)
            exec_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await exec_task
        finally:
            await backend.stop()
        return backend.get_metrics()

    metrics = asyncio.run(scenario())
    assert metrics.running == 0
    # Cancellation is neither a structured failure nor a backend error --
    # it's its own category (see ExecutionBackend.execute's Exception-vs-
    # BaseException handling in worker/backend.py).
    assert metrics.completed == 0
    assert metrics.failed == 0
    assert metrics.backend_errors == 0


def test_cancelled_while_waiting_for_in_flight_capacity_does_not_leave_running_incremented():
    registry.register_function("slow", _sleep_briefly)

    async def scenario():
        backend = MultiprocessingBackend(max_workers=1, max_in_flight=1)
        await backend.start()
        try:
            holder = asyncio.create_task(backend.execute("slow", {}))
            await asyncio.sleep(0.02)  # let holder acquire the permit
            waiter = asyncio.create_task(backend.execute("slow", {}))
            await asyncio.sleep(0.02)  # waiter is now blocked on the semaphore
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
            await holder
        finally:
            await backend.stop()
        return backend.get_metrics()

    metrics = asyncio.run(scenario())
    assert metrics.running == 0
    assert metrics.completed == 1  # the holder still finished normally
    assert metrics.backend_errors == 0


# -- 9. metrics snapshots cannot mutate internal counters -----------------


def test_direct_backend_cancellation_before_execute_starts_leaves_running_at_zero():
    """DirectBackend's execute() body has no `await` point -- once started
    it always runs to completion uninterrupted (asyncio can only actually
    deliver a cancellation at an await point) -- so the only cancellation
    DirectBackend can ever really observe is one delivered BEFORE the
    coroutine starts running at all, in which case its body (and the
    `running` increment inside it) never executes."""

    async def scenario():
        backend = DirectBackend()
        exec_task = asyncio.create_task(backend.execute("ADD", {"a": 1, "b": 1}))
        exec_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await exec_task
        return backend.get_metrics()

    metrics = asyncio.run(scenario())
    assert metrics.running == 0
    assert metrics.submitted == 0


def test_metrics_snapshot_cannot_mutate_internal_counters():
    async def scenario():
        backend = DirectBackend()
        await backend.execute("ADD", {"a": 1, "b": 1})
        snapshot = backend.get_metrics()
        snapshot.submitted = 999
        snapshot.completed = 999
        return backend.get_metrics()

    fresh = asyncio.run(scenario())
    assert fresh.submitted == 1
    assert fresh.completed == 1


# -- 10. direct and multiprocessing backends expose the same interface ----


def test_direct_and_multiprocessing_backends_expose_the_same_metrics_interface():
    direct_metrics = DirectBackend().get_metrics()
    mp_metrics = MultiprocessingBackend().get_metrics()
    assert type(direct_metrics) is type(mp_metrics) is BackendMetrics


# -- 11. metrics remain correct after backend restart ----------------------


def test_metrics_reset_on_restart_documented_contract():
    """Documented contract (Phase 11.5): start() gives a fresh metrics
    snapshot, matching the pool/semaphore reset -- a restarted backend's
    metrics reflect only its NEW lifetime, not numbers carried over from
    before stop()."""
    registry.register_function("noop", lambda: "ok")

    async def scenario():
        backend = MultiprocessingBackend(max_workers=2)
        await backend.start()
        await backend.execute("noop", {})
        await backend.stop()
        before_restart = backend.get_metrics()

        await backend.start()
        after_restart_before_execute = backend.get_metrics()
        await backend.execute("noop", {})
        after_restart_after_execute = backend.get_metrics()
        await backend.stop()
        return before_restart, after_restart_before_execute, after_restart_after_execute

    before_restart, fresh, after = asyncio.run(scenario())
    assert before_restart.submitted == 1
    assert fresh == BackendMetrics()
    assert after.submitted == 1
    assert after.completed == 1


# -- 12. concurrent executions update counters safely -----------------------


def test_concurrent_executions_update_counters_safely():
    def square(x):
        total = 0
        for _ in range(50_000):
            total += 1
        return x * x

    registry.register_function("square", square)

    async def scenario():
        backend = MultiprocessingBackend(max_workers=4)
        await backend.start()
        try:
            results = await asyncio.gather(*(backend.execute("square", {"x": i}) for i in range(10)))
        finally:
            await backend.stop()
        return results, backend.get_metrics()

    results, metrics = asyncio.run(scenario())
    assert [r["result"] for r in results] == [i * i for i in range(10)]
    assert metrics.submitted == 10
    assert metrics.completed == 10
    assert metrics.failed == 0
    assert metrics.running == 0
