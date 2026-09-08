"""Phase 11.2: MultiprocessingBackend proven through the REAL async_worker/
master pipeline, not just in isolation (tests/test_multiprocessing_backend.py)
-- a real TCP connection, real heartbeats, real dispatch_assigned_task,
real Scheduler retry semantics. This is what actually proves the
end-to-end property the phase is about:

    CPU-bound task -> child process -> result/exception -> ExecutionBackend
    -> the existing execution-result contract

while the worker's own async control/heartbeat machinery stays responsive,
AND that a child process death is distinguished from a normal execution
failure exactly the way Phase 10's retry semantics require:

    user function raises  -> USER_FUNCTION_ERROR -> worker stays IDLE,
                              task FAILED, no retry (unchanged from Phase 10)
    child process dies    -> connection dies -> worker marked FAILED,
                              task requeued -- the EXISTING connection-death
                              retry path (Phase 8/9), not a new one
"""

import asyncio
import time

import pytest

from common.models import TaskStatus, WorkerStatus
from master import async_server, rpc_handler
from worker import async_worker, registry
from worker.backend import MultiprocessingBackend
from worker.executor import ExecutionErrorCode


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
    except (asyncio.CancelledError, Exception):
        # A worker whose backend.execute() raised (the crash scenario
        # below) already completed run_worker() with that exception
        # stored on the task -- cancel() on an already-done task is a
        # no-op, and awaiting it re-raises that stored exception rather
        # than CancelledError. Either outcome means the worker is no
        # longer running, which is all cleanup needs here.
        pass


def _slow_double(x):
    time.sleep(0.5)
    return x * 2


def _crash():
    import os

    os._exit(1)


def test_real_worker_with_multiprocessing_backend_executes_registered_function():
    registry.register_function("double", lambda x: x * 2)

    async def scenario():
        server, host, port = await start_master_server()
        backend = MultiprocessingBackend(max_workers=2)
        worker_task = asyncio.create_task(
            async_worker.run_worker(host, port, worker_id="worker-1", backend=backend)
        )
        await async_server.wait_for_workers(1)

        try:
            task = async_server.scheduler.submit_task("t1", "double", {"x": 21})
            [response] = await async_server.wait_for_tasks({task.task_id})
            return response
        finally:
            await stop_worker(worker_task)
            server.close()
            await server.wait_closed()

    response = asyncio.run(scenario())
    assert response["payload"]["result"] == 42


def test_heartbeats_continue_arriving_during_a_real_cpu_bound_dispatch():
    """The full-system version of the event-loop-responsiveness property:
    a worker executing a slow CPU-bound task via MultiprocessingBackend
    must still send HEARTBEAT on schedule -- proving the worker's own
    async loop (not just the backend in isolation) stays responsive."""
    registry.register_function("slow_double", _slow_double)

    async def scenario():
        server, host, port = await start_master_server()
        backend = MultiprocessingBackend(max_workers=2)
        worker_task = asyncio.create_task(
            async_worker.run_worker(
                host, port, worker_id="worker-1", backend=backend, heartbeat_interval=0.1
            )
        )
        await async_server.wait_for_workers(1)

        try:
            before = async_server.worker_manager.get_worker("worker-1").last_heartbeat
            task = async_server.scheduler.submit_task("t1", "slow_double", {"x": 10})
            [response] = await async_server.wait_for_tasks({task.task_id})
            after = async_server.worker_manager.get_worker("worker-1").last_heartbeat
            return response, before, after
        finally:
            await stop_worker(worker_task)
            server.close()
            await server.wait_closed()

    response, before, after = asyncio.run(scenario())
    assert response["payload"]["result"] == 20
    # If the worker's event loop had been blocked for the ~0.5s the slow
    # task took, no heartbeat could have arrived in the meantime --
    # `after` must have advanced well past `before`.
    assert after > before


def test_execution_failure_via_multiprocessing_backend_does_not_retry():
    """Contrast case: a NORMAL execution failure (the function raises)
    through MultiprocessingBackend must behave EXACTLY like DirectBackend
    always has -- worker stays IDLE, task FAILED, no retry. The process
    boundary must not change this policy."""

    def always_fails():
        raise ValueError("deliberate")

    registry.register_function("always_fails", always_fails)

    async def scenario():
        server, host, port = await start_master_server()
        backend = MultiprocessingBackend(max_workers=2)
        worker_task = asyncio.create_task(
            async_worker.run_worker(host, port, worker_id="worker-1", backend=backend)
        )
        await async_server.wait_for_workers(1)

        try:
            task = async_server.scheduler.submit_task("t1", "always_fails", {})
            [response] = await async_server.wait_for_tasks({task.task_id})
            return response, task
        finally:
            await stop_worker(worker_task)
            server.close()
            await server.wait_closed()

    response, task = asyncio.run(scenario())
    assert response["payload"]["code"] == ExecutionErrorCode.USER_FUNCTION_ERROR
    assert task.status == TaskStatus.FAILED
    assert task.attempt == 1  # never retried
    assert async_server.worker_manager.get_worker("worker-1").status == WorkerStatus.IDLE


def test_child_process_death_is_treated_as_worker_failure_and_the_task_is_requeued():
    """The other half of the contrast: a child process dying (not the
    function raising) must NOT come back as a normal TASK_RESULT at all
    -- it tears down the worker's connection, which the master already
    treats exactly like any other worker crash: FAILED status, task
    requeued for retry on someone else. This is the "Scheduler may
    retry" path from a genuinely different failure category than a
    deterministic execution error, achieved with ZERO changes to the
    wire protocol or Scheduler -- see worker/backend.py's
    ExecutionBackend.execute docstring."""
    registry.register_function("crash", _crash)

    async def scenario():
        server, host, port = await start_master_server()
        backend = MultiprocessingBackend(max_workers=2)
        worker_task = asyncio.create_task(
            async_worker.run_worker(host, port, worker_id="worker-1", backend=backend)
        )
        await async_server.wait_for_workers(1)

        try:
            task = async_server.scheduler.submit_task("t1", "crash", {})
            assigned = async_server.scheduler.assign_next_pending_task()
            assert assigned.assigned_worker_id == "worker-1"
            response = await async_server.dispatch_assigned_task(task)
            return response, task
        finally:
            await stop_worker(worker_task)
            server.close()
            await server.wait_closed()

    response, task = asyncio.run(scenario())
    # A connection-death error, not a normal execution-error code -- the
    # crash never reached execute_task's own error handling at all.
    assert response["type"] == "ERROR"
    assert response["payload"]["code"] == "WORKER_UNREACHABLE"
    assert async_server.worker_manager.get_worker("worker-1").status == WorkerStatus.FAILED
    # Requeued for retry (attempt 1 < MAX_TASK_ATTEMPTS) -- contrast with
    # the execution-failure test above, where the task went straight to
    # FAILED with attempt staying at 1 and the worker staying IDLE.
    assert task.status == TaskStatus.PENDING
    assert task.assigned_worker_id is None
    assert task.attempt == 1


# -- Phase 11.4: resource/concurrency controls, full pipeline ---------


def test_full_worker_master_integration_under_concurrency_limits():
    """A real worker, real master, real dispatched tasks -- with BOTH
    max_workers and max_in_flight configured -- must still process every
    task correctly. Concurrency controls are a backend-internal resource
    concern; they must not change what a caller sees on the wire."""
    registry.register_function("square", lambda x: x * x)

    async def scenario():
        server, host, port = await start_master_server()
        backend = MultiprocessingBackend(max_workers=2, max_in_flight=2)
        worker_task = asyncio.create_task(
            async_worker.run_worker(host, port, worker_id="worker-1", backend=backend)
        )
        await async_server.wait_for_workers(1)

        try:
            tasks = [
                async_server.scheduler.submit_task(f"sq-{i}", "square", {"x": i}) for i in range(5)
            ]
            responses = await async_server.wait_for_tasks({t.task_id for t in tasks})
            return {r["payload"]["task_id"]: r["payload"]["result"] for r in responses}
        finally:
            await stop_worker(worker_task)
            server.close()
            await server.wait_closed()

    results = asyncio.run(scenario())
    assert results == {f"sq-{i}": i * i for i in range(5)}


def test_heartbeat_remains_responsive_under_max_in_flight_limit():
    """The in-flight semaphore must not itself become a source of
    unresponsiveness -- a worker configured with a tight max_in_flight,
    executing a real CPU-bound task, must still send HEARTBEAT on
    schedule (mirrors test_heartbeats_continue_arriving_during_a_real_cpu_bound_dispatch,
    with concurrency limits configured this time)."""
    registry.register_function("slow_double", _slow_double)

    async def scenario():
        server, host, port = await start_master_server()
        backend = MultiprocessingBackend(max_workers=2, max_in_flight=1)
        worker_task = asyncio.create_task(
            async_worker.run_worker(
                host, port, worker_id="worker-1", backend=backend, heartbeat_interval=0.1
            )
        )
        await async_server.wait_for_workers(1)

        try:
            before = async_server.worker_manager.get_worker("worker-1").last_heartbeat
            task = async_server.scheduler.submit_task("t1", "slow_double", {"x": 5})
            [response] = await async_server.wait_for_tasks({task.task_id})
            after = async_server.worker_manager.get_worker("worker-1").last_heartbeat
            return response, before, after
        finally:
            await stop_worker(worker_task)
            server.close()
            await server.wait_closed()

    response, before, after = asyncio.run(scenario())
    assert response["payload"]["result"] == 10
    assert after > before
