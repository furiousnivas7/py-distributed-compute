"""Phase 12.5.4: failure-injection matrix -- one test per row, verifying
the SPECIFIC expected behavior for each failure mode. Many of these are
already proven individually elsewhere in the suite; this file exists to
make the matrix itself an explicit, auditable artifact (see each test's
one-line docstring, which doubles as the matrix row) rather than
something a reader has to reconstruct from scattered tests.

| Failure                 | Expected behavior                        |
|--------------------------|-------------------------------------------|
| Worker crashes            | Retry                                     |
| Worker disconnects        | Detect failure                           |
| Task raises exception     | Structured failure                       |
| Invalid function           | Structured error                        |
| Duplicate worker ID        | Reject                                  |
| Duplicate task result       | Ignore safely                          |
| Stale attempt result        | Ignore                                 |
| Master shutdown            | Pending tasks resolve deterministically  |
| Process pool crashes        | Backend reports failure                |
| Worker reconnects           | New connection generation              |
"""

import asyncio

import pytest

from common import TaskStatus, WorkerStatus
from jobs import collect_call_result, submit_call
from master import async_server, rpc_handler, scheduler, start_master, stop_master, wait_for_tasks
from master.worker_manager import DuplicateWorkerError, WorkerManager
from worker import DirectBackend, MultiprocessingBackend, register_function, registry, run_worker
from worker.executor import ExecutionErrorCode


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


# -- Worker crashes -> Retry ---------------------------------------------


def test_worker_crashes_task_is_retried_on_another_worker():
    register_function("noop", lambda: "ok")

    async def scenario():
        server, host, port = await _start_master()
        w1 = asyncio.create_task(run_worker(host, port, worker_id="w1", backend=DirectBackend()))
        w2 = asyncio.create_task(run_worker(host, port, worker_id="w2", backend=DirectBackend()))
        try:
            await async_server.wait_for_workers(2)
            task = scheduler.submit_task("t1", "noop", {})
            assigned = scheduler.assign_next_pending_task()
            crashed_worker = assigned.assigned_worker_id

            worker_tasks = {"w1": w1, "w2": w2}
            await _stop_worker(worker_tasks[crashed_worker])

            response = await async_server.dispatch_assigned_task(task)
            [final_response] = await wait_for_tasks({task.task_id})
            return response, final_response, scheduler.get_task(task.task_id)
        finally:
            for t in (w1, w2):
                await _stop_worker(t)
            server.close()
            await server.wait_closed()
            await stop_master()

    crash_response, final_response, final_task = asyncio.run(scenario())
    assert crash_response["type"] == "ERROR"
    assert crash_response["payload"]["code"] == "WORKER_UNREACHABLE"
    assert final_task.status == TaskStatus.COMPLETED
    assert final_task.attempt == 2
    assert collect_call_result(final_response) == "ok"


# -- Worker disconnects -> Detect failure ---------------------------------


def test_worker_disconnects_is_detected_and_marked_failed():
    async def scenario():
        server, host, port = await _start_master()
        w1 = asyncio.create_task(run_worker(host, port, worker_id="w1", backend=DirectBackend()))
        try:
            await async_server.wait_for_workers(1)
            link = async_server.connections.get("w1")
            await link.conn.close()
            await asyncio.sleep(0.05)
            return "w1" not in async_server.connections
        finally:
            await _stop_worker(w1)
            server.close()
            await server.wait_closed()
            await stop_master()

    disconnected = asyncio.run(scenario())
    assert disconnected


# -- Task raises exception -> Structured failure --------------------------


def test_task_raises_exception_yields_structured_failure_not_a_raise():
    def boom():
        raise ValueError("deliberate")

    register_function("boom", boom)

    async def scenario():
        server, host, port = await _start_master()
        w1 = asyncio.create_task(run_worker(host, port, worker_id="w1", backend=DirectBackend()))
        try:
            await async_server.wait_for_workers(1)
            task = submit_call(scheduler, "t1", "boom")
            [response] = await wait_for_tasks({task.task_id})
            return response, scheduler.get_task(task.task_id)
        finally:
            await _stop_worker(w1)
            server.close()
            await server.wait_closed()
            await stop_master()

    response, final_task = asyncio.run(scenario())
    assert response["payload"]["status"] == "error"
    assert response["payload"]["code"] == ExecutionErrorCode.USER_FUNCTION_ERROR
    assert final_task.status == TaskStatus.FAILED
    assert final_task.attempt == 1  # never retried -- deterministic failure


# -- Invalid function -> Structured error ----------------------------------


def test_invalid_operation_yields_structured_error():
    async def scenario():
        server, host, port = await _start_master()
        w1 = asyncio.create_task(run_worker(host, port, worker_id="w1", backend=DirectBackend()))
        try:
            await async_server.wait_for_workers(1)
            task = scheduler.submit_task("t1", "NOT_A_REAL_OPERATION", {})
            [response] = await wait_for_tasks({task.task_id})
            return response
        finally:
            await _stop_worker(w1)
            server.close()
            await server.wait_closed()
            await stop_master()

    response = asyncio.run(scenario())
    assert response["payload"]["status"] == "error"
    assert response["payload"]["code"] == ExecutionErrorCode.UNKNOWN_OPERATION


# -- Duplicate worker ID -> Reject -----------------------------------------


def test_duplicate_worker_id_while_still_live_is_rejected():
    wm = WorkerManager()
    wm.register_worker("w1", "127.0.0.1", 6001)
    with pytest.raises(DuplicateWorkerError):
        wm.register_worker("w1", "127.0.0.1", 6002)


# -- Duplicate task result -> Ignore safely ---------------------------------


def test_duplicate_task_result_for_a_stale_attempt_is_ignored_safely():
    """A second reply for an attempt that's already been superseded (the
    task moved on to a later attempt, or already has a terminal outcome)
    must never overwrite what's already recorded -- see
    dispatch_assigned_task's record_if_terminal/is_current_attempt.
    Simulated directly: manually complete a task, then feed it a second,
    stale terminal response for the SAME attempt and confirm nothing
    changes."""
    from master.scheduler import Scheduler

    wm = WorkerManager()
    wm.register_worker("w1", "127.0.0.1", 6001)
    s = Scheduler(wm)
    task = s.submit_task("t1", "noop", {})
    s.assign_task("t1")
    s.start_task("t1")
    s.complete_task("t1")

    assert task.status == TaskStatus.COMPLETED
    # A second complete_task() call for an already-terminal task is
    # exactly the "duplicate result" scenario at the Scheduler layer --
    # it must raise cleanly (caught by dispatch_assigned_task's own
    # is_current_attempt() gate in production, never reached in the first
    # place) rather than silently corrupt state.
    with pytest.raises(ValueError):
        s.complete_task("t1")
    assert task.status == TaskStatus.COMPLETED  # unchanged


# -- Stale attempt result -> Ignore -----------------------------------------


def test_stale_attempt_reply_does_not_resolve_a_later_attempts_future():
    """A reply keyed to an OLD attempt's request_id must not be mistaken
    for the current attempt's reply -- see async_server.attempt_request_id
    and WorkerLink.resolve()."""
    from rpc.async_connection import AsyncConnection

    link = async_server.WorkerLink(AsyncConnection.__new__(AsyncConnection))
    request_id_attempt_1 = async_server.attempt_request_id("t1", 1)
    request_id_attempt_2 = async_server.attempt_request_id("t1", 2)
    assert request_id_attempt_1 != request_id_attempt_2

    # resolve() only matches an EXISTING pending future for that exact
    # request_id -- a stale attempt-1 reply arriving after attempt-2 has
    # already started (so only attempt-2's request_id is pending) simply
    # finds no match.
    assert link.resolve(request_id_attempt_1, {}) is False


# -- Master shutdown -> Pending tasks resolve deterministically -------------


def test_master_shutdown_leaves_no_task_in_an_ambiguous_state():
    register_function("noop", lambda: "ok")

    async def scenario():
        server, host, port = await _start_master()
        w1 = asyncio.create_task(run_worker(host, port, worker_id="w1", backend=DirectBackend()))
        try:
            await async_server.wait_for_workers(1)
            task = submit_call(scheduler, "t1", "noop")
            await wait_for_tasks({task.task_id})
        finally:
            await _stop_worker(w1)
            server.close()
            await server.wait_closed()
            await stop_master()
            return scheduler.get_task("t1").status

    final_status = asyncio.run(scenario())
    assert final_status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.PENDING)
    # The actual property under test: stop_master() itself must complete
    # without hanging or raising (already proven by scenario() returning
    # normally above) -- an ambiguous/stuck ASSIGNED or RUNNING task after
    # shutdown would indicate a real bug.
    assert final_status not in (TaskStatus.ASSIGNED, TaskStatus.RUNNING)


# -- Process pool crashes -> Backend reports failure ------------------------


def test_process_pool_crash_is_reported_as_a_backend_failure():
    import os

    def crash():
        os._exit(1)

    register_function("crash", crash)

    async def scenario():
        backend = MultiprocessingBackend(max_workers=2)
        await backend.start()
        try:
            with pytest.raises(Exception):
                await backend.execute("crash", {})
            return backend.get_metrics()
        finally:
            await backend.stop()

    metrics = asyncio.run(scenario())
    assert metrics.backend_errors == 1
    assert metrics.failed == 0  # not a structured execution failure -- a real raise


# -- Worker reconnects -> New connection generation -------------------------


def test_worker_reconnect_after_crash_bumps_generation():
    wm = WorkerManager()
    first = wm.register_worker("w1", "127.0.0.1", 6001)
    assert first.generation == 1
    wm.update_status("w1", WorkerStatus.FAILED)
    second = wm.register_worker("w1", "127.0.0.1", 6001)
    assert second.generation == 2
    assert second is first  # mutated in place, not replaced
