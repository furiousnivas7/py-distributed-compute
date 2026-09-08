"""Phase 10.6 test matrix: how function-execution tasks (registered and
serialized) compose with the pre-existing failure/retry/generation
machinery (Phases 8.9, 9.1, 9.2) -- proving that machinery, already
proven for ADD-style tasks, works identically for the new execution
paths rather than needing its own parallel implementation.
"""

import asyncio
import time

import pytest

from common.models import TaskStatus, WorkerStatus
from jobs.call import submit_call, submit_serialized_call
from master import async_server, rpc_handler
from rpc import protocol
from rpc.async_connection import AsyncConnection
from rpc.async_rpc import receive_message, send_message, send_request
from rpc.protocol import build_message
from worker import async_worker, registry
from worker.executor import ExecutionErrorCode
from worker.executor import execute_task as _execute_task


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


# -- retry behavior after execution failure ---------------------------------


def test_execution_failure_does_not_trigger_a_retry():
    """A user function's own exception is a DETERMINISTIC failure -- the
    same function called again with the same arguments fails the exact
    same way. Unlike a worker crash (which DOES requeue/retry, see
    Phases 8/9), an execution failure must leave the task FAILED at its
    current attempt, not retried on another worker for no benefit."""
    registry.register_function("always_fails", lambda: 1 / 0)

    async def scenario():
        server, host, port = await start_master_server()
        worker_task = asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-1"))
        await async_server.wait_for_workers(1)

        try:
            task = submit_call(async_server.scheduler, "t1", "always_fails")
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
    # The worker itself is unaffected -- it's fine, the FUNCTION failed,
    # not the worker.
    assert async_server.worker_manager.get_worker("worker-1").status == WorkerStatus.IDLE


# -- stale attempt response handling -----------------------------------------


async def run_delayed_reply_worker(
    host: str, port: int, worker_id: str, ready_event: asyncio.Event, release_event: asyncio.Event
) -> None:
    """Registers, receives one TASK, signals ready, then waits before
    finally executing (via the real execute_task, so this behaves exactly
    like a real worker would for a CALL-style task) and replying."""
    reader, writer = await asyncio.open_connection(host, port)
    conn = AsyncConnection(reader, writer)
    await send_request(conn, protocol.PING)
    await send_request(conn, protocol.REGISTER, {"worker_id": worker_id, "host": "127.0.0.1", "port": 6000})

    message = await receive_message(conn)
    assert message["type"] == protocol.TASK
    ready_event.set()
    await release_event.wait()

    payload = message["payload"]
    result = _execute_task(payload["task_type"], payload["task_payload"])
    response = build_message(
        protocol.TASK_RESULT,
        message["request_id"],
        {"task_id": payload["task_id"], "attempt": payload.get("attempt", 1), **result},
    )
    await send_message(conn, response)
    await conn.close()


def test_stale_attempt_response_for_a_call_task_is_ignored():
    """worker-1 holds a CALL task and goes quiet (connection stays open,
    no heartbeat -- the classic staleness scenario, same as Phase 9.1's
    ADD-based tests). The failure monitor reassigns to worker-2, which
    completes it, and ONLY THEN does worker-1's long-delayed attempt-1
    reply finally arrive. It must not overwrite the already-COMPLETED
    attempt-2 state."""
    registry.register_function("add", lambda a, b: a + b)

    async def scenario():
        server, host, port = await start_master_server()
        ready = asyncio.Event()
        release = asyncio.Event()
        worker1_task = asyncio.create_task(run_delayed_reply_worker(host, port, "worker-1", ready, release))
        await async_server.wait_for_workers(1)
        worker2_task = asyncio.create_task(
            async_worker.run_worker(host, port, worker_id="worker-2", heartbeat_interval=0.1)
        )
        await async_server.wait_for_workers(2)

        try:
            task = submit_call(async_server.scheduler, "t1", "add", a=10, b=20)
            waiter = asyncio.create_task(async_server.wait_for_tasks({task.task_id}))
            await asyncio.wait_for(ready.wait(), timeout=5)

            async_server.worker_manager.get_worker("worker-1").last_heartbeat = time.time() - 10
            stale = async_server.worker_manager.get_stale_workers(async_server.HEARTBEAT_TIMEOUT)
            assert any(w.worker_id == "worker-1" for w in stale)
            async_server.scheduler.requeue_tasks_for_worker("worker-1")

            [response] = await asyncio.wait_for(waiter, timeout=5)
            assert response["payload"]["result"] == 30
            assert response["payload"]["attempt"] == 2

            # NOW let worker-1's long-delayed attempt-1 reply arrive.
            release.set()
            await asyncio.sleep(0.1)

            return response, task
        finally:
            await stop_worker(worker1_task)
            await stop_worker(worker2_task)
            server.close()
            await server.wait_closed()

    response, task = asyncio.run(scenario())
    assert task.status == TaskStatus.COMPLETED
    assert task.assigned_worker_id == "worker-2"
    assert task.attempt == 2


# -- worker restart and generation handling ----------------------------------


async def run_single_crash_worker(host: str, port: int, worker_id: str, ready: asyncio.Event) -> None:
    reader, writer = await asyncio.open_connection(host, port)
    conn = AsyncConnection(reader, writer)
    await send_request(conn, protocol.PING)
    await send_request(conn, protocol.REGISTER, {"worker_id": worker_id, "host": "127.0.0.1", "port": 6000})

    while True:
        message = await receive_message(conn)
        if message["type"] == protocol.TASK:
            break

    ready.set()
    await conn.close()


def test_call_task_reassigned_after_worker_restart_gets_new_generation():
    """worker-1 crashes mid-CALL-task; a fresh connection reconnects
    under the same worker_id. It must get a bumped generation and be
    able to receive and complete a brand-new registered-function task
    normally -- the reconnect/generation machinery (Phase 9.2.1) composes
    with the function-execution paths exactly like it does for ADD."""
    registry.register_function("double", lambda x: x * 2)

    async def scenario():
        server, host, port = await start_master_server()
        ready = asyncio.Event()
        crashing_task = asyncio.create_task(run_single_crash_worker(host, port, "worker-1", ready))
        await async_server.wait_for_workers(1)

        try:
            first_generation = async_server.worker_manager.get_worker("worker-1").generation
            doomed = submit_call(async_server.scheduler, "doomed", "double", x=1)
            assigned = async_server.scheduler.assign_next_pending_task()
            assert assigned.assigned_worker_id == "worker-1"
            dispatch = asyncio.create_task(async_server.dispatch_assigned_task(doomed))
            await asyncio.wait_for(ready.wait(), timeout=5)
            doomed_response = await dispatch

            rescue_task = asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-1"))
            deadline = time.monotonic() + 5
            while (
                async_server.worker_manager.get_worker("worker-1").generation == first_generation
                and time.monotonic() < deadline
            ):
                await asyncio.sleep(0.02)

            worker = async_server.worker_manager.get_worker("worker-1")
            assert worker.generation == first_generation + 1
            assert worker.status == WorkerStatus.IDLE

            fresh = submit_call(async_server.scheduler, "fresh", "double", x=21)
            [fresh_response] = await async_server.wait_for_tasks({fresh.task_id})
            return doomed_response, fresh_response
        finally:
            await stop_worker(crashing_task)
            await stop_worker(rescue_task)
            server.close()
            await server.wait_closed()

    doomed_response, fresh_response = asyncio.run(scenario())
    assert doomed_response["payload"]["code"] == "WORKER_UNREACHABLE"
    assert fresh_response["payload"]["result"] == 42


def test_serialized_callable_task_reassigned_after_worker_restart():
    """Same generation/reconnect guarantee, for a serialized callable
    this time -- proving it's not specific to the registered-function
    path."""

    def times_three(x):
        return x * 3

    async def scenario():
        server, host, port = await start_master_server()
        ready = asyncio.Event()
        crashing_task = asyncio.create_task(run_single_crash_worker(host, port, "worker-1", ready))
        await async_server.wait_for_workers(1)

        try:
            first_generation = async_server.worker_manager.get_worker("worker-1").generation
            doomed = submit_serialized_call(async_server.scheduler, "doomed", times_three, args=[1])
            assigned = async_server.scheduler.assign_next_pending_task()
            assert assigned.assigned_worker_id == "worker-1"
            dispatch = asyncio.create_task(async_server.dispatch_assigned_task(doomed))
            await asyncio.wait_for(ready.wait(), timeout=5)
            await dispatch

            rescue_task = asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-1"))
            deadline = time.monotonic() + 5
            while (
                async_server.worker_manager.get_worker("worker-1").generation == first_generation
                and time.monotonic() < deadline
            ):
                await asyncio.sleep(0.02)
            assert async_server.worker_manager.get_worker("worker-1").generation == first_generation + 1

            fresh = submit_serialized_call(async_server.scheduler, "fresh", times_three, args=[14])
            [fresh_response] = await async_server.wait_for_tasks({fresh.task_id})
            return fresh_response
        finally:
            await stop_worker(crashing_task)
            await stop_worker(rescue_task)
            server.close()
            await server.wait_closed()

    fresh_response = asyncio.run(scenario())
    assert fresh_response["payload"]["result"] == 42


# -- backward compatibility with existing ADD tasks --------------------------


def test_add_tasks_are_completely_unaffected_by_the_function_execution_model():
    """ADD is still handled by its own dedicated HANDLERS entry -- the
    registry fallback in execute_task is never even consulted for it."""

    async def scenario():
        server, host, port = await start_master_server()
        worker_task = asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-1"))
        await async_server.wait_for_workers(1)

        try:
            task = async_server.scheduler.submit_task("t1", "ADD", {"a": 10, "b": 20})
            [response] = await async_server.wait_for_tasks({task.task_id})
            return response
        finally:
            await stop_worker(worker_task)
            server.close()
            await server.wait_closed()

    response = asyncio.run(scenario())
    assert response["payload"] == {"task_id": "t1", "attempt": 1, "status": "success", "result": 30}
