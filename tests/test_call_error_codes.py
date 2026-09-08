"""Phase 10.4: jobs.call.CallError/collect_call_result surface the
structured ExecutionErrorCode from a failed task's response, both as a
plain response dict (unit level) and through a real dispatched task
(end-to-end, proving the "code" field actually survives the JSON wire
round trip -- not just present in the in-process response dict)."""

import asyncio

import pytest

from jobs.call import CallError, collect_call_result, submit_call, submit_serialized_call
from master import async_server, rpc_handler
from worker import registry
from worker.executor import ExecutionErrorCode


@pytest.fixture(autouse=True)
def reset_registry():
    registry.clear()
    yield
    registry.clear()


def test_collect_call_result_raises_call_error_with_code_and_message():
    response = {"payload": {"status": "error", "code": "USER_FUNCTION_ERROR", "message": "ValueError: boom"}}
    with pytest.raises(CallError) as excinfo:
        collect_call_result(response)
    assert excinfo.value.code == "USER_FUNCTION_ERROR"
    assert excinfo.value.message == "ValueError: boom"


def test_call_error_is_a_value_error_for_backward_compatible_handling():
    response = {"payload": {"status": "error", "code": "INVALID_ARGUMENTS", "message": "bad payload"}}
    with pytest.raises(ValueError):
        collect_call_result(response)


def test_collect_call_result_surfaces_transport_level_code_unchanged():
    """A task that was never even attempted (WORKER_UNREACHABLE) carries
    its own code the exact same way an execution failure does -- no
    special-casing needed to expose either kind through the same field."""
    response = {"payload": {"code": "WORKER_UNREACHABLE", "message": "no connection"}}
    with pytest.raises(CallError) as excinfo:
        collect_call_result(response)
    assert excinfo.value.code == "WORKER_UNREACHABLE"


def test_collect_call_result_defaults_to_unknown_code_when_absent():
    response = {"payload": {"status": "error", "message": "something went wrong"}}
    with pytest.raises(CallError) as excinfo:
        collect_call_result(response)
    assert excinfo.value.code == "UNKNOWN"


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


@pytest.fixture(autouse=True)
def reset_async_master_state():
    rpc_handler.worker_manager.clear()
    async_server.scheduler.clear()
    async_server.connections.clear()
    yield


def test_execution_error_code_survives_the_real_wire_round_trip():
    """Proves the 'code' field isn't just present on the in-process
    response dict (already covered by test_execution_error_codes.py) --
    it must also survive JSON encoding, a real TCP send, JSON decoding on
    the master side, and collect_call_result's own extraction."""
    from worker import async_worker

    def boom():
        raise ValueError("deliberate failure")

    async def scenario():
        server, host, port = await start_master_server()
        worker_task = asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-1"))
        await async_server.wait_for_workers(1)

        try:
            task = submit_serialized_call(async_server.scheduler, "boom-1", boom)
            [response] = await async_server.wait_for_tasks({task.task_id})
            return response
        finally:
            await stop_worker(worker_task)
            server.close()
            await server.wait_closed()

    response = asyncio.run(scenario())
    with pytest.raises(CallError) as excinfo:
        collect_call_result(response)
    assert excinfo.value.code == ExecutionErrorCode.USER_FUNCTION_ERROR
    assert "deliberate failure" in excinfo.value.message


def test_unknown_operation_code_survives_the_real_wire_round_trip():
    from worker import async_worker

    async def scenario():
        server, host, port = await start_master_server()
        worker_task = asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-1"))
        await async_server.wait_for_workers(1)

        try:
            task = submit_call(async_server.scheduler, "unknown-1", "never_registered")
            [response] = await async_server.wait_for_tasks({task.task_id})
            return response
        finally:
            await stop_worker(worker_task)
            server.close()
            await server.wait_closed()

    response = asyncio.run(scenario())
    with pytest.raises(CallError) as excinfo:
        collect_call_result(response)
    assert excinfo.value.code == ExecutionErrorCode.UNKNOWN_OPERATION
