"""Phase 10.3: EXECUTE-envelope tasks proven end-to-end over the real
async transport -- a serialized callable (or a registered-function
envelope) submitted via jobs.call, dispatched to a real worker process
through master.async_server's existing Scheduler / dispatch_assigned_task
/ wait_for_tasks, exactly like ADD or MULTIPLY. This is the important one
to prove over a REAL connection, not just in-process: the callable is
cloudpickled, base64-encoded, sent as a JSON TASK message, decoded, and
deserialized on the "other side" -- a worker/async_worker.py process that
never imported the function from anywhere.
"""

import asyncio

import pytest

from jobs.call import collect_call_result, submit_registered_call, submit_serialized_call
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


def test_serialized_callable_dispatched_to_a_real_worker_and_executed():
    """A closure (captures `multiplier`, never registered anywhere) is
    serialized on the submitting side, sent over a real TCP connection as
    part of a TASK message, and executed on a worker that never imported
    it -- proving the whole round trip, not just serialize/deserialize in
    the same process."""
    multiplier = 7

    def scaled(x):
        return x * multiplier

    async def scenario():
        server, host, port = await start_master_server()
        worker_task = asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-1"))
        await async_server.wait_for_workers(1)

        try:
            task = submit_serialized_call(async_server.scheduler, "scaled-1", scaled, args=[6])
            [response] = await async_server.wait_for_tasks({task.task_id})
            return response
        finally:
            await stop_worker(worker_task)
            server.close()
            await server.wait_closed()

    response = asyncio.run(scenario())
    assert collect_call_result(response) == 42


def test_serialized_callable_failure_surfaces_through_collect_call_result():
    def divide(a, b):
        return a / b

    async def scenario():
        server, host, port = await start_master_server()
        worker_task = asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-1"))
        await async_server.wait_for_workers(1)

        try:
            task = submit_serialized_call(async_server.scheduler, "divide-1", divide, args=[1, 0])
            [response] = await async_server.wait_for_tasks({task.task_id})
            return response
        finally:
            await stop_worker(worker_task)
            server.close()
            await server.wait_closed()

    response = asyncio.run(scenario())
    with pytest.raises(ValueError, match="ZeroDivisionError"):
        collect_call_result(response)


def test_registered_call_via_execute_envelope_dispatched_to_a_real_worker():
    registry.register_function("square", lambda x: x * x)

    async def scenario():
        server, host, port = await start_master_server()
        worker_task = asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-1"))
        await async_server.wait_for_workers(1)

        try:
            task = submit_registered_call(async_server.scheduler, "sq-1", "square", x=9)
            [response] = await async_server.wait_for_tasks({task.task_id})
            return response
        finally:
            await stop_worker(worker_task)
            server.close()
            await server.wait_closed()

    response = asyncio.run(scenario())
    assert collect_call_result(response) == 81


def test_mixed_workload_of_serialized_and_registered_and_builtin_tasks():
    """A single worker pool handling ADD (built-in), a registered
    function (bare task_type), and a serialized callable (EXECUTE
    envelope) all in the same batch -- proving these three execution
    paths compose without interfering with each other."""
    registry.register_function("double", lambda x: x * 2)

    def cube(x):
        return x**3

    async def scenario():
        server, host, port = await start_master_server()
        worker_tasks = [
            asyncio.create_task(async_worker.run_worker(host, port, worker_id=f"worker-{i}")) for i in range(1, 3)
        ]
        await async_server.wait_for_workers(2)

        try:
            add_task = async_server.scheduler.submit_task("add-1", "ADD", {"a": 1, "b": 1})
            double_task = async_server.scheduler.submit_task("double-1", "double", {"x": 5})
            cube_task = submit_serialized_call(async_server.scheduler, "cube-1", cube, args=[3])

            responses = await async_server.wait_for_tasks(
                {add_task.task_id, double_task.task_id, cube_task.task_id}
            )
            by_id = {r["payload"]["task_id"]: r for r in responses}
            return by_id
        finally:
            for t in worker_tasks:
                await stop_worker(t)
            server.close()
            await server.wait_closed()

    by_id = asyncio.run(scenario())
    assert by_id["add-1"]["payload"]["result"] == 2
    assert by_id["double-1"]["payload"]["result"] == 10
    assert collect_call_result(by_id["cube-1"]) == 27
