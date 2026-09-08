"""Phase 10.2: registered-function tasks proven end-to-end over the real
async transport -- a function registered via worker.registry, submitted
via jobs.call.submit_call, dispatched to a real worker process (well,
coroutine, but a separate connection/registration just like a real one)
through master.async_server's existing Scheduler / dispatch_assigned_task
/ wait_for_tasks, exactly like ADD or MULTIPLY. No CALL-specific dispatch
logic exists anywhere -- a registered operation's task_type is its own
name -- and this file is what proves that's actually true, not just true
by design intent.
"""

import asyncio

import pytest

from jobs.call import collect_call_result, submit_call
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


def test_registered_function_dispatched_to_a_real_worker_and_executed():
    def fibonacci(n):
        a, b = 0, 1
        for _ in range(n):
            a, b = b, a + b
        return a

    registry.register_function("fibonacci", fibonacci)

    async def scenario():
        server, host, port = await start_master_server()
        worker_task = asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-1"))
        await async_server.wait_for_workers(1)

        try:
            task = submit_call(async_server.scheduler, "fib-10", "fibonacci", n=10)
            [response] = await async_server.wait_for_tasks({task.task_id})
            return response
        finally:
            await stop_worker(worker_task)
            server.close()
            await server.wait_closed()

    response = asyncio.run(scenario())
    assert collect_call_result(response) == 55


def test_registered_function_with_multiple_keyword_arguments():
    registry.register_function("greet", lambda name, greeting="Hello": f"{greeting}, {name}!")

    async def scenario():
        server, host, port = await start_master_server()
        worker_task = asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-1"))
        await async_server.wait_for_workers(1)

        try:
            task = submit_call(async_server.scheduler, "greet-1", "greet", name="Ada", greeting="Hi")
            [response] = await async_server.wait_for_tasks({task.task_id})
            return response
        finally:
            await stop_worker(worker_task)
            server.close()
            await server.wait_closed()

    response = asyncio.run(scenario())
    assert collect_call_result(response) == "Hi, Ada!"


def test_registered_function_failure_surfaces_through_collect_call_result():
    def picky(x):
        if x < 0:
            raise ValueError("x must be non-negative")
        return x

    registry.register_function("picky", picky)

    async def scenario():
        server, host, port = await start_master_server()
        worker_task = asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-1"))
        await async_server.wait_for_workers(1)

        try:
            task = submit_call(async_server.scheduler, "picky-1", "picky", x=-5)
            [response] = await async_server.wait_for_tasks({task.task_id})
            return response
        finally:
            await stop_worker(worker_task)
            server.close()
            await server.wait_closed()

    response = asyncio.run(scenario())
    with pytest.raises(ValueError, match="ValueError: x must be non-negative"):
        collect_call_result(response)


def test_multiple_registered_function_tasks_share_worker_pool_like_any_other_task_type():
    registry.register_function("square", lambda x: x * x)

    async def scenario():
        server, host, port = await start_master_server()
        worker_tasks = [
            asyncio.create_task(async_worker.run_worker(host, port, worker_id=f"worker-{i}")) for i in range(1, 3)
        ]
        await async_server.wait_for_workers(2)

        try:
            tasks = [submit_call(async_server.scheduler, f"sq-{i}", "square", x=i) for i in range(5)]
            responses = await async_server.wait_for_tasks({t.task_id for t in tasks})
            return {r["payload"]["task_id"]: collect_call_result(r) for r in responses}
        finally:
            for t in worker_tasks:
                await stop_worker(t)
            server.close()
            await server.wait_closed()

    results = asyncio.run(scenario())
    assert results == {f"sq-{i}": i * i for i in range(5)}
