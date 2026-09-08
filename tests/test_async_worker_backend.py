"""Phase 11.1: proves backend selection is real and testable through the
actual worker/master pipeline, not just at the worker/backend.py unit
level (tests/test_backend.py) -- run_worker() must actually route task
execution through whatever backend it's given, and call start()/stop()
at the right points in its own lifecycle.
"""

import asyncio

import pytest

from master import async_server, rpc_handler
from worker import async_worker
from worker.backend import ExecutionBackend


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


class RecordingBackend(ExecutionBackend):
    """A stub backend that ignores the real task entirely and returns a
    fixed, recognizable result -- proving run_worker/serve_tasks actually
    dispatch through the INJECTED backend rather than silently falling
    back to DirectBackend. Also records lifecycle calls, to prove
    start()/stop() are invoked exactly once each, at the right points
    relative to serving tasks."""

    def __init__(self):
        self.events = []
        self.executed = []

    async def start(self) -> None:
        self.events.append("start")

    async def stop(self) -> None:
        self.events.append("stop")

    async def execute(self, task_type: str, payload: dict) -> dict:
        self.executed.append((task_type, payload))
        return {"status": "success", "result": "handled-by-recording-backend"}


def test_run_worker_routes_execution_through_the_injected_backend():
    backend = RecordingBackend()

    async def scenario():
        server, host, port = await start_master_server()
        worker_task = asyncio.create_task(
            async_worker.run_worker(host, port, worker_id="worker-1", backend=backend)
        )
        await async_server.wait_for_workers(1)

        try:
            # A real ADD task -- if DirectBackend were still being used
            # under the hood, this would come back as {"result": 3}, not
            # the RecordingBackend's fixed marker value.
            task = async_server.scheduler.submit_task("t1", "ADD", {"a": 1, "b": 2})
            [response] = await async_server.wait_for_tasks({task.task_id})
            return response
        finally:
            await stop_worker(worker_task)
            server.close()
            await server.wait_closed()

    response = asyncio.run(scenario())
    assert response["payload"]["result"] == "handled-by-recording-backend"
    assert backend.executed == [("ADD", {"a": 1, "b": 2})]


def test_run_worker_calls_backend_start_before_serving_and_stop_on_exit():
    backend = RecordingBackend()

    async def scenario():
        server, host, port = await start_master_server()
        worker_task = asyncio.create_task(
            async_worker.run_worker(host, port, worker_id="worker-1", backend=backend)
        )
        await async_server.wait_for_workers(1)

        try:
            # start() must have already happened by the time registration
            # (which only succeeds once serve_tasks-adjacent setup has
            # run) completes.
            assert backend.events == ["start"]
        finally:
            await stop_worker(worker_task)
            server.close()
            await server.wait_closed()

        return backend.events

    events = asyncio.run(scenario())
    assert events == ["start", "stop"]


def test_default_backend_is_direct_when_none_is_given():
    """No `backend` argument -- run_worker must fall back to DirectBackend
    and behave exactly as it did before Phase 11 (a real ADD, correctly
    computed, not a stub value)."""

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
    assert response["payload"]["result"] == 30
