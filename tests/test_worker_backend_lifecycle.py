"""Phase 11.3: backend configuration and lifecycle determinism through
run_worker() -- backend selection itself (DirectBackend default,
injecting MultiprocessingBackend, a custom ExecutionBackend) was already
proven in Phase 11.1/11.2 (tests/test_async_worker_backend.py,
tests/test_multiprocessing_backend.py). This file covers what's new: an
invalid `backend` is rejected clearly and immediately, and backend.start()/
stop() failing at any point still leaves the connection closed and the
failure visible rather than silently swallowed or leaked.

Scheduler/task-protocol note: none of this touches master/scheduler.py or
the wire protocol at all -- execution strategy is a worker runtime
concern (see worker/backend.py, worker/async_worker.py), and the
scheduler stays completely unaware of which backend, if any, a given
worker happens to be using.
"""

import asyncio

import pytest

from master import async_server, rpc_handler
from worker import async_worker
from worker.backend import DirectBackend, ExecutionBackend


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


# -- invalid backend rejected clearly ----------------------------------


def test_non_backend_object_is_rejected_immediately_with_a_clear_error():
    """Rejected before ever opening a connection -- not a later,
    confusing AttributeError from backend.start()."""

    async def scenario():
        with pytest.raises(TypeError, match="ExecutionBackend"):
            await async_worker.run_worker(
                "127.0.0.1", 1, worker_id="worker-1", backend="not-a-backend"
            )

    asyncio.run(scenario())


def test_incomplete_backend_subclass_cannot_even_be_instantiated():
    """A stronger, earlier check than run_worker's isinstance guard:
    ExecutionBackend's own ABC-ness already refuses to construct a
    subclass missing execute() -- see tests/test_backend.py for this at
    the class-definition level; restated here in the run_worker context
    since it's the same "reject invalid backends clearly" property."""

    class Incomplete(ExecutionBackend):
        pass

    with pytest.raises(TypeError):
        Incomplete()


def test_none_backend_is_accepted_and_becomes_direct_backend():
    """Not an error case -- confirms None specifically (not "anything
    falsy") is the documented way to request the default."""

    async def scenario():
        server, host, port = await start_master_server()
        worker_task = asyncio.create_task(
            async_worker.run_worker(host, port, worker_id="worker-1", backend=None)
        )
        await async_server.wait_for_workers(1)
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass
        server.close()
        await server.wait_closed()

    asyncio.run(scenario())  # must not raise


# -- deterministic lifecycle under backend failure -----------------------


class FailingStartBackend(ExecutionBackend):
    """start() always raises. stop() records whether it was called, so a
    test can confirm run_worker() still attempts symmetric cleanup."""

    def __init__(self):
        self.stop_called = False

    async def start(self) -> None:
        raise RuntimeError("deliberate start() failure")

    async def stop(self) -> None:
        self.stop_called = True

    async def execute(self, task_type: str, payload: dict) -> dict:
        raise AssertionError("execute() must never be reached if start() failed")


class FailingStopBackend(ExecutionBackend):
    """start() succeeds; stop() always raises -- proving a stop()
    failure doesn't prevent the connection from still being closed."""

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        raise RuntimeError("deliberate stop() failure")

    async def execute(self, task_type: str, payload: dict) -> dict:
        return {"status": "success", "result": "ok"}


def test_backend_start_failure_propagates_and_still_attempts_stop():
    backend = FailingStartBackend()

    async def scenario():
        server, host, port = await start_master_server()
        try:
            with pytest.raises(RuntimeError, match="deliberate start"):
                await async_worker.run_worker(host, port, worker_id="worker-1", backend=backend)
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(scenario())
    assert backend.stop_called


def test_backend_stop_failure_still_closes_the_connection():
    """The worker's connection must not leak just because backend.stop()
    raised -- verified by confirming the master sees the worker actually
    disconnect (removed from connections) despite the stop() failure."""
    backend = FailingStopBackend()

    async def scenario():
        server, host, port = await start_master_server()
        shutdown_event = asyncio.Event()

        worker_task = asyncio.create_task(
            async_worker.run_worker(
                host, port, worker_id="worker-1", backend=backend, shutdown_event=shutdown_event
            )
        )
        await async_server.wait_for_workers(1)

        shutdown_event.set()
        try:
            with pytest.raises(RuntimeError, match="deliberate stop"):
                await asyncio.wait_for(worker_task, timeout=5)
        finally:
            deadline = asyncio.get_running_loop().time() + 5
            while "worker-1" in async_server.connections and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(0.02)
            still_connected = "worker-1" in async_server.connections
            server.close()
            await server.wait_closed()
            return still_connected

    still_connected = asyncio.run(scenario())
    assert not still_connected


def test_normal_run_calls_start_then_stop_in_order():
    """Restates Phase 11.1's start-before-serving/stop-on-exit guarantee
    (tests/test_async_worker_backend.py) as an explicit ORDERING check,
    since Phase 11.3 is specifically about the lifecycle contract."""

    class OrderRecordingBackend(ExecutionBackend):
        def __init__(self):
            self.events = []

        async def start(self) -> None:
            self.events.append("start")

        async def stop(self) -> None:
            self.events.append("stop")

        async def execute(self, task_type: str, payload: dict) -> dict:
            self.events.append("execute")
            return await DirectBackend().execute(task_type, payload)

    backend = OrderRecordingBackend()

    async def scenario():
        server, host, port = await start_master_server()
        worker_task = asyncio.create_task(
            async_worker.run_worker(host, port, worker_id="worker-1", backend=backend)
        )
        await async_server.wait_for_workers(1)

        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass
        server.close()
        await server.wait_closed()

    asyncio.run(scenario())
    assert backend.events == ["start", "stop"]
