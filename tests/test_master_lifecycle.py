"""Phase 9.3: master runtime lifecycle (start_master/stop_master).

Distinct from Phase 9.2's worker shutdown: a worker draining is one
participant leaving in an orderly way, coordinated through the master.
Master shutdown is the whole runtime -- accepting connections, the
dispatcher, the failure monitor, every open worker connection -- coming
down together, deterministically, in an order chosen so nothing already
stopped can still act on what a later step does (see stop_master's own
docstring for the full ordering rationale):

    1. stop accepting new connections
    2. stop the dispatcher (and await everything it already spawned)
    3. stop the failure monitor
    4/5. close every open worker connection (in-flight work fails through
         the ordinary ConnectionError path, same as a real crash)
    6. give that cleanup a bounded chance to finish, then force-clear
       `connections` as a safety net
    7. clear the response registry

Both start_master() and stop_master() are idempotent -- safe to call
repeatedly or when nothing (or everything) is already running -- which is
what makes repeated start/stop cycles (test_repeated_start_stop_cycles_*)
safe to run without accumulating leaked state or asyncio tasks.
"""

import asyncio

import pytest

from common.models import TaskStatus, WorkerStatus
from master import async_server, rpc_handler
from rpc import protocol
from rpc.async_connection import AsyncConnection
from rpc.async_rpc import receive_message, send_request
from worker import async_worker


@pytest.fixture(autouse=True)
def reset_async_master_state():
    rpc_handler.worker_manager.clear()
    async_server.scheduler.clear()
    async_server.connections.clear()
    yield
    # start_master/stop_master are module-level state, like everything
    # else this file's autouse fixtures already reset -- but unlike
    # those, a test that fails mid-scenario could leave the server
    # genuinely still listening. Guard against that leaking into the
    # NEXT test regardless of pass/fail.
    if async_server.is_master_running():
        asyncio.run(async_server.stop_master())


async def stop_worker(task: asyncio.Task) -> None:
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


def test_start_master_is_idempotent_returns_same_server():
    async def scenario():
        try:
            server1 = await async_server.start_master(port=0)
            server2 = await async_server.start_master(port=0)
            return server1, server2
        finally:
            await async_server.stop_master()

    server1, server2 = asyncio.run(scenario())
    assert server1 is server2


def test_stop_master_is_idempotent_when_nothing_running():
    async def scenario():
        assert not async_server.is_master_running()
        await async_server.stop_master()  # must not raise
        assert not async_server.is_master_running()

    asyncio.run(scenario())


def test_stop_master_is_idempotent_when_called_twice():
    async def scenario():
        await async_server.start_master(port=0)
        await async_server.stop_master()
        assert not async_server.is_master_running()
        await async_server.stop_master()  # must not raise, still a no-op
        assert not async_server.is_master_running()

    asyncio.run(scenario())


def test_stop_master_stops_accepting_new_worker_connections():
    async def scenario():
        server = await async_server.start_master(port=0)
        host, port = server.sockets[0].getsockname()[:2]
        await async_server.stop_master()

        with pytest.raises((ConnectionRefusedError, OSError)):
            await asyncio.wait_for(asyncio.open_connection(host, port), timeout=2)

    asyncio.run(scenario())


def test_stop_master_stops_dispatcher_and_failure_monitor():
    async def scenario():
        await async_server.start_master(port=0)
        async_server.ensure_dispatcher_running()
        assert async_server.is_dispatcher_running()

        await async_server.stop_master()

        assert not async_server.is_dispatcher_running()
        assert async_server._master_failure_monitor_task is None

    asyncio.run(scenario())


async def run_non_replying_worker(host: str, port: int, worker_id: str, ready: asyncio.Event) -> None:
    """Registers, receives exactly one TASK, signals `ready`, then just
    sits on the connection without ever replying -- a real async_worker
    would execute+reply to ADD/MULTIPLY near-instantly, too fast to
    reliably still be "in flight" by the time a test gets around to
    closing its connection out from under it."""
    reader, writer = await asyncio.open_connection(host, port)
    conn = AsyncConnection(reader, writer)
    await send_request(conn, protocol.PING)
    await send_request(conn, protocol.REGISTER, {"worker_id": worker_id, "host": "127.0.0.1", "port": 6000})

    message = await receive_message(conn)
    assert message["type"] == protocol.TASK
    ready.set()
    await asyncio.Event().wait()  # never set -- just holds the connection open


def test_stop_master_closes_worker_connections_and_requeues_in_flight_work():
    """A task still in flight when stop_master() runs must fail through
    the ordinary ConnectionError path -- worker marked FAILED, task
    requeued -- exactly like a real crash. There's no special "shutdown"
    outcome invented for it; it just stays PENDING afterward since
    nothing is left running to pick it back up."""

    async def scenario():
        server = await async_server.start_master(port=0)
        host, port = server.sockets[0].getsockname()[:2]
        ready = asyncio.Event()
        worker_task = asyncio.create_task(run_non_replying_worker(host, port, "worker-1", ready))
        await async_server.wait_for_workers(1)

        task = async_server.scheduler.submit_task("t1", "ADD", {"a": 1, "b": 1})
        assigned = async_server.scheduler.assign_task("t1")
        assert assigned.assigned_worker_id == "worker-1"
        dispatch = asyncio.create_task(async_server.dispatch_assigned_task(task))
        await asyncio.wait_for(ready.wait(), timeout=5)

        try:
            await async_server.stop_master()
            response = await asyncio.wait_for(dispatch, timeout=5)
            return response, task, async_server.worker_manager.get_worker("worker-1").status
        finally:
            await stop_worker(worker_task)

    response, task, worker_status = asyncio.run(scenario())
    assert response["payload"]["code"] == "WORKER_UNREACHABLE"
    assert worker_status == WorkerStatus.FAILED
    assert task.status == TaskStatus.PENDING
    assert task.assigned_worker_id is None


def test_stop_master_clears_response_registry():
    async def scenario():
        server = await async_server.start_master(port=0)
        host, port = server.sockets[0].getsockname()[:2]
        worker_task = asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-1"))
        await async_server.wait_for_workers(1)

        task = async_server.scheduler.submit_task("t1", "ADD", {"a": 2, "b": 2})
        [response] = await async_server.wait_for_tasks({task.task_id})
        assert response["payload"]["result"] == 4
        assert "t1" in async_server._task_responses

        try:
            await async_server.stop_master()
        finally:
            await stop_worker(worker_task)

        return "t1" in async_server._task_responses

    still_present = asyncio.run(scenario())
    assert not still_present


def test_repeated_start_stop_cycles_leave_no_leaked_asyncio_tasks():
    """Start and stop the master runtime several times in a row, with
    real workers and real dispatched work each cycle. The set of
    non-done asyncio tasks left running after the final stop must match
    what existed before the very first start -- no dispatcher, failure
    monitor, or per-connection task from any cycle left dangling."""

    async def one_cycle(i: int):
        server = await async_server.start_master(port=0)
        host, port = server.sockets[0].getsockname()[:2]
        worker_task = asyncio.create_task(async_worker.run_worker(host, port, worker_id=f"worker-{i}"))
        await async_server.wait_for_workers(1)

        task = async_server.scheduler.submit_task(f"t-{i}", "ADD", {"a": i, "b": i})
        [response] = await async_server.wait_for_tasks({task.task_id})

        await stop_worker(worker_task)
        await async_server.stop_master()
        return response

    async def scenario():
        baseline = {t for t in asyncio.all_tasks() if not t.done()}

        responses = []
        for i in range(3):
            rpc_handler.worker_manager.clear()
            async_server.scheduler.clear()
            async_server.connections.clear()
            responses.append(await one_cycle(i))

        # Let any just-cancelled/just-finished tasks' done-callbacks settle.
        await asyncio.sleep(0.05)
        leftover = {t for t in asyncio.all_tasks() if not t.done()} - baseline - {asyncio.current_task()}
        return responses, leftover

    responses, leftover = asyncio.run(scenario())
    for i, response in enumerate(responses):
        assert response["payload"]["result"] == i + i
    assert leftover == set(), f"leaked tasks after repeated start/stop cycles: {leftover}"


def test_stop_master_lets_a_draining_worker_finish_reclassifying_to_stopped():
    """A DRAINING worker's connection gets closed by stop_master() the
    same as any other -- but if it had already finished its last task and
    was just waiting to be closed, it should still end up STOPPED (not
    FAILED), same as the ordinary Phase 9.2.3 shutdown path, since
    stop_master() gives connection cleanup a bounded chance to run before
    forcing anything."""

    async def scenario():
        server = await async_server.start_master(port=0)
        host, port = server.sockets[0].getsockname()[:2]
        reader, writer = await asyncio.open_connection(host, port)
        conn = AsyncConnection(reader, writer)
        await send_request(conn, protocol.PING)
        await send_request(conn, protocol.REGISTER, {"worker_id": "worker-1", "host": "127.0.0.1", "port": 6000})
        await async_server.wait_for_workers(1)

        shutdown_response = await send_request(conn, protocol.SHUTDOWN, {"worker_id": "worker-1"})
        assert shutdown_response["type"] == protocol.SHUTDOWN_ACK

        # Nothing was ever assigned, so the master already closed this
        # connection in response to SHUTDOWN -- confirm that before
        # involving stop_master() at all.
        with pytest.raises(ConnectionError):
            await asyncio.wait_for(receive_message(conn), timeout=2)

        await async_server.stop_master()
        return async_server.worker_manager.get_worker("worker-1").status

    status = asyncio.run(scenario())
    assert status == WorkerStatus.STOPPED
