"""Phase 8.9: centralized task ownership and response routing.

Phase 8.8 fixed cross-job contamination by scoping each caller's OWN
assignment loop to its OWN task_ids (drain_tasks_for). That's a real fix,
but it leaves a structural hazard the user's review called out: every
caller still runs its own independent assign-and-dispatch loop against the
same shared scheduler, so nothing stops two independently-written call
sites (or a caller and the failure monitor) from drifting out of sync
again in the future -- the safety property lives in each caller remembering
to scope itself, not in the architecture.

This phase replaces that with a single authoritative dispatcher
(dispatcher_loop, see master/async_server.py) that is the ONLY thing that
ever calls scheduler.assign_next_pending_task() once it's running,
plus one response registry (_task_responses / _task_futures) keyed by
task_id that every dispatch path -- old and new -- writes into via
dispatch_assigned_task's record_if_terminal. wait_for_tasks() is the new
caller-facing API: it starts the dispatcher if needed and then only ever
*waits* on its own task_ids, never assigns or dispatches anything itself.
There is no caller-specific response list left for a response to go
missing from.

Writing these tests surfaced a real, previously-latent bug in
record_if_terminal: mark_worker_failed_and_requeue() clears
task.assigned_worker_id as a side effect (via requeue_tasks_for_worker),
and record_if_terminal used to re-check is_current_attempt() AFTER that
mutation -- so a worker-unreachable failure that exhausted the task's
final retry attempt (mark_worker_failed_and_requeue's requeue call marks
it FAILED, not PENDING, once attempts are exhausted) would look "stale"
against the very state change it had just made itself, and the terminal
FAILED response would never be recorded. wait_for_tasks() would then hang
until its 30s timeout instead of returning the failure. Fixed by capturing
`is_current_attempt()` ONCE, before any mutation, and threading that
boolean through both the requeue guard and the record decision (there is
no `await` between them, so nothing else could legitimately change
"current-ness" in between). test_dispatcher_reports_failure_after_retries_exhausted_via_connection_death
below is the regression test for exactly this.

Every test here drives the dispatcher for real (ensure_dispatcher_running/
wait_for_tasks), not the manual assign_next_pending_task/dispatch_assigned_task
plumbing the pre-8.9 fault-tolerance tests use -- the whole point is to
prove the dispatcher itself, as an architecture, gets this right without
any caller having to know about scoping, retries, or staleness at all.
"""

import asyncio
import time

import pytest

from common.models import TaskStatus, WorkerStatus
from master import async_server, rpc_handler
from master.scheduler import MAX_TASK_ATTEMPTS
from rpc import protocol
from rpc.async_connection import AsyncConnection
from rpc.async_rpc import receive_message, send_message, send_request
from rpc.protocol import build_message
from worker import async_worker
from worker.executor import execute_task


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


async def run_single_crash_worker(host: str, port: int, worker_id: str, ready: asyncio.Event) -> None:
    """Registers, receives exactly one TASK message without replying,
    signals `ready`, then disconnects -- a genuine connection-death
    failure, the same shape dispatch_assigned_task's except branch and the
    exhausted-retries bug above both hinge on."""
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


async def run_delayed_reply_worker(
    host: str, port: int, worker_id: str, ready_event: asyncio.Event, release_event: asyncio.Event
) -> None:
    """Registers, receives one TASK, signals `ready_event`, then waits on
    `release_event` before executing and replying -- a heartbeat-timeout
    style staleness, not a connection death, so the connection stays open
    for a genuinely late reply to arrive on."""
    reader, writer = await asyncio.open_connection(host, port)
    conn = AsyncConnection(reader, writer)
    await send_request(conn, protocol.PING)
    await send_request(conn, protocol.REGISTER, {"worker_id": worker_id, "host": "127.0.0.1", "port": 6000})

    message = await receive_message(conn)
    assert message["type"] == protocol.TASK
    ready_event.set()
    await release_event.wait()

    payload = message["payload"]
    result = execute_task(payload["task_type"], payload["task_payload"])
    response = build_message(
        protocol.TASK_RESULT,
        message["request_id"],
        {"task_id": payload["task_id"], "attempt": payload.get("attempt", 1), **result},
    )
    await send_message(conn, response)
    await conn.close()


def test_dispatcher_starts_lazily_and_is_reused_across_calls():
    """The dispatcher must not exist until something actually needs it
    (existing tests that never touch wait_for_tasks must stay unaffected),
    and once started, a second wait_for_tasks() call must reuse the SAME
    dispatcher task rather than spawning a competing one."""

    async def scenario():
        server, host, port = await start_master_server()
        worker_task = asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-1"))
        try:
            await async_server.wait_for_workers(1)
            assert not async_server.is_dispatcher_running()

            task_a = async_server.scheduler.submit_task("a", "ADD", {"a": 1, "b": 1})
            await async_server.wait_for_tasks({task_a.task_id})
            assert async_server.is_dispatcher_running()
            dispatcher_after_first = async_server._dispatcher_task

            task_b = async_server.scheduler.submit_task("b", "ADD", {"a": 2, "b": 2})
            await async_server.wait_for_tasks({task_b.task_id})
            assert async_server._dispatcher_task is dispatcher_after_first
        finally:
            await stop_worker(worker_task)
            server.close()
            await server.wait_closed()

    asyncio.run(scenario())


def test_dispatcher_recovers_task_after_worker_connection_death():
    """A worker crashes (connection death) mid-task; a rescue worker joins
    afterward. wait_for_tasks() -- not any manual dispatch call -- must
    return the correct final result once the dispatcher itself notices the
    requeued task and reassigns it."""

    async def scenario():
        server, host, port = await start_master_server()
        ready = asyncio.Event()
        crashing_task = asyncio.create_task(run_single_crash_worker(host, port, "worker-1", ready))
        await async_server.wait_for_workers(1)

        try:
            task = async_server.scheduler.submit_task("doomed", "ADD", {"a": 10, "b": 5})
            waiter = asyncio.create_task(async_server.wait_for_tasks({task.task_id}))

            await asyncio.wait_for(ready.wait(), timeout=5)
            rescue_task = asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-2"))

            [response] = await asyncio.wait_for(waiter, timeout=5)
            return response
        finally:
            await stop_worker(crashing_task)
            await stop_worker(rescue_task)
            server.close()
            await server.wait_closed()

    response = asyncio.run(scenario())
    assert response["payload"]["status"] == "success"
    assert response["payload"]["result"] == 15
    assert response["payload"]["attempt"] == 2


def test_dispatcher_reports_failure_after_retries_exhausted_via_connection_death():
    """Regression test for the record_if_terminal bug found while writing
    this phase's tests: MAX_TASK_ATTEMPTS crashing workers in a row, each a
    genuine connection death. The task must end up FAILED and
    wait_for_tasks() must return the terminal WORKER_UNREACHABLE response
    -- NOT hang until its timeout -- even though the very call that
    exhausts the last attempt is the same call that flips the task to
    FAILED and clears its assigned_worker_id."""

    assert MAX_TASK_ATTEMPTS == 3, "test assumes the current default of 3"

    async def scenario():
        server, host, port = await start_master_server()
        task = async_server.scheduler.submit_task("doomed", "ADD", {"a": 1, "b": 1})
        waiter = asyncio.create_task(async_server.wait_for_tasks({task.task_id}, timeout=10))

        worker_tasks = []
        try:
            for i in range(1, MAX_TASK_ATTEMPTS + 1):
                ready = asyncio.Event()
                worker_id = f"worker-{i}"
                # No wait_for_worker_id() step here: this worker registers,
                # immediately gets dispatched the task, and disconnects --
                # all on loopback with no real I/O latency, so that whole
                # window can close between two 10ms polls of `connections`
                # and be missed entirely. ready.wait() is the correct sync
                # point: it can only fire after registration AND dispatch
                # have both already happened.
                worker_tasks.append(asyncio.create_task(run_single_crash_worker(host, port, worker_id, ready)))
                await asyncio.wait_for(ready.wait(), timeout=5)

            [response] = await asyncio.wait_for(waiter, timeout=5)
            return response, task
        finally:
            for t in worker_tasks:
                await stop_worker(t)
            server.close()
            await server.wait_closed()

    response, task = asyncio.run(scenario())
    assert task.status == TaskStatus.FAILED
    assert task.attempt == MAX_TASK_ATTEMPTS
    assert response["type"] == protocol.ERROR
    assert response["payload"]["code"] == "WORKER_UNREACHABLE"
    assert response["payload"]["task_id"] == "doomed"
    assert response["payload"]["attempt"] == MAX_TASK_ATTEMPTS


def test_late_stale_response_after_dispatcher_reassignment_is_not_returned():
    """worker-1 holds the task and goes quiet (connection stays open, no
    heartbeat) while worker-2 completes the dispatcher-driven retry;
    worker-1's very late attempt-1 reply must never overwrite the registry
    entry the winning attempt-2 already wrote, and wait_for_tasks() must
    return attempt-2's result to every caller regardless of when it was
    called relative to the late reply."""

    async def scenario():
        server, host, port = await start_master_server()
        ready = asyncio.Event()
        release = asyncio.Event()
        worker1_task = asyncio.create_task(run_delayed_reply_worker(host, port, "worker-1", ready, release))
        await async_server.wait_for_workers(1)
        worker2_task = asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-2"))
        await async_server.wait_for_workers(2)

        try:
            task = async_server.scheduler.submit_task("flaky", "ADD", {"a": 7, "b": 8})
            waiter = asyncio.create_task(async_server.wait_for_tasks({task.task_id}))

            await asyncio.wait_for(ready.wait(), timeout=5)
            async_server.worker_manager.get_worker("worker-1").last_heartbeat = time.time() - 10
            stale = async_server.worker_manager.get_stale_workers(async_server.HEARTBEAT_TIMEOUT)
            assert any(w.worker_id == "worker-1" for w in stale)
            async_server.scheduler.requeue_tasks_for_worker("worker-1")

            [response] = await asyncio.wait_for(waiter, timeout=5)
            assert response["payload"]["attempt"] == 2

            # A caller asking AFTER completion must get the same registry
            # entry, not re-dispatch or block.
            [again] = await asyncio.wait_for(async_server.wait_for_tasks({task.task_id}), timeout=1)
            assert again == response

            # NOW let worker-1's stale attempt-1 reply land.
            release.set()
            await asyncio.sleep(0.1)

            [still] = await asyncio.wait_for(async_server.wait_for_tasks({task.task_id}), timeout=1)
            return response, still, task
        finally:
            await stop_worker(worker1_task)
            await stop_worker(worker2_task)
            server.close()
            await server.wait_closed()

    response, still, task = asyncio.run(scenario())
    assert response["payload"]["result"] == 15
    assert still == response, "the stale attempt-1 reply must not have overwritten the registry"
    assert task.status == TaskStatus.COMPLETED
    assert task.assigned_worker_id == "worker-2"


def test_duplicate_terminal_response_recording_is_idempotent():
    """A duplicate recording for the same (already-terminal) task_id --
    the shape a buggy/duplicating transport would produce -- must not
    corrupt the registry or break a future waiting on it."""

    async def scenario():
        server, host, port = await start_master_server()
        worker_task = asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-1"))
        try:
            await async_server.wait_for_workers(1)
            task = async_server.scheduler.submit_task("dup", "ADD", {"a": 3, "b": 4})
            [response] = await async_server.wait_for_tasks({task.task_id})
            assert response["payload"]["result"] == 7

            # Simulate a duplicate terminal delivery for the same task_id.
            async_server._record_task_response(task.task_id, response)

            [again] = await async_server.wait_for_tasks({task.task_id})
            return response, again
        finally:
            await stop_worker(worker_task)
            server.close()
            await server.wait_closed()

    response, again = asyncio.run(scenario())
    assert again == response


def test_concurrent_callers_share_one_dispatcher_and_get_disjoint_results():
    """Two callers submit and await entirely separate task_ids at the same
    time. Both must get correct, uncorrupted results, and both must have
    been served by the exact same dispatcher instance (proving there is
    only ever one authoritative assignment loop, not one per caller)."""

    async def scenario():
        server, host, port = await start_master_server()
        worker_tasks = [
            asyncio.create_task(async_worker.run_worker(host, port, worker_id=f"worker-{i}")) for i in range(1, 4)
        ]
        try:
            await async_server.wait_for_workers(3)

            tasks_a = {async_server.scheduler.submit_task(f"a-{i}", "ADD", {"a": i, "b": 1}).task_id for i in range(3)}
            tasks_b = {
                async_server.scheduler.submit_task(f"b-{i}", "MULTIPLY", {"a": i, "b": 10}).task_id for i in range(3)
            }

            waiter_a = asyncio.create_task(async_server.wait_for_tasks(tasks_a))
            waiter_b = asyncio.create_task(async_server.wait_for_tasks(tasks_b))

            responses_a, responses_b = await asyncio.gather(waiter_a, waiter_b)
            dispatcher = async_server._dispatcher_task
            return responses_a, responses_b, dispatcher
        finally:
            server.close()
            await server.wait_closed()
            for t in worker_tasks:
                await stop_worker(t)

    responses_a, responses_b, dispatcher = asyncio.run(scenario())
    assert sorted(r["payload"]["result"] for r in responses_a) == [1, 2, 3]
    assert sorted(r["payload"]["result"] for r in responses_b) == [0, 10, 20]
    assert dispatcher is not None
    assert async_server._dispatcher_task is dispatcher


def test_stop_dispatcher_then_restart_still_dispatches_correctly():
    """stop_dispatcher() must actually stop the loop (is_dispatcher_running()
    goes False), and a later call needing dispatch must transparently start
    a fresh one that works exactly as before -- no leftover state from the
    stopped dispatcher should corrupt the new one."""

    async def scenario():
        server, host, port = await start_master_server()
        worker_task = asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-1"))
        try:
            await async_server.wait_for_workers(1)

            first = async_server.scheduler.submit_task("first", "ADD", {"a": 1, "b": 2})
            [first_response] = await async_server.wait_for_tasks({first.task_id})
            assert async_server.is_dispatcher_running()

            await async_server.stop_dispatcher()
            assert not async_server.is_dispatcher_running()

            second = async_server.scheduler.submit_task("second", "ADD", {"a": 5, "b": 6})
            [second_response] = await async_server.wait_for_tasks({second.task_id})
            assert async_server.is_dispatcher_running()
            return first_response, second_response
        finally:
            await stop_worker(worker_task)
            server.close()
            await server.wait_closed()

    first_response, second_response = asyncio.run(scenario())
    assert first_response["payload"]["result"] == 3
    assert second_response["payload"]["result"] == 11


def test_failure_monitor_defers_to_running_dispatcher_instead_of_double_dispatching(monkeypatch):
    """With the centralized dispatcher active, failure_monitor must only
    requeue a dead worker's tasks and leave dispatching to the dispatcher
    -- it must NOT also call drain_pending_tasks() itself, which would
    resurrect exactly the two-independent-drainers race Phase 8.8 fixed.
    Heartbeat timeout (connection stays open) is used here specifically so
    failure_monitor, not dispatch_assigned_task's own except branch, is
    the one detecting and requeuing the failure."""
    monkeypatch.setattr(async_server, "FAILURE_CHECK_INTERVAL", 0.05)
    monkeypatch.setattr(async_server, "HEARTBEAT_TIMEOUT", 1.0)

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

        stop_event = asyncio.Event()
        monitor_task = asyncio.create_task(async_server.failure_monitor(stop_event))
        try:
            task = async_server.scheduler.submit_task("watched", "ADD", {"a": 2, "b": 3})
            waiter = asyncio.create_task(async_server.wait_for_tasks({task.task_id}))

            await asyncio.wait_for(ready.wait(), timeout=5)
            # Don't touch last_heartbeat manually -- let failure_monitor's
            # own polling loop detect the real timeout, since the point is
            # to prove ITS behavior once a dispatcher already exists.
            [response] = await asyncio.wait_for(waiter, timeout=5)
            release.set()
            await asyncio.sleep(0.1)
            return response, task
        finally:
            stop_event.set()
            await monitor_task
            await stop_worker(worker1_task)
            await stop_worker(worker2_task)
            server.close()
            await server.wait_closed()

    response, task = asyncio.run(scenario())
    assert response["payload"]["status"] == "success"
    assert response["payload"]["result"] == 5
    assert response["payload"]["attempt"] == 2
    assert task.status == TaskStatus.COMPLETED
