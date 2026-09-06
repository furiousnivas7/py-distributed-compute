"""Phase 7.7: full async fault-tolerance regression.

No changes to AsyncConnection, async_rpc, async_server, or async_worker --
this file only combines and stresses what those modules already do, under
realistic combinations of concurrency, failure, retry, stale results, and
worker replacement. Each test drives its scenario with asyncio.run() (no
pytest-asyncio dependency, matching the rest of the async suite).
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


async def run_multi_task_crashing_worker(
    host: str, port: int, worker_id: str, task_count: int, ready: asyncio.Event
) -> None:
    """Registers, reads `task_count` TASK messages without replying to any
    of them, signals `ready`, then disconnects."""
    reader, writer = await asyncio.open_connection(host, port)
    conn = AsyncConnection(reader, writer)
    await send_request(conn, protocol.PING)
    await send_request(conn, protocol.REGISTER, {"worker_id": worker_id, "host": "127.0.0.1", "port": 6000})

    seen = 0
    while seen < task_count:
        message = await receive_message(conn)
        if message["type"] == protocol.TASK:
            seen += 1

    ready.set()
    await conn.close()


async def run_delayed_reply_worker(
    host: str, port: int, worker_id: str, ready_event: asyncio.Event, release_event: asyncio.Event
) -> None:
    """Registers, receives exactly one TASK, signals `ready_event`, then
    waits on `release_event` before finally executing and replying. Never
    heartbeats, so it goes stale on its own without the connection dying --
    proving a reply that arrives long after reassignment is still ignored."""
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


def test_four_tasks_two_workers_genuinely_overlap():
    """Two rounds of two tasks each across two workers. Round 1 is gated
    with asyncio.Events to prove both tasks are RUNNING simultaneously
    (deterministic, not timing-based); round 2 runs freely."""

    async def scenario():
        server, host, port = await start_master_server()

        ready = {"worker-1": asyncio.Event(), "worker-2": asyncio.Event()}
        release = {"worker-1": asyncio.Event(), "worker-2": asyncio.Event()}

        async def gated_worker(worker_id: str) -> None:
            reader, writer = await asyncio.open_connection(host, port)
            conn = AsyncConnection(reader, writer)
            await send_request(conn, protocol.PING)
            await send_request(conn, protocol.REGISTER, {"worker_id": worker_id, "host": "127.0.0.1", "port": 6000})

            gated_once = False
            while True:
                try:
                    message = await receive_message(conn)
                except ConnectionError:
                    return
                if message["type"] != protocol.TASK:
                    continue

                if not gated_once:
                    gated_once = True
                    ready[worker_id].set()
                    await release[worker_id].wait()

                payload = message["payload"]
                result = execute_task(payload["task_type"], payload["task_payload"])
                response = build_message(
                    protocol.TASK_RESULT,
                    message["request_id"],
                    {"task_id": payload["task_id"], "attempt": payload.get("attempt", 1), **result},
                )
                await send_message(conn, response)

        worker_tasks = [asyncio.create_task(gated_worker(wid)) for wid in ("worker-1", "worker-2")]
        try:
            await async_server.wait_for_workers(2)

            for task_id, payload in [
                ("task-1", {"a": 1, "b": 1}),
                ("task-2", {"a": 2, "b": 2}),
                ("task-3", {"a": 3, "b": 3}),
                ("task-4", {"a": 4, "b": 4}),
            ]:
                async_server.scheduler.submit_task(task_id, "ADD", payload)

            drain_coro = asyncio.create_task(async_server.drain_pending_tasks())

            await asyncio.wait_for(ready["worker-1"].wait(), timeout=5)
            await asyncio.wait_for(ready["worker-2"].wait(), timeout=5)

            running = [t for t in async_server.scheduler.get_all_tasks() if t.status == TaskStatus.RUNNING]
            assert len(running) == 2

            release["worker-1"].set()
            release["worker-2"].set()

            return await drain_coro
        finally:
            server.close()
            await server.wait_closed()
            for t in worker_tasks:
                await stop_worker(t)

    responses = asyncio.run(scenario())
    assert len(responses) == 4
    assert all(r["payload"]["status"] == "success" for r in responses)


def test_worker_failure_during_concurrent_execution(monkeypatch):
    """worker-1 holds task-1 and task-3 when it crashes; worker-2 holds
    task-2 and task-4 and completes both regardless. worker-1's pair get
    requeued and finished by worker-2 afterward."""
    monkeypatch.setattr(async_server, "FAILURE_CHECK_INTERVAL", 0.05)
    monkeypatch.setattr(async_server, "HEARTBEAT_TIMEOUT", 1.0)

    async def scenario():
        server, host, port = await start_master_server()

        ready = asyncio.Event()
        worker1_task = asyncio.create_task(run_multi_task_crashing_worker(host, port, "worker-1", 2, ready))
        await async_server.wait_for_workers(1)
        worker2_task = asyncio.create_task(
            async_worker.run_worker(host, port, worker_id="worker-2", heartbeat_interval=0.1)
        )
        await async_server.wait_for_workers(2)

        stop_event = asyncio.Event()
        monitor_task = asyncio.create_task(async_server.failure_monitor(stop_event))

        try:
            for task_id, payload in [
                ("task-1", {"a": 1, "b": 1}),
                ("task-2", {"a": 2, "b": 2}),
                ("task-3", {"a": 3, "b": 3}),
                ("task-4", {"a": 4, "b": 4}),
            ]:
                async_server.scheduler.submit_task(task_id, "ADD", payload)

            # Deterministically put task-1/task-3 on worker-1 and
            # task-2/task-4 on worker-2 (assign_task only ever picks an
            # IDLE worker, so toggling status is what stacks two on one).
            async_server.scheduler.assign_task("task-1")
            async_server.scheduler.assign_task("task-2")
            async_server.worker_manager.update_status("worker-1", WorkerStatus.IDLE)
            async_server.scheduler.assign_task("task-3")
            async_server.worker_manager.update_status("worker-2", WorkerStatus.IDLE)
            async_server.scheduler.assign_task("task-4")

            tasks = {t.task_id: t for t in async_server.scheduler.get_all_tasks()}
            assert tasks["task-1"].assigned_worker_id == "worker-1"
            assert tasks["task-3"].assigned_worker_id == "worker-1"
            assert tasks["task-2"].assigned_worker_id == "worker-2"
            assert tasks["task-4"].assigned_worker_id == "worker-2"

            dispatches = {
                task_id: asyncio.create_task(async_server.dispatch_assigned_task(tasks[task_id]))
                for task_id in ("task-1", "task-2", "task-3", "task-4")
            }

            await asyncio.wait_for(ready.wait(), timeout=5)

            deadline = time.monotonic() + 5
            while (
                async_server.worker_manager.get_worker("worker-1").status != WorkerStatus.FAILED
                and time.monotonic() < deadline
            ):
                await asyncio.sleep(0.02)
            assert async_server.worker_manager.get_worker("worker-1").status == WorkerStatus.FAILED

            for d in dispatches.values():
                await d

            # worker-2's own tasks succeeded independently of worker-1's fate.
            assert tasks["task-2"].status == TaskStatus.COMPLETED
            assert tasks["task-4"].status == TaskStatus.COMPLETED
            assert tasks["task-1"].status == TaskStatus.PENDING
            assert tasks["task-3"].status == TaskStatus.PENDING

            responses = await async_server.drain_pending_tasks()
            assert all(r["payload"]["status"] == "success" for r in responses)

            assert tasks["task-1"].status == TaskStatus.COMPLETED
            assert tasks["task-3"].status == TaskStatus.COMPLETED
            assert tasks["task-1"].assigned_worker_id == "worker-2"
            assert tasks["task-3"].assigned_worker_id == "worker-2"
            assert tasks["task-1"].attempt == 2
            assert tasks["task-3"].attempt == 2
        finally:
            stop_event.set()
            await monitor_task
            await stop_worker(worker1_task)
            await stop_worker(worker2_task)
            server.close()
            await server.wait_closed()

    asyncio.run(scenario())


def test_stale_result_cannot_overwrite_newer_attempt_after_delay(monkeypatch):
    """The connection never dies here -- worker-1 just never heartbeats and
    holds its reply. The failure monitor reassigns to worker-2, which
    completes the task, and ONLY THEN does worker-1's stale attempt-1 reply
    finally arrive on its still-open connection. It must be ignored."""
    monkeypatch.setattr(async_server, "FAILURE_CHECK_INTERVAL", 0.05)
    monkeypatch.setattr(async_server, "HEARTBEAT_TIMEOUT", 0.3)

    async def scenario():
        server, host, port = await start_master_server()

        ready = asyncio.Event()
        release = asyncio.Event()
        worker1_task = asyncio.create_task(run_delayed_reply_worker(host, port, "worker-1", ready, release))
        await async_server.wait_for_workers(1)
        worker2_task = asyncio.create_task(
            async_worker.run_worker(host, port, worker_id="worker-2", heartbeat_interval=0.05)
        )
        await async_server.wait_for_workers(2)

        stop_event = asyncio.Event()
        monitor_task = asyncio.create_task(async_server.failure_monitor(stop_event))

        try:
            task = async_server.scheduler.submit_task("task-1", "ADD", {"a": 10, "b": 20})
            assigned = async_server.scheduler.assign_task("task-1")
            assert assigned.assigned_worker_id == "worker-1"
            assert assigned.attempt == 1

            dispatch1 = asyncio.create_task(async_server.dispatch_assigned_task(task))
            await asyncio.wait_for(ready.wait(), timeout=5)

            # worker-1's connection never dies here -- this is detected
            # purely by the failure monitor's heartbeat check, which marks
            # the worker FAILED and calls drain_pending_tasks() in the same
            # synchronous chain with no yield point in between. So unlike a
            # connection-death detection (which leaves a task observably
            # PENDING until something explicitly reassigns it), this task
            # jumps straight from RUNNING to reassigned/COMPLETED -- there's
            # no window to catch it mid-PENDING. Poll for the outcome
            # instead of an intermediate state.
            deadline = time.monotonic() + 5
            while task.status != TaskStatus.COMPLETED and time.monotonic() < deadline:
                await asyncio.sleep(0.02)

            assert async_server.worker_manager.get_worker("worker-1").status == WorkerStatus.FAILED
            assert task.status == TaskStatus.COMPLETED
            assert task.assigned_worker_id == "worker-2"
            assert task.attempt == 2

            # NOW let worker-1's long-delayed attempt-1 reply arrive.
            release.set()
            stale_response = await dispatch1

            assert stale_response["payload"]["attempt"] == 1
            assert task.status == TaskStatus.COMPLETED
            assert task.assigned_worker_id == "worker-2"
            assert task.attempt == 2
        finally:
            stop_event.set()
            await monitor_task
            await stop_worker(worker1_task)
            await stop_worker(worker2_task)
            server.close()
            await server.wait_closed()

    asyncio.run(scenario())


def test_task_succeeds_on_third_attempt_after_two_worker_failures(monkeypatch):
    """attempt 1 -> worker-1 -> FAIL, attempt 2 -> worker-2 -> FAIL,
    attempt 3 -> worker-3 -> SUCCESS. Final: COMPLETED, attempt == 3."""
    monkeypatch.setattr(async_server, "FAILURE_CHECK_INTERVAL", 0.05)
    monkeypatch.setattr(async_server, "HEARTBEAT_TIMEOUT", 1.0)

    async def scenario():
        server, host, port = await start_master_server()

        ready1 = asyncio.Event()
        ready2 = asyncio.Event()
        worker1_task = asyncio.create_task(run_multi_task_crashing_worker(host, port, "worker-1", 1, ready1))
        await async_server.wait_for_workers(1)
        worker2_task = asyncio.create_task(run_multi_task_crashing_worker(host, port, "worker-2", 1, ready2))
        await async_server.wait_for_workers(2)
        worker3_task = asyncio.create_task(
            async_worker.run_worker(host, port, worker_id="worker-3", heartbeat_interval=0.1)
        )
        await async_server.wait_for_workers(3)

        stop_event = asyncio.Event()
        monitor_task = asyncio.create_task(async_server.failure_monitor(stop_event))

        try:
            task = async_server.scheduler.submit_task("task-1", "ADD", {"a": 5, "b": 7})

            assigned1 = async_server.scheduler.assign_next_pending_task()
            assert assigned1.assigned_worker_id == "worker-1"
            assert assigned1.attempt == 1
            dispatch1 = asyncio.create_task(async_server.dispatch_assigned_task(task))
            await asyncio.wait_for(ready1.wait(), timeout=5)

            deadline = time.monotonic() + 5
            while (
                async_server.worker_manager.get_worker("worker-1").status != WorkerStatus.FAILED
                and time.monotonic() < deadline
            ):
                await asyncio.sleep(0.02)
            await dispatch1
            assert task.status == TaskStatus.PENDING

            assigned2 = async_server.scheduler.assign_next_pending_task()
            assert assigned2.assigned_worker_id == "worker-2"
            assert assigned2.attempt == 2
            dispatch2 = asyncio.create_task(async_server.dispatch_assigned_task(task))
            await asyncio.wait_for(ready2.wait(), timeout=5)

            deadline = time.monotonic() + 5
            while (
                async_server.worker_manager.get_worker("worker-2").status != WorkerStatus.FAILED
                and time.monotonic() < deadline
            ):
                await asyncio.sleep(0.02)
            await dispatch2
            assert task.status == TaskStatus.PENDING

            assigned3 = async_server.scheduler.assign_next_pending_task()
            assert assigned3.assigned_worker_id == "worker-3"
            assert assigned3.attempt == 3
            response3 = await async_server.dispatch_assigned_task(task)

            assert response3["payload"]["status"] == "success"
            assert response3["payload"]["attempt"] == 3
            assert task.status == TaskStatus.COMPLETED
            assert task.attempt == 3
        finally:
            stop_event.set()
            await monitor_task
            await stop_worker(worker1_task)
            await stop_worker(worker2_task)
            await stop_worker(worker3_task)
            server.close()
            await server.wait_closed()

    asyncio.run(scenario())


def test_retry_exhaustion_leaves_a_spare_idle_worker_untouched(monkeypatch):
    """Same exhaustion guarantee as test_async_fault_tolerance.py's version,
    but this time a 4th worker sits IDLE the whole time -- proving that
    once a task is FAILED (exhausted), it's never assigned again even
    though an available worker exists to take it."""
    assert MAX_TASK_ATTEMPTS == 3, "test assumes the current default of 3"
    monkeypatch.setattr(async_server, "FAILURE_CHECK_INTERVAL", 0.05)
    monkeypatch.setattr(async_server, "HEARTBEAT_TIMEOUT", 1.0)

    async def scenario():
        server, host, port = await start_master_server()

        crashing_tasks = []
        ready_events = []
        for i in range(1, MAX_TASK_ATTEMPTS + 1):
            ready = asyncio.Event()
            ready_events.append(ready)
            crashing_tasks.append(
                asyncio.create_task(run_multi_task_crashing_worker(host, port, f"worker-{i}", 1, ready))
            )
            await async_server.wait_for_workers(i)

        spare_task = asyncio.create_task(
            async_worker.run_worker(host, port, worker_id="spare-worker", heartbeat_interval=0.1)
        )
        await async_server.wait_for_workers(MAX_TASK_ATTEMPTS + 1)

        stop_event = asyncio.Event()
        monitor_task = asyncio.create_task(async_server.failure_monitor(stop_event))

        try:
            task = async_server.scheduler.submit_task("task-1", "ADD", {"a": 1, "b": 1})

            for i in range(1, MAX_TASK_ATTEMPTS + 1):
                worker_id = f"worker-{i}"
                assigned = async_server.scheduler.assign_task("task-1")
                assert assigned.assigned_worker_id == worker_id
                assert assigned.attempt == i

                dispatch = asyncio.create_task(async_server.dispatch_assigned_task(task))
                await asyncio.wait_for(ready_events[i - 1].wait(), timeout=5)

                deadline = time.monotonic() + 5
                while (
                    async_server.worker_manager.get_worker(worker_id).status != WorkerStatus.FAILED
                    and time.monotonic() < deadline
                ):
                    await asyncio.sleep(0.02)
                assert async_server.worker_manager.get_worker(worker_id).status == WorkerStatus.FAILED
                await dispatch

            assert task.status == TaskStatus.FAILED
            assert task.attempt == MAX_TASK_ATTEMPTS
            assert task.attempt != MAX_TASK_ATTEMPTS + 1

            # spare-worker was IDLE and available the entire time, yet the
            # exhausted task must never be handed to it.
            assert async_server.worker_manager.get_worker("spare-worker").status == WorkerStatus.IDLE
            assert async_server.scheduler.assign_next_pending_task() is None
            assert task.assigned_worker_id is None
        finally:
            stop_event.set()
            await monitor_task
            for t in crashing_tasks:
                await stop_worker(t)
            await stop_worker(spare_task)
            server.close()
            await server.wait_closed()

    asyncio.run(scenario())


def test_healthy_workers_continue_after_one_is_isolated(monkeypatch):
    """Once worker-1 is FAILED, it's isolated from all future assignment --
    the rest of a larger workload flows only to the healthy workers."""
    monkeypatch.setattr(async_server, "FAILURE_CHECK_INTERVAL", 0.05)
    monkeypatch.setattr(async_server, "HEARTBEAT_TIMEOUT", 1.0)

    async def scenario():
        server, host, port = await start_master_server()

        ready = asyncio.Event()
        worker1_task = asyncio.create_task(run_multi_task_crashing_worker(host, port, "worker-1", 1, ready))
        await async_server.wait_for_workers(1)
        worker2_task = asyncio.create_task(
            async_worker.run_worker(host, port, worker_id="worker-2", heartbeat_interval=0.1)
        )
        worker3_task = asyncio.create_task(
            async_worker.run_worker(host, port, worker_id="worker-3", heartbeat_interval=0.1)
        )
        await async_server.wait_for_workers(3)

        stop_event = asyncio.Event()
        monitor_task = asyncio.create_task(async_server.failure_monitor(stop_event))

        try:
            task_a = async_server.scheduler.submit_task("task-a", "ADD", {"a": 1, "b": 1})
            assigned_a = async_server.scheduler.assign_task("task-a")
            assert assigned_a.assigned_worker_id == "worker-1"

            dispatch_a = asyncio.create_task(async_server.dispatch_assigned_task(task_a))
            await asyncio.wait_for(ready.wait(), timeout=5)

            deadline = time.monotonic() + 5
            while (
                async_server.worker_manager.get_worker("worker-1").status != WorkerStatus.FAILED
                and time.monotonic() < deadline
            ):
                await asyncio.sleep(0.02)
            assert async_server.worker_manager.get_worker("worker-1").status == WorkerStatus.FAILED
            await dispatch_a

            for i in range(1, 7):
                async_server.scheduler.submit_task(f"task-{i}", "ADD", {"a": i, "b": i})

            responses = await async_server.drain_pending_tasks()
            assert all(r["payload"]["status"] == "success" for r in responses)

            all_tasks = async_server.scheduler.get_all_tasks()
            assert all(t.status == TaskStatus.COMPLETED for t in all_tasks)
            workers_used = {t.assigned_worker_id for t in all_tasks}
            assert "worker-1" not in workers_used
            assert workers_used == {"worker-2", "worker-3"}
        finally:
            stop_event.set()
            await monitor_task
            await stop_worker(worker1_task)
            await stop_worker(worker2_task)
            await stop_worker(worker3_task)
            server.close()
            await server.wait_closed()

    asyncio.run(scenario())


def test_heartbeats_and_tasks_interleave_without_corruption():
    """Several tasks dispatched with pauses in between (to let multiple
    heartbeats land) on the same connection. Every result must still come
    back correct -- proof the framing survives the interleaving."""

    async def scenario():
        server, host, port = await start_master_server()
        worker_task = asyncio.create_task(
            async_worker.run_worker(host, port, worker_id="worker-1", heartbeat_interval=0.03)
        )
        try:
            await async_server.wait_for_workers(1)

            results = []
            for i, (task_type, a, b) in enumerate([("ADD", 1, 1), ("MULTIPLY", 3, 3), ("ADD", 10, 20)], start=1):
                await asyncio.sleep(0.1)  # let several heartbeats land first
                async_server.scheduler.submit_task(f"task-{i}", task_type, {"a": a, "b": b})
                results.extend(await async_server.drain_pending_tasks())

            last_heartbeat = async_server.worker_manager.get_worker("worker-1").last_heartbeat
            return results, last_heartbeat
        finally:
            await stop_worker(worker_task)
            server.close()
            await server.wait_closed()

    results, last_heartbeat = asyncio.run(scenario())
    assert [r["payload"]["result"] for r in results] == [2, 9, 30]
    assert all(r["payload"]["status"] == "success" for r in results)
    assert last_heartbeat is not None


def test_both_failure_detection_paths_converge_to_failed_and_requeue(monkeypatch):
    """worker-a's connection actually dies; worker-b's connection stays
    open but its heartbeats stop. Both converge on FAILED + requeue, and a
    third, untouched worker mops up both tasks."""
    monkeypatch.setattr(async_server, "FAILURE_CHECK_INTERVAL", 0.05)
    monkeypatch.setattr(async_server, "HEARTBEAT_TIMEOUT", 0.3)

    async def scenario():
        server, host, port = await start_master_server()

        ready_a = asyncio.Event()
        worker_a_task = asyncio.create_task(run_multi_task_crashing_worker(host, port, "worker-a", 1, ready_a))
        await async_server.wait_for_workers(1)

        # worker-b: connection stays open the whole test (never disconnects,
        # never replies), and it never sends a heartbeat either -- a
        # genuinely stuck worker, not merely one with heartbeats turned off.
        # (Disabling heartbeat_interval alone isn't enough: a healthy
        # serve_tasks loop would just reply to its task almost instantly,
        # completing it before staleness could ever matter.)
        ready_b = asyncio.Event()
        release_b = asyncio.Event()  # intentionally never set
        worker_b_task = asyncio.create_task(run_delayed_reply_worker(host, port, "worker-b", ready_b, release_b))
        await async_server.wait_for_workers(2)

        rescue_task = asyncio.create_task(
            async_worker.run_worker(host, port, worker_id="rescue-worker", heartbeat_interval=0.05)
        )
        await async_server.wait_for_workers(3)

        stop_event = asyncio.Event()
        monitor_task = asyncio.create_task(async_server.failure_monitor(stop_event))

        try:
            task_a = async_server.scheduler.submit_task("task-a", "ADD", {"a": 1, "b": 1})
            task_b = async_server.scheduler.submit_task("task-b", "ADD", {"a": 2, "b": 2})

            assigned_a = async_server.scheduler.assign_task("task-a")
            assigned_b = async_server.scheduler.assign_task("task-b")
            assert assigned_a.assigned_worker_id == "worker-a"
            assert assigned_b.assigned_worker_id == "worker-b"

            dispatch_a = asyncio.create_task(async_server.dispatch_assigned_task(task_a))
            dispatch_b = asyncio.create_task(async_server.dispatch_assigned_task(task_b))

            await asyncio.wait_for(ready_a.wait(), timeout=5)
            await asyncio.wait_for(ready_b.wait(), timeout=5)

            deadline = time.monotonic() + 5
            while (
                async_server.worker_manager.get_worker("worker-a").status != WorkerStatus.FAILED
                or async_server.worker_manager.get_worker("worker-b").status != WorkerStatus.FAILED
            ) and time.monotonic() < deadline:
                await asyncio.sleep(0.02)

            assert async_server.worker_manager.get_worker("worker-a").status == WorkerStatus.FAILED
            assert async_server.worker_manager.get_worker("worker-b").status == WorkerStatus.FAILED
            assert "worker-a" not in async_server.connections  # connection actually died
            assert "worker-b" in async_server.connections  # connection still open, just stale

            # dispatch_a completes on its own: worker-a's connection really
            # died, so handle_worker_connection's cleanup fails its pending
            # future with ConnectionError. dispatch_b never will, though --
            # worker-b never sends anything else on its still-open
            # connection (that's the point), so nothing ever resolves its
            # future. It's cancelled in the finally block instead of awaited.
            await dispatch_a

            # Whichever failure the monitor's own tick detected may already
            # have been auto-reassigned to rescue-worker in that same
            # synchronous step (heartbeat-based detection drains
            # immediately; connection-death detection doesn't) -- so rather
            # than assume a specific intermediate state for task_a/task_b,
            # just drive both to their deterministic final outcome.
            deadline = time.monotonic() + 5
            while (
                task_a.status != TaskStatus.COMPLETED or task_b.status != TaskStatus.COMPLETED
            ) and time.monotonic() < deadline:
                await async_server.drain_pending_tasks()
                await asyncio.sleep(0.02)

            assert task_a.status == TaskStatus.COMPLETED
            assert task_b.status == TaskStatus.COMPLETED
            assert task_a.assigned_worker_id == "rescue-worker"
            assert task_b.assigned_worker_id == "rescue-worker"
        finally:
            dispatch_b.cancel()
            stop_event.set()
            await monitor_task
            await stop_worker(worker_a_task)
            await stop_worker(worker_b_task)
            await stop_worker(rescue_task)
            server.close()
            await server.wait_closed()

    asyncio.run(scenario())


def test_mixed_workload_survives_worker_failure(monkeypatch):
    """worker-1: ADD, worker-2: MULTIPLY, worker-1: ADD, then worker-2
    crashes on its next task. Remaining and requeued work all lands on
    worker-1, the only one left."""
    monkeypatch.setattr(async_server, "FAILURE_CHECK_INTERVAL", 0.05)
    monkeypatch.setattr(async_server, "HEARTBEAT_TIMEOUT", 1.0)

    async def scenario():
        server, host, port = await start_master_server()
        worker1_task = asyncio.create_task(
            async_worker.run_worker(host, port, worker_id="worker-1", heartbeat_interval=0.1)
        )
        await async_server.wait_for_workers(1)

        ready2 = asyncio.Event()
        worker2_task = asyncio.create_task(run_multi_task_crashing_worker(host, port, "worker-2", 1, ready2))
        await async_server.wait_for_workers(2)

        stop_event = asyncio.Event()
        monitor_task = asyncio.create_task(async_server.failure_monitor(stop_event))

        try:
            async_server.scheduler.submit_task("add-1", "ADD", {"a": 1, "b": 1})
            async_server.scheduler.submit_task("multiply-1", "MULTIPLY", {"a": 2, "b": 2})
            t_add1 = async_server.scheduler.assign_next_pending_task()
            t_mul1 = async_server.scheduler.assign_next_pending_task()
            assert t_add1.assigned_worker_id == "worker-1"
            assert t_mul1.assigned_worker_id == "worker-2"

            response_add1 = await async_server.dispatch_assigned_task(t_add1)
            assert response_add1["payload"]["result"] == 2

            async_server.scheduler.submit_task("add-2", "ADD", {"a": 5, "b": 5})
            t_add2 = async_server.scheduler.assign_next_pending_task()
            assert t_add2.assigned_worker_id == "worker-1"
            response_add2 = await async_server.dispatch_assigned_task(t_add2)
            assert response_add2["payload"]["result"] == 10

            dispatch_mul1 = asyncio.create_task(async_server.dispatch_assigned_task(t_mul1))
            await asyncio.wait_for(ready2.wait(), timeout=5)

            deadline = time.monotonic() + 5
            while (
                async_server.worker_manager.get_worker("worker-2").status != WorkerStatus.FAILED
                and time.monotonic() < deadline
            ):
                await asyncio.sleep(0.02)
            assert async_server.worker_manager.get_worker("worker-2").status == WorkerStatus.FAILED
            await dispatch_mul1
            assert t_mul1.status == TaskStatus.PENDING

            async_server.scheduler.submit_task("add-3", "ADD", {"a": 100, "b": 1})
            responses = await async_server.drain_pending_tasks()

            assert all(r["payload"]["status"] == "success" for r in responses)
            assert t_mul1.status == TaskStatus.COMPLETED
            assert t_mul1.assigned_worker_id == "worker-1"
            add3 = async_server.scheduler.get_task("add-3")
            assert add3.status == TaskStatus.COMPLETED
            assert add3.assigned_worker_id == "worker-1"
        finally:
            stop_event.set()
            await monitor_task
            await stop_worker(worker1_task)
            await stop_worker(worker2_task)
            server.close()
            await server.wait_closed()

    asyncio.run(scenario())


def test_stress_three_workers_twenty_tasks_with_one_failure(monkeypatch):
    """20 tasks, 3 workers, one crashes on its first task. Everything must
    still complete via the two survivors."""
    monkeypatch.setattr(async_server, "FAILURE_CHECK_INTERVAL", 0.05)
    monkeypatch.setattr(async_server, "HEARTBEAT_TIMEOUT", 1.0)

    async def scenario():
        server, host, port = await start_master_server()

        ready = asyncio.Event()
        worker1_task = asyncio.create_task(run_multi_task_crashing_worker(host, port, "worker-1", 1, ready))
        await async_server.wait_for_workers(1)
        worker2_task = asyncio.create_task(
            async_worker.run_worker(host, port, worker_id="worker-2", heartbeat_interval=0.1)
        )
        worker3_task = asyncio.create_task(
            async_worker.run_worker(host, port, worker_id="worker-3", heartbeat_interval=0.1)
        )
        await async_server.wait_for_workers(3)

        stop_event = asyncio.Event()
        monitor_task = asyncio.create_task(async_server.failure_monitor(stop_event))

        try:
            for i in range(20):
                async_server.scheduler.submit_task(f"task-{i}", "ADD", {"a": i, "b": i})

            drain_task = asyncio.create_task(async_server.drain_pending_tasks())

            await asyncio.wait_for(ready.wait(), timeout=5)
            deadline = time.monotonic() + 5
            while (
                async_server.worker_manager.get_worker("worker-1").status != WorkerStatus.FAILED
                and time.monotonic() < deadline
            ):
                await asyncio.sleep(0.02)
            assert async_server.worker_manager.get_worker("worker-1").status == WorkerStatus.FAILED

            await drain_task
            await async_server.drain_pending_tasks()  # safety net for any leftover round

            all_tasks = async_server.scheduler.get_all_tasks()
            assert len(all_tasks) == 20
            assert all(t.status == TaskStatus.COMPLETED for t in all_tasks)
            assert all(t.assigned_worker_id in ("worker-2", "worker-3") for t in all_tasks)
        finally:
            stop_event.set()
            await monitor_task
            await stop_worker(worker1_task)
            await stop_worker(worker2_task)
            await stop_worker(worker3_task)
            server.close()
            await server.wait_closed()

    asyncio.run(scenario())
