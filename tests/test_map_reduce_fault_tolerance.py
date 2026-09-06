"""Phase 8.7: MapReduce fault tolerance, over real async workers.

Two families of failure are tested, deliberately differently:

- Connection-death failures (a worker crashes / closes its connection):
  detected entirely inside dispatch_assigned_task, with no background task
  needed. These are safe to drive through the real run_map_reduce()
  pipeline directly.

- Heartbeat-timeout / stale-result failures (connection stays open, worker
  just goes quiet or replies late): normally detected by failure_monitor
  running as an ongoing background task. But failure_monitor calls
  drain_pending_tasks() itself, and drain_pending_tasks() is NOT safe for
  concurrent callers (see Phase 8.6's job-isolation test) -- running it
  continuously in the background WHILE run_map_reduce()'s own dispatch()
  is also mid-flight would race the two. So these tests force staleness
  and requeue directly (the same "force staleness" technique used in
  Phases 6.5/7.5/7.7), without a live failure_monitor task, and drive the
  pipeline's building blocks directly instead of the top-level
  run_map_reduce() helper.

Along the way, fixing this phase surfaced a real, previously-latent bug:
drain_pending_tasks()'s aggregate response list can contain a master-
generated WORKER_UNREACHABLE error (when a worker dies mid-dispatch)
alongside a normal TASK_RESULT for the SAME task_id (from the successful
retry) -- and the job-orchestration layer had only ever been exercised
against TASK_RESULT-shaped payloads. See master/async_server.py (and
master/server.py) and jobs/map.py / jobs/reduce.py for the fix: error
responses now carry task_id, and the orchestration layer uses .get()
instead of direct indexing so an unfamiliar response shape degrades to
"missing/failed" instead of raising KeyError.
"""

import asyncio
import time

import pytest

from common.models import TaskStatus, WorkerStatus
from jobs.map import build_intermediate_results, build_map_job
from jobs.map_reduce import run_map_reduce
from jobs.reduce import build_reduce_job, collect_reduce_results
from jobs.shuffle import shuffle
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
    """Registers, reads `task_count` TASK messages without replying to
    any of them, signals `ready`, then disconnects."""
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


async def run_worker_crashing_on_task_type(
    host: str, port: int, worker_id: str, crash_on_type: str, ready: asyncio.Event
) -> None:
    """Behaves like a normal worker for any task whose type != crash_on_type
    (executes and replies normally). The first task matching crash_on_type
    is received, signals `ready`, and the connection is closed without a
    reply -- simulating a worker that's fine until it happens to draw a
    task of that type."""
    reader, writer = await asyncio.open_connection(host, port)
    conn = AsyncConnection(reader, writer)
    await send_request(conn, protocol.PING)
    await send_request(conn, protocol.REGISTER, {"worker_id": worker_id, "host": "127.0.0.1", "port": 6000})

    try:
        while True:
            try:
                message = await receive_message(conn)
            except ConnectionError:
                return
            if message["type"] != protocol.TASK:
                continue

            payload = message["payload"]
            if payload["task_type"] == crash_on_type:
                ready.set()
                await conn.close()
                return

            result = execute_task(payload["task_type"], payload["task_payload"])
            response = build_message(
                protocol.TASK_RESULT,
                message["request_id"],
                {"task_id": payload["task_id"], "attempt": payload.get("attempt", 1), **result},
            )
            await send_message(conn, response)
    finally:
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


async def force_stale_and_requeue(worker_id: str) -> None:
    """The manual equivalent of one failure_monitor tick, without running
    it as a background task (see module docstring for why)."""
    async_server.worker_manager.get_worker(worker_id).last_heartbeat = time.time() - 10
    stale = async_server.worker_manager.get_stale_workers(async_server.HEARTBEAT_TIMEOUT)
    assert any(w.worker_id == worker_id for w in stale)
    async_server.scheduler.requeue_tasks_for_worker(worker_id)


WORDS = ["apple", "banana", "apple", "orange", "banana", "apple"]


def test_map_task_retries_after_failure():
    """A single Map partition's worker crashes mid-task; the retry
    completes on another worker, and the full job still produces the
    correct result."""

    async def scenario():
        server, host, port = await start_master_server()
        ready = asyncio.Event()
        worker1_task = asyncio.create_task(run_multi_task_crashing_worker(host, port, "worker-1", 1, ready))
        await async_server.wait_for_workers(1)
        worker2_task = asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-2"))
        await async_server.wait_for_workers(2)

        try:
            result = await run_map_reduce(
                async_server.scheduler,
                async_server.drain_pending_tasks,
                "job-1",
                WORDS,
                "WORD_COUNT",
                "SUM",
                num_partitions=1,
            )
            task = async_server.scheduler.get_task("job-1-map-0")
            return result, task
        finally:
            await stop_worker(worker1_task)
            await stop_worker(worker2_task)
            server.close()
            await server.wait_closed()

    result, task = asyncio.run(scenario())
    assert result == {"apple": 3, "banana": 2, "orange": 1}
    assert task.status == TaskStatus.COMPLETED
    assert task.assigned_worker_id == "worker-2"
    assert task.attempt == 2


def test_reduce_task_retries_after_failure():
    """Map completes normally; worker-1 then crashes specifically on a
    Reduce task, which worker-2 picks up and completes."""

    async def scenario():
        server, host, port = await start_master_server()
        ready = asyncio.Event()
        worker1_task = asyncio.create_task(
            run_worker_crashing_on_task_type(host, port, "worker-1", crash_on_type="REDUCE", ready=ready)
        )
        await async_server.wait_for_workers(1)
        worker2_task = asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-2"))
        await async_server.wait_for_workers(2)

        try:
            # Single word -> single partition -> single key, so whichever
            # worker gets the Map task, the (only) Reduce task can land on
            # either worker; run_map_reduce's own retry handles it either way.
            result = await run_map_reduce(
                async_server.scheduler,
                async_server.drain_pending_tasks,
                "job-1",
                ["apple", "apple", "apple"],
                "WORD_COUNT",
                "SUM",
                num_partitions=1,
            )
            reduce_task = async_server.scheduler.get_task("job-1-reduce-apple")
            return result, reduce_task
        finally:
            await stop_worker(worker1_task)
            await stop_worker(worker2_task)
            server.close()
            await server.wait_closed()

    result, reduce_task = asyncio.run(scenario())
    assert result == {"apple": 3}
    assert reduce_task.status == TaskStatus.COMPLETED
    assert reduce_task.assigned_worker_id == "worker-2"
    assert reduce_task.attempt == 2


def test_map_worker_failure_requeues_tasks():
    """worker-1 is deliberately given 2 Map partitions (via the same
    IDLE-toggle technique used elsewhere to stack multiple tasks on one
    worker) and crashes holding both. Both get requeued, both complete via
    worker-2, and the final job result is correct."""

    async def scenario():
        server, host, port = await start_master_server()
        ready = asyncio.Event()
        worker1_task = asyncio.create_task(run_multi_task_crashing_worker(host, port, "worker-1", 2, ready))
        await async_server.wait_for_workers(1)
        worker2_task = asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-2"))
        await async_server.wait_for_workers(2)

        try:
            job_id = "job-1"
            map_tasks = build_map_job(async_server.scheduler, job_id, "WORD_COUNT", WORDS, num_partitions=2)

            async_server.scheduler.assign_task(map_tasks[0].task_id)
            async_server.worker_manager.update_status("worker-1", WorkerStatus.IDLE)
            async_server.scheduler.assign_task(map_tasks[1].task_id)
            assert map_tasks[0].assigned_worker_id == "worker-1"
            assert map_tasks[1].assigned_worker_id == "worker-1"

            dispatch0 = asyncio.create_task(async_server.dispatch_assigned_task(map_tasks[0]))
            dispatch1 = asyncio.create_task(async_server.dispatch_assigned_task(map_tasks[1]))

            await asyncio.wait_for(ready.wait(), timeout=5)

            deadline = time.monotonic() + 5
            while (
                async_server.worker_manager.get_worker("worker-1").status != WorkerStatus.FAILED
                and time.monotonic() < deadline
            ):
                await asyncio.sleep(0.02)
            assert async_server.worker_manager.get_worker("worker-1").status == WorkerStatus.FAILED

            await dispatch0
            await dispatch1

            # Both partitions requeued -- neither stuck ASSIGNED/RUNNING.
            assert map_tasks[0].status == TaskStatus.PENDING
            assert map_tasks[1].status == TaskStatus.PENDING

            map_responses = await async_server.drain_pending_tasks()
            intermediate = build_intermediate_results(job_id, map_tasks, map_responses)
            grouped = shuffle(intermediate)

            reduce_tasks = build_reduce_job(async_server.scheduler, job_id, grouped, "SUM")
            reduce_responses = await async_server.drain_pending_tasks()
            final = collect_reduce_results(reduce_tasks, reduce_responses)

            return final, map_tasks
        finally:
            await stop_worker(worker1_task)
            await stop_worker(worker2_task)
            server.close()
            await server.wait_closed()

    final, map_tasks = asyncio.run(scenario())
    assert final == {"apple": 3, "banana": 2, "orange": 1}
    assert all(t.status == TaskStatus.COMPLETED for t in map_tasks)
    assert all(t.assigned_worker_id == "worker-2" for t in map_tasks)
    assert all(t.attempt == 2 for t in map_tasks)


def test_reduce_worker_failure_requeues_tasks():
    """worker-1 is given 2 Reduce keys and crashes holding both. Both get
    requeued and completed via worker-2."""

    async def scenario():
        server, host, port = await start_master_server()
        worker_task1 = asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-1"))
        await async_server.wait_for_workers(1)
        worker_task2 = asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-2"))
        await async_server.wait_for_workers(2)

        try:
            job_id = "job-1"
            map_tasks = build_map_job(async_server.scheduler, job_id, "WORD_COUNT", WORDS, num_partitions=1)
            map_responses = await async_server.drain_pending_tasks()
            intermediate = build_intermediate_results(job_id, map_tasks, map_responses)
            grouped = shuffle(intermediate)
            assert set(grouped) == {"apple", "banana", "orange"}

            reduce_tasks = build_reduce_job(async_server.scheduler, job_id, grouped, "SUM")

            # Cancel the now-idle real workers and replace worker-1 with a
            # crashing one, so the Reduce phase specifically can be forced
            # onto a worker that will crash holding multiple keys.
            await stop_worker(worker_task1)
            await stop_worker(worker_task2)
            rpc_handler.worker_manager.clear()
            async_server.connections.clear()

            ready = asyncio.Event()
            crashing_task = asyncio.create_task(
                run_multi_task_crashing_worker(host, port, "worker-1", 3, ready)
            )
            await async_server.wait_for_workers(1)
            survivor_task = asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-2"))
            await async_server.wait_for_workers(2)

            # Re-register the same 3 reduce tasks' worker assignment from
            # scratch, since the worker registry was cleared above.
            for key in ("apple", "banana", "orange"):
                task = reduce_tasks[key]
                task.status = TaskStatus.PENDING
                task.assigned_worker_id = None
                task.attempt = 0

            async_server.scheduler.assign_task(reduce_tasks["apple"].task_id)
            async_server.worker_manager.update_status("worker-1", WorkerStatus.IDLE)
            async_server.scheduler.assign_task(reduce_tasks["banana"].task_id)
            async_server.worker_manager.update_status("worker-1", WorkerStatus.IDLE)
            async_server.scheduler.assign_task(reduce_tasks["orange"].task_id)

            assert all(reduce_tasks[k].assigned_worker_id == "worker-1" for k in ("apple", "banana", "orange"))

            dispatches = [
                asyncio.create_task(async_server.dispatch_assigned_task(reduce_tasks[k]))
                for k in ("apple", "banana", "orange")
            ]

            await asyncio.wait_for(ready.wait(), timeout=5)

            deadline = time.monotonic() + 5
            while (
                async_server.worker_manager.get_worker("worker-1").status != WorkerStatus.FAILED
                and time.monotonic() < deadline
            ):
                await asyncio.sleep(0.02)
            assert async_server.worker_manager.get_worker("worker-1").status == WorkerStatus.FAILED

            for d in dispatches:
                await d

            assert all(reduce_tasks[k].status == TaskStatus.PENDING for k in ("apple", "banana", "orange"))

            reduce_responses = await async_server.drain_pending_tasks()
            final = collect_reduce_results(reduce_tasks, reduce_responses)
            return final, reduce_tasks
        finally:
            await stop_worker(crashing_task)
            await stop_worker(survivor_task)
            server.close()
            await server.wait_closed()

    final, reduce_tasks = asyncio.run(scenario())
    assert final == {"apple": 3, "banana": 2, "orange": 1}
    for key in ("apple", "banana", "orange"):
        assert reduce_tasks[key].status == TaskStatus.COMPLETED
        assert reduce_tasks[key].assigned_worker_id == "worker-2"
        assert reduce_tasks[key].attempt == 2


def test_stale_map_result_is_ignored():
    """worker-1 holds a Map task and never heartbeats (connection stays
    open); it's forced stale and requeued (no live failure_monitor task --
    see module docstring), worker-2 completes the retry, and only THEN
    does worker-1's long-delayed attempt-1 reply arrive. It must not
    corrupt the now-COMPLETED attempt-2 state, and the full job still
    produces the correct final result."""

    async def scenario():
        server, host, port = await start_master_server()
        ready = asyncio.Event()
        release = asyncio.Event()
        worker1_task = asyncio.create_task(run_delayed_reply_worker(host, port, "worker-1", ready, release))
        await async_server.wait_for_workers(1)
        worker2_task = asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-2"))
        await async_server.wait_for_workers(2)

        try:
            job_id = "job-1"
            map_tasks = build_map_job(async_server.scheduler, job_id, "WORD_COUNT", WORDS, num_partitions=1)
            task = map_tasks[0]
            async_server.scheduler.assign_task(task.task_id)
            assert task.assigned_worker_id == "worker-1"
            assert task.attempt == 1

            dispatch1 = asyncio.create_task(async_server.dispatch_assigned_task(task))
            await asyncio.wait_for(ready.wait(), timeout=5)

            await force_stale_and_requeue("worker-1")
            assert task.status == TaskStatus.PENDING

            reassigned = async_server.scheduler.assign_next_pending_task()
            assert reassigned is task
            assert task.assigned_worker_id == "worker-2"
            assert task.attempt == 2

            map_response = await async_server.dispatch_assigned_task(task)
            assert map_response["payload"]["status"] == "success"
            assert task.status == TaskStatus.COMPLETED

            # NOW let worker-1's long-delayed attempt-1 reply arrive.
            release.set()
            stale_response = await dispatch1
            assert stale_response["payload"]["attempt"] == 1
            assert task.status == TaskStatus.COMPLETED
            assert task.assigned_worker_id == "worker-2"

            intermediate = build_intermediate_results(job_id, map_tasks, [map_response])
            grouped = shuffle(intermediate)
            reduce_tasks = build_reduce_job(async_server.scheduler, job_id, grouped, "SUM")
            reduce_responses = await async_server.drain_pending_tasks()
            final = collect_reduce_results(reduce_tasks, reduce_responses)
            return final, task
        finally:
            await stop_worker(worker1_task)
            await stop_worker(worker2_task)
            server.close()
            await server.wait_closed()

    final, task = asyncio.run(scenario())
    assert final == {"apple": 3, "banana": 2, "orange": 1}
    assert task.attempt == 2


def test_stale_reduce_result_is_ignored():
    """Same race as test_stale_map_result_is_ignored, for a Reduce task."""

    async def scenario():
        server, host, port = await start_master_server()
        worker1_task = asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-1"))
        await async_server.wait_for_workers(1)

        try:
            job_id = "job-1"
            map_tasks = build_map_job(async_server.scheduler, job_id, "WORD_COUNT", ["apple", "apple"], num_partitions=1)
            map_responses = await async_server.drain_pending_tasks()
            intermediate = build_intermediate_results(job_id, map_tasks, map_responses)
            grouped = shuffle(intermediate)
            reduce_tasks = build_reduce_job(async_server.scheduler, job_id, grouped, "SUM")

            await stop_worker(worker1_task)
            rpc_handler.worker_manager.clear()
            async_server.connections.clear()

            ready = asyncio.Event()
            release = asyncio.Event()
            stuck_task = asyncio.create_task(
                run_delayed_reply_worker(host, port, "worker-stuck", ready, release)
            )
            await async_server.wait_for_workers(1)
            survivor_task = asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-2"))
            await async_server.wait_for_workers(2)

            task = reduce_tasks["apple"]
            task.status = TaskStatus.PENDING
            task.assigned_worker_id = None
            task.attempt = 0

            async_server.scheduler.assign_task(task.task_id)
            assert task.assigned_worker_id == "worker-stuck"

            dispatch1 = asyncio.create_task(async_server.dispatch_assigned_task(task))
            await asyncio.wait_for(ready.wait(), timeout=5)

            await force_stale_and_requeue("worker-stuck")
            assert task.status == TaskStatus.PENDING

            reassigned = async_server.scheduler.assign_next_pending_task()
            assert reassigned is task
            assert task.assigned_worker_id == "worker-2"
            assert task.attempt == 2

            reduce_response = await async_server.dispatch_assigned_task(task)
            assert reduce_response["payload"]["status"] == "success"
            assert task.status == TaskStatus.COMPLETED

            release.set()
            stale_response = await dispatch1
            assert stale_response["payload"]["attempt"] == 1
            assert task.status == TaskStatus.COMPLETED
            assert task.assigned_worker_id == "worker-2"

            final = collect_reduce_results(reduce_tasks, [reduce_response])
            return final, task
        finally:
            await stop_worker(stuck_task)
            await stop_worker(survivor_task)
            server.close()
            await server.wait_closed()

    final, task = asyncio.run(scenario())
    assert final == {"apple": 2}
    assert task.attempt == 2


def test_retry_exhaustion_fails_job():
    """A Reduce key crashes MAX_TASK_ATTEMPTS times in a row: attempt 1
    fails -> PENDING, attempt 2 fails -> PENDING, attempt 3 fails -> FAILED,
    no attempt 4, and the OVERALL job raises -- Reduce failure must be
    explicit, per the asymmetry established in Phase 8.5/8.6."""
    assert MAX_TASK_ATTEMPTS == 3, "test assumes the current default of 3"

    async def scenario():
        server, host, port = await start_master_server()
        worker_task = asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-setup"))
        await async_server.wait_for_workers(1)

        job_id = "job-1"
        map_tasks = build_map_job(async_server.scheduler, job_id, "WORD_COUNT", ["apple", "apple"], num_partitions=1)
        map_responses = await async_server.drain_pending_tasks()
        intermediate = build_intermediate_results(job_id, map_tasks, map_responses)
        grouped = shuffle(intermediate)
        reduce_tasks = build_reduce_job(async_server.scheduler, job_id, grouped, "SUM")
        task = reduce_tasks["apple"]

        await stop_worker(worker_task)
        rpc_handler.worker_manager.clear()
        async_server.connections.clear()
        task.status = TaskStatus.PENDING
        task.assigned_worker_id = None
        task.attempt = 0

        crashing_tasks = []
        ready_events = []
        for i in range(1, MAX_TASK_ATTEMPTS + 1):
            ready = asyncio.Event()
            ready_events.append(ready)
            crashing_tasks.append(
                asyncio.create_task(run_multi_task_crashing_worker(host, port, f"worker-{i}", 1, ready))
            )
            await async_server.wait_for_workers(i)

        try:
            for i in range(1, MAX_TASK_ATTEMPTS + 1):
                worker_id = f"worker-{i}"
                assigned = async_server.scheduler.assign_task(task.task_id)
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
            assert async_server.scheduler.assign_next_pending_task() is None

            with pytest.raises(ValueError):
                collect_reduce_results(reduce_tasks, [])

            return task
        finally:
            for t in crashing_tasks:
                await stop_worker(t)
            server.close()
            await server.wait_closed()

    task = asyncio.run(scenario())
    assert task.status == TaskStatus.FAILED
    assert task.attempt == MAX_TASK_ATTEMPTS


def test_map_retry_exhaustion_partition_becomes_error_but_job_completes():
    """Contrast with the Reduce case above: a Map partition that exhausts
    retries is tolerated -- it becomes an ERROR IntermediateResult, Shuffle
    skips it, and the job still completes from whatever else succeeded."""
    assert MAX_TASK_ATTEMPTS == 3

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

        survivor_task = asyncio.create_task(async_worker.run_worker(host, port, worker_id="survivor"))
        await async_server.wait_for_workers(MAX_TASK_ATTEMPTS + 1)

        try:
            job_id = "job-1"
            # 2 partitions: partition 0 ("apple apple") will be doomed to
            # exhaust on worker-1/2/3; partition 1 ("banana") goes straight
            # to the survivor.
            map_tasks = build_map_job(
                async_server.scheduler, job_id, "WORD_COUNT", ["apple", "apple", "banana"], num_partitions=2
            )
            doomed_task = map_tasks[0]

            for i in range(1, MAX_TASK_ATTEMPTS + 1):
                worker_id = f"worker-{i}"
                assigned = async_server.scheduler.assign_task(doomed_task.task_id)
                assert assigned.assigned_worker_id == worker_id

                dispatch = asyncio.create_task(async_server.dispatch_assigned_task(doomed_task))
                await asyncio.wait_for(ready_events[i - 1].wait(), timeout=5)

                deadline = time.monotonic() + 5
                while (
                    async_server.worker_manager.get_worker(worker_id).status != WorkerStatus.FAILED
                    and time.monotonic() < deadline
                ):
                    await asyncio.sleep(0.02)
                await dispatch

            assert doomed_task.status == TaskStatus.FAILED

            map_responses = await async_server.drain_pending_tasks()  # picks up partition 1 via survivor
            intermediate = build_intermediate_results(job_id, map_tasks, map_responses)
            grouped = shuffle(intermediate)

            reduce_tasks = build_reduce_job(async_server.scheduler, job_id, grouped, "SUM")
            reduce_responses = await async_server.drain_pending_tasks()
            final = collect_reduce_results(reduce_tasks, reduce_responses)
            return final, intermediate
        finally:
            for t in crashing_tasks:
                await stop_worker(t)
            await stop_worker(survivor_task)
            server.close()
            await server.wait_closed()

    final, intermediate = asyncio.run(scenario())
    # "apple" never got counted (its only partition was exhausted); banana did.
    assert final == {"banana": 1}
    assert intermediate[0].status == "error"
    assert intermediate[1].status == "success"


def test_full_map_reduce_recovers_from_worker_failure():
    """The capstone: a full Map -> Shuffle -> Reduce job, real async
    workers, with a genuine worker crash during the Map phase. The pipeline
    recovers and the final result is fully correct."""

    async def scenario():
        server, host, port = await start_master_server()
        ready = asyncio.Event()
        worker1_task = asyncio.create_task(run_multi_task_crashing_worker(host, port, "worker-1", 1, ready))
        await async_server.wait_for_workers(1)
        worker2_task = asyncio.create_task(async_worker.run_worker(host, port, worker_id="worker-2"))
        await async_server.wait_for_workers(2)

        try:
            return await run_map_reduce(
                async_server.scheduler,
                async_server.drain_pending_tasks,
                "job-1",
                WORDS,
                "WORD_COUNT",
                "SUM",
                num_partitions=1,
            )
        finally:
            await stop_worker(worker1_task)
            await stop_worker(worker2_task)
            server.close()
            await server.wait_closed()

    assert asyncio.run(scenario()) == {"apple": 3, "banana": 2, "orange": 1}
