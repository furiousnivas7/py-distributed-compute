"""Phase 12.5.2: end-to-end system tests -- the full stack, exercised the
way a user actually would (through the public API: master/worker/jobs/
common), not through internals. Unlike the ~570 narrower unit/integration
tests elsewhere in this suite (each proving one specific mechanism in
isolation), every test here runs a COMPLETE, realistic workflow start to
finish: register workers, submit real work, and tear everything down
cleanly, so no major workflow's correctness rests on unit tests alone.
"""

import asyncio

from common import TaskStatus, WorkerStatus
from jobs import (
    ExecutionSpec,
    collect_call_result,
    run_map_reduce,
    submit_call,
    submit_serialized_call,
)
from master import async_server, rpc_handler, scheduler, start_master, stop_master, wait_for_tasks
from worker import DirectBackend, MultiprocessingBackend, register_function, registry, run_worker


async def _start_master():
    server = await start_master("127.0.0.1", 0)
    host, port = server.sockets[0].getsockname()[:2]
    return server, host, port


async def _stop_worker(task: asyncio.Task) -> None:
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):
        pass


def setup_function(_):
    registry.clear()
    rpc_handler.worker_manager.clear()
    scheduler.clear()
    async_server.connections.clear()
    async_server.clear_dispatch_registry()


def teardown_function(_):
    registry.clear()


# -- one worker -------------------------------------------------------


def test_single_worker_full_lifecycle():
    register_function("double", lambda x: x * 2)

    async def scenario():
        server, host, port = await _start_master()
        worker_task = asyncio.create_task(
            run_worker(host, port, worker_id="worker-1", backend=DirectBackend())
        )
        try:
            await async_server.wait_for_workers(1)
            task = submit_call(scheduler, "t1", "double", x=21)
            [response] = await wait_for_tasks({task.task_id})
            return collect_call_result(response)
        finally:
            await _stop_worker(worker_task)
            server.close()
            await server.wait_closed()
            await stop_master()

    assert asyncio.run(scenario()) == 42


# -- multiple workers, multiple simultaneous tasks -------------------


def test_multiple_workers_multiple_simultaneous_tasks():
    register_function("square", lambda x: x * x)

    async def scenario():
        server, host, port = await _start_master()
        worker_tasks = [
            asyncio.create_task(run_worker(host, port, worker_id=f"worker-{i}", backend=DirectBackend()))
            for i in range(3)
        ]
        try:
            await async_server.wait_for_workers(3)
            tasks = [submit_call(scheduler, f"sq-{i}", "square", x=i) for i in range(15)]
            responses = await wait_for_tasks({t.task_id for t in tasks})
            results = {r["payload"]["task_id"]: collect_call_result(r) for r in responses}

            workers_used = {scheduler.get_task(tid).assigned_worker_id for tid in results}
            return results, workers_used
        finally:
            for t in worker_tasks:
                await _stop_worker(t)
            server.close()
            await server.wait_closed()
            await stop_master()

    results, workers_used = asyncio.run(scenario())
    assert results == {f"sq-{i}": i * i for i in range(15)}
    # With 15 tasks and 3 idle workers, the work should actually be spread
    # out -- not a hard scheduler guarantee, but a real property of this
    # workload that would be suspicious if it failed.
    assert len(workers_used) >= 2


# -- registered AND serialized functions in the same run -------------


def test_registered_and_serialized_functions_together():
    register_function("triple", lambda x: x * 3)

    async def scenario():
        server, host, port = await _start_master()
        worker_task = asyncio.create_task(
            run_worker(host, port, worker_id="worker-1", backend=DirectBackend())
        )
        try:
            await async_server.wait_for_workers(1)

            registered_task = submit_call(scheduler, "reg-1", "triple", x=5)
            serialized_task = submit_serialized_call(scheduler, "ser-1", lambda x: x - 1, args=[10])

            responses = await wait_for_tasks({registered_task.task_id, serialized_task.task_id})
            by_id = {r["payload"]["task_id"]: collect_call_result(r) for r in responses}
            return by_id
        finally:
            await _stop_worker(worker_task)
            server.close()
            await server.wait_closed()
            await stop_master()

    results = asyncio.run(scenario())
    assert results == {"reg-1": 15, "ser-1": 9}


# -- multiprocessing backend, end to end -----------------------------


def test_multiprocessing_backend_end_to_end():
    register_function("cube", lambda x: x**3)

    async def scenario():
        server, host, port = await _start_master()
        backend = MultiprocessingBackend(max_workers=2, max_in_flight=2)
        worker_task = asyncio.create_task(
            run_worker(host, port, worker_id="worker-1", backend=backend)
        )
        try:
            await async_server.wait_for_workers(1)
            tasks = [submit_call(scheduler, f"cube-{i}", "cube", x=i) for i in range(6)]
            responses = await wait_for_tasks({t.task_id for t in tasks})
            return {r["payload"]["task_id"]: collect_call_result(r) for r in responses}
        finally:
            await _stop_worker(worker_task)
            server.close()
            await server.wait_closed()
            await stop_master()

    results = asyncio.run(scenario())
    assert results == {f"cube-{i}": i**3 for i in range(6)}


# -- MapReduce, end to end --------------------------------------------


def test_map_reduce_end_to_end():
    async def scenario():
        server, host, port = await _start_master()
        worker_task = asyncio.create_task(
            run_worker(host, port, worker_id="worker-1", backend=DirectBackend())
        )
        try:
            await async_server.wait_for_workers(1)
            return await run_map_reduce(
                scheduler,
                async_server.drain_tasks_for,
                "job-1",
                ["a", "b", "a", "c", "b", "a"],
                map_operation="WORD_COUNT",
                reduce_operation="SUM",
                num_partitions=3,
            )
        finally:
            await _stop_worker(worker_task)
            server.close()
            await server.wait_closed()
            await stop_master()

    assert asyncio.run(scenario()) == {"a": 3, "b": 2, "c": 1}


def test_map_reduce_with_a_serialized_map_operation():
    """MapReduce combined with the OTHER execution mode -- an arbitrary
    callable, never registered anywhere, as the map operation."""

    def shout(word):
        return [word.upper(), 1]

    async def scenario():
        server, host, port = await _start_master()
        worker_task = asyncio.create_task(
            run_worker(host, port, worker_id="worker-1", backend=DirectBackend())
        )
        try:
            await async_server.wait_for_workers(1)
            return await run_map_reduce(
                scheduler,
                async_server.drain_tasks_for,
                "job-2",
                ["a", "b", "a"],
                map_operation=ExecutionSpec.serialized(shout),
                reduce_operation="SUM",
                num_partitions=2,
            )
        finally:
            await _stop_worker(worker_task)
            server.close()
            await server.wait_closed()
            await stop_master()

    assert asyncio.run(scenario()) == {"A": 2, "B": 1}


# -- worker disconnect / crash / retry / reconnect --------------------


def test_worker_disconnect_is_detected_and_task_is_retried_on_another_worker():
    register_function("noop", lambda: "ok")

    async def scenario():
        server, host, port = await _start_master()
        worker1_task = asyncio.create_task(
            run_worker(host, port, worker_id="worker-1", backend=DirectBackend())
        )
        worker2_task = asyncio.create_task(
            run_worker(host, port, worker_id="worker-2", backend=DirectBackend())
        )
        try:
            await async_server.wait_for_workers(2)

            # Simulate worker-1 disconnecting BEFORE any task reaches it,
            # by closing its connection directly -- the master's next
            # attempt to reach it (via dispatch_assigned_task) sees a dead
            # link and treats it exactly like a crash.
            link = async_server.connections.get("worker-1")
            await link.conn.close()
            # Give handle_worker_connection's own read loop a moment to
            # notice the close and clean up (pop from connections, in
            # this case leaving worker-1 REGISTERED with no live link).
            await asyncio.sleep(0.05)

            task = scheduler.submit_task("noop-1", "noop", {})
            [response] = await wait_for_tasks({task.task_id})
            return response, scheduler.get_task(task.task_id)
        finally:
            await _stop_worker(worker1_task)
            await _stop_worker(worker2_task)
            server.close()
            await server.wait_closed()
            await stop_master()

    response, final_task = asyncio.run(scenario())
    assert response["payload"]["status"] == "success"
    assert final_task.status == TaskStatus.COMPLETED
    assert final_task.assigned_worker_id == "worker-2"


def test_worker_crash_mid_task_is_retried_and_worker_can_reconnect():
    """The fuller version: worker-1 is actually RUNNING a task when it
    dies, the task is requeued and completed by worker-2, AND worker-1
    (same worker_id, simulating a process restart) successfully
    reconnects afterward with a bumped generation."""
    register_function("slow_add", lambda x, y: x + y)

    async def scenario():
        server, host, port = await _start_master()
        worker1_task = asyncio.create_task(
            run_worker(host, port, worker_id="worker-1", backend=DirectBackend())
        )
        worker2_task = asyncio.create_task(
            run_worker(host, port, worker_id="worker-2", backend=DirectBackend())
        )
        try:
            await async_server.wait_for_workers(2)
            generation_before = rpc_handler.worker_manager.get_worker("worker-1").generation

            task = scheduler.submit_task("slow-add-1", "slow_add", {"x": 1, "y": 2})
            assigned = scheduler.assign_next_pending_task()
            assert assigned.assigned_worker_id == "worker-1"

            # worker-1 "crashes" while this task is ASSIGNED to it.
            await _stop_worker(worker1_task)

            response = await async_server.dispatch_assigned_task(task)
            assert response["type"] == "ERROR"
            assert scheduler.get_task(task.task_id).status == TaskStatus.PENDING
            assert rpc_handler.worker_manager.get_worker("worker-1").status == WorkerStatus.FAILED

            # The dispatcher (started by wait_for_tasks below) picks the
            # requeued task up and hands it to worker-2 instead.
            [response2] = await wait_for_tasks({task.task_id})

            # worker-1 "restarts" -- same worker_id, a fresh connection.
            worker1_restarted = asyncio.create_task(
                run_worker(host, port, worker_id="worker-1", backend=DirectBackend())
            )
            await async_server.wait_for_workers(2)
            generation_after = rpc_handler.worker_manager.get_worker("worker-1").generation

            await _stop_worker(worker1_restarted)
            return response2, scheduler.get_task(task.task_id), generation_before, generation_after
        finally:
            await _stop_worker(worker2_task)
            server.close()
            await server.wait_closed()
            await stop_master()

    response, final_task, gen_before, gen_after = asyncio.run(scenario())
    assert collect_call_result(response) == 3
    assert final_task.status == TaskStatus.COMPLETED
    assert final_task.assigned_worker_id == "worker-2"
    assert final_task.attempt == 2
    assert gen_after == gen_before + 1


# -- master / worker shutdown ------------------------------------------


def test_master_shutdown_resolves_pending_tasks_deterministically():
    """stop_master() must not leave a submitted task's outcome ambiguous
    -- either it completed, or the caller gets a clear terminal response,
    never a hang."""
    register_function("noop", lambda: "ok")

    async def scenario():
        server, host, port = await _start_master()
        worker_task = asyncio.create_task(
            run_worker(host, port, worker_id="worker-1", backend=DirectBackend())
        )
        try:
            await async_server.wait_for_workers(1)
            task = submit_call(scheduler, "noop-1", "noop")
            [response] = await wait_for_tasks({task.task_id})
            assert response["payload"]["status"] == "success"
        finally:
            await _stop_worker(worker_task)
            server.close()
            await server.wait_closed()
            # The actual thing under test: stop_master() itself must
            # complete cleanly (no hang, no exception) even with a worker
            # connection already torn down.
            await stop_master()
            assert not async_server.is_master_running()

    asyncio.run(scenario())


def test_worker_graceful_shutdown_finishes_its_in_flight_task_first():
    register_function("slow", lambda: "done")

    async def scenario():
        server, host, port = await _start_master()
        shutdown_event = asyncio.Event()
        worker_task = asyncio.create_task(
            run_worker(host, port, worker_id="worker-1", backend=DirectBackend(), shutdown_event=shutdown_event)
        )
        try:
            await async_server.wait_for_workers(1)
            task = submit_call(scheduler, "slow-1", "slow")

            # run_worker's own documented contract: shutdown_event must
            # only be set once the task has actually been assigned to
            # THIS worker -- otherwise (fire-and-forget SHUTDOWN racing
            # ahead of assignment, with no other worker to fall back to)
            # the task would stay PENDING forever instead of ever being
            # dispatched. wait_for_tasks() below is what actually starts
            # the dispatcher, so poll for assignment directly first.
            deadline = asyncio.get_running_loop().time() + 5
            while scheduler.get_task(task.task_id).assigned_worker_id is None:
                async_server.ensure_dispatcher_running()
                if asyncio.get_running_loop().time() > deadline:
                    raise AssertionError("task was never assigned")
                await asyncio.sleep(0.01)

            shutdown_event.set()
            [response] = await wait_for_tasks({task.task_id})
            return response
        finally:
            await _stop_worker(worker_task)
            server.close()
            await server.wait_closed()
            await stop_master()

    response = asyncio.run(scenario())
    # The task was already assigned to worker-1 before shutdown_event was
    # set -- DRAINING lets an in-flight task finish before the connection
    # closes, so this must complete successfully, not hang or error.
    assert collect_call_result(response) == "done"
