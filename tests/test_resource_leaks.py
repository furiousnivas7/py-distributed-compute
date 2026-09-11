"""Phase 12.5.6: resource leak audit. Runs repeated
start -> submit -> complete -> shutdown cycles and checks that asyncio
tasks, worker (child) processes, threads, and file descriptors all
return to their pre-run baseline afterward -- not just that the LAST
cycle looks clean, but that repeating the cycle doesn't accumulate
anything across iterations.
"""

import asyncio
import multiprocessing
import os
import threading

from jobs import collect_call_result, submit_call
from master import async_server, rpc_handler, scheduler, start_master, stop_master, wait_for_tasks
from worker import DirectBackend, MultiprocessingBackend, register_function, registry, run_worker


def setup_function(_):
    registry.clear()
    rpc_handler.worker_manager.clear()
    scheduler.clear()
    async_server.connections.clear()
    async_server.clear_dispatch_registry()


def teardown_function(_):
    registry.clear()


def _open_fd_count() -> int | None:
    """None if this platform doesn't expose /dev/fd (skip the FD check
    there rather than failing spuriously)."""
    try:
        return len(os.listdir("/dev/fd"))
    except (FileNotFoundError, NotADirectoryError, PermissionError):
        return None


async def _one_master_worker_cycle(host_port_holder: dict) -> None:
    server = await start_master("127.0.0.1", 0)
    host, port = server.sockets[0].getsockname()[:2]
    host_port_holder["host"], host_port_holder["port"] = host, port

    worker_task = asyncio.create_task(run_worker(host, port, worker_id="w1", backend=DirectBackend()))
    await async_server.wait_for_workers(1)

    task = submit_call(scheduler, "t1", "leak_check_shared_op", x=1)
    [response] = await wait_for_tasks({task.task_id})
    assert collect_call_result(response) == 2

    worker_task.cancel()
    try:
        await worker_task
    except (asyncio.CancelledError, Exception):
        pass
    server.close()
    await server.wait_closed()
    await stop_master()

    # Let anything scheduled via a done-callback (connection cleanup,
    # dispatcher-task self-removal) actually run before the caller checks
    # for leftovers.
    await asyncio.sleep(0.02)


def test_repeated_direct_backend_cycles_leave_no_leaked_asyncio_tasks():
    register_function("leak_check_shared_op", lambda x: x + 1)

    async def run_n_cycles(n: int) -> tuple[set, set]:
        baseline = {t for t in asyncio.all_tasks() if not t.done()}
        for _ in range(n):
            await _one_master_worker_cycle({})
            scheduler.clear()
            async_server.connections.clear()
            async_server.clear_dispatch_registry()
            rpc_handler.worker_manager.clear()
        remaining = {t for t in asyncio.all_tasks() if not t.done()}
        return baseline, remaining

    baseline, remaining = asyncio.run(run_n_cycles(5))
    leaked = remaining - baseline
    assert not leaked, f"leaked asyncio tasks after 5 cycles: {[t.get_name() for t in leaked]}"


def test_repeated_multiprocessing_backend_cycles_leave_no_child_processes():
    register_function("leak_check_mp_op", lambda x: x + 1)

    async def run_n_cycles(n: int):
        for _ in range(n):
            backend = MultiprocessingBackend(max_workers=2)
            await backend.start()
            result = await backend.execute("leak_check_mp_op", {"x": 1})
            assert result == {"status": "success", "result": 2}
            await backend.stop()

    baseline = len(multiprocessing.active_children())
    asyncio.run(run_n_cycles(5))
    remaining = len(multiprocessing.active_children())
    assert remaining == baseline, f"expected {baseline} child processes, found {remaining}"


def test_repeated_multiprocessing_backend_cycles_do_not_accumulate_threads():
    """backend.stop() runs pool.shutdown() via loop.run_in_executor(None, ...)
    -- a transient helper thread, not a persistent one. Across repeated
    cycles, the active thread count must return to baseline each time,
    not climb."""
    register_function("leak_check_thread_op", lambda x: x + 1)

    async def run_n_cycles(n: int) -> list[int]:
        counts = []
        for _ in range(n):
            backend = MultiprocessingBackend(max_workers=2)
            await backend.start()
            await backend.execute("leak_check_thread_op", {"x": 1})
            await backend.stop()
            await asyncio.sleep(0.02)
            counts.append(threading.active_count())
        return counts

    counts = asyncio.run(run_n_cycles(5))
    # Every cycle's post-stop thread count should match the FIRST cycle's
    # -- if it climbs cycle over cycle, something's accumulating threads.
    assert counts == [counts[0]] * len(counts), f"thread count grew across cycles: {counts}"


def test_repeated_full_cycles_do_not_leak_file_descriptors():
    register_function("leak_check_shared_op", lambda x: x + 1)

    async def run_n_cycles(n: int) -> None:
        for _ in range(n):
            await _one_master_worker_cycle({})
            scheduler.clear()
            async_server.connections.clear()
            async_server.clear_dispatch_registry()
            rpc_handler.worker_manager.clear()

    baseline = _open_fd_count()
    if baseline is None:
        import pytest

        pytest.skip("/dev/fd not available on this platform")

    asyncio.run(run_n_cycles(10))
    final = _open_fd_count()
    # A small amount of slack (pytest/coverage/OS-level bookkeeping can
    # legitimately open/close a handful of unrelated descriptors between
    # snapshots) -- the important property is "doesn't grow with N", not
    # "exactly zero delta".
    assert final <= baseline + 5, f"file descriptor count grew from {baseline} to {final} over 10 cycles"
