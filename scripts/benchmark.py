"""Phase 12.5.5: performance baseline.

Not a pytest test -- these numbers are machine-dependent and not
pass/fail assertions; the goal is a reproducible baseline future changes
can be compared against, not a maximum-performance claim. Run it
directly:

    python scripts/benchmark.py

Uses only the public API (see docs/api.md). Recorded results from one
run live in docs/performance_baseline.md.
"""

import asyncio
import statistics
import time

from jobs import collect_call_result, submit_call
from master import async_server, rpc_handler, scheduler, start_master, stop_master, wait_for_tasks
from rpc.async_rpc import send_request
from rpc import protocol
from worker import DirectBackend, MultiprocessingBackend, register_function, registry, run_worker


def _reset():
    registry.clear()
    rpc_handler.worker_manager.clear()
    scheduler.clear()
    async_server.connections.clear()
    async_server.clear_dispatch_registry()


async def _start():
    server = await start_master("127.0.0.1", 0)
    host, port = server.sockets[0].getsockname()[:2]
    return server, host, port


async def _stop_worker(t):
    t.cancel()
    try:
        await t
    except (asyncio.CancelledError, Exception):
        pass


async def bench_worker_registration_time(n=20):
    _reset()
    server, host, port = await _start()
    try:
        times = []
        for i in range(n):
            start = time.perf_counter()
            worker_task = asyncio.create_task(
                run_worker(host, port, worker_id=f"reg-{i}", backend=DirectBackend())
            )
            # Always 1: each iteration's worker is removed before the next
            # one registers (see below), so the live count never
            # accumulates.
            await async_server.wait_for_workers(1)
            times.append(time.perf_counter() - start)
            await _stop_worker(worker_task)
            async_server.connections.pop(f"reg-{i}", None)
            rpc_handler.worker_manager.remove_worker(f"reg-{i}")
        return times
    finally:
        server.close()
        await server.wait_closed()
        await stop_master()


async def bench_ping_rpc_overhead(n=200):
    _reset()
    server, host, port = await _start()
    try:
        reader, writer = await asyncio.open_connection(host, port)
        from rpc.async_connection import AsyncConnection

        conn = AsyncConnection(reader, writer)
        times = []
        for _ in range(n):
            start = time.perf_counter()
            await send_request(conn, protocol.PING)
            times.append(time.perf_counter() - start)
        await conn.close()
        return times
    finally:
        server.close()
        await server.wait_closed()
        await stop_master()


async def bench_submission_and_scheduling_latency(n=100):
    """Time from submit_task() to the task reaching RUNNING (assigned +
    dispatched) -- isolates scheduling/dispatch overhead from actual
    execution time."""
    _reset()
    register_function("noop", lambda: "ok")
    server, host, port = await _start()
    worker_task = asyncio.create_task(run_worker(host, port, worker_id="w1", backend=DirectBackend()))
    try:
        await async_server.wait_for_workers(1)
        times = []
        for i in range(n):
            start = time.perf_counter()
            task = submit_call(scheduler, f"lat-{i}", "noop")
            await wait_for_tasks({task.task_id})
            times.append(time.perf_counter() - start)
        return times
    finally:
        await _stop_worker(worker_task)
        server.close()
        await server.wait_closed()
        await stop_master()


async def bench_throughput(num_workers: int, backend_factory, num_tasks=200):
    _reset()
    register_function("bench_op", lambda x: x + 1)
    server, host, port = await _start()
    worker_tasks = [
        asyncio.create_task(run_worker(host, port, worker_id=f"w{i}", backend=backend_factory()))
        for i in range(num_workers)
    ]
    try:
        await async_server.wait_for_workers(num_workers)
        start = time.perf_counter()
        tasks = [submit_call(scheduler, f"t{i}", "bench_op", x=i) for i in range(num_tasks)]
        responses = await wait_for_tasks({t.task_id for t in tasks}, timeout=60.0)
        elapsed = time.perf_counter() - start
        assert len(responses) == num_tasks
        return num_tasks / elapsed
    finally:
        for t in worker_tasks:
            await _stop_worker(t)
        server.close()
        await server.wait_closed()
        await stop_master()


def _summarize(label, times):
    times_ms = [t * 1000 for t in times]
    print(
        f"{label}: mean={statistics.mean(times_ms):.3f}ms "
        f"median={statistics.median(times_ms):.3f}ms "
        f"p95={sorted(times_ms)[int(len(times_ms) * 0.95)]:.3f}ms "
        f"min={min(times_ms):.3f}ms max={max(times_ms):.3f}ms (n={len(times_ms)})"
    )


async def main():
    print("=== worker registration time ===")
    _summarize("registration", await bench_worker_registration_time())

    print("\n=== PING RPC round-trip overhead ===")
    _summarize("ping", await bench_ping_rpc_overhead())

    print("\n=== submission -> result latency (DirectBackend, 1 worker) ===")
    _summarize("submit+wait", await bench_submission_and_scheduling_latency())

    print("\n=== throughput: DirectBackend, workers scaling ===")
    print(f"{'workers':>8} {'tasks/sec':>12}")
    for n in (1, 2, 4, 8):
        rate = await bench_throughput(n, DirectBackend)
        print(f"{n:>8} {rate:>12.1f}")

    print("\n=== throughput: DirectBackend vs MultiprocessingBackend (4 workers) ===")
    direct_rate = await bench_throughput(4, DirectBackend)
    mp_rate = await bench_throughput(4, lambda: MultiprocessingBackend(max_workers=1))
    print(f"{'DirectBackend':>20} {direct_rate:>12.1f} tasks/sec")
    print(f"{'MultiprocessingBackend':>20} {mp_rate:>12.1f} tasks/sec")


if __name__ == "__main__":
    asyncio.run(main())
