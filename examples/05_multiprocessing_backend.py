"""Example: CPU-bound work via MultiprocessingBackend.

Run it directly:

    python examples/05_multiprocessing_backend.py

Unlike DirectBackend (see 01_registered_function.py), MultiprocessingBackend
runs each task in a real child process, so a CPU-bound task doesn't block
the worker's own event loop -- its heartbeat keeps arriving on schedule
even while a task is running. See docs/architecture.md's "execution-
backend abstraction" section.
"""

import asyncio
import time

from jobs import collect_call_result, submit_call
from master import async_server, scheduler, start_master, stop_master, wait_for_tasks
from worker import MultiprocessingBackend, register_function, run_worker


def slow_square(x):
    time.sleep(0.5)  # simulate real CPU-bound work
    return x * x


async def main():
    register_function("slow_square", slow_square)

    server = await start_master("127.0.0.1", 0)
    host, port = server.sockets[0].getsockname()[:2]

    backend = MultiprocessingBackend(max_workers=2, max_in_flight=2)
    worker_task = asyncio.create_task(
        run_worker(host, port, worker_id="worker-1", backend=backend, heartbeat_interval=0.1)
    )
    await async_server.wait_for_workers(1)

    try:
        task = submit_call(scheduler, "slow-square-7", "slow_square", x=7)
        [response] = await wait_for_tasks({task.task_id})
        result = collect_call_result(response)
        print(f"slow_square(7) = {result} (computed in a real child process)")
    finally:
        worker_task.cancel()
        try:
            await worker_task
        except (asyncio.CancelledError, Exception):
            pass
        await stop_master()


if __name__ == "__main__":
    asyncio.run(main())
