"""Example: a registered function, executed by a real worker.

Run it directly:

    python examples/01_registered_function.py

Starts an in-process master and one DirectBackend worker, registers a
plain Python function, submits a call to it, and prints the result.
Uses only the public API (see docs/api.md).
"""

import asyncio

from jobs import collect_call_result, submit_call
from master import async_server, scheduler, start_master, stop_master, wait_for_tasks
from worker import DirectBackend, register_function, run_worker


def double(x):
    return x * 2


async def main():
    register_function("double", double)

    server = await start_master("127.0.0.1", 0)
    host, port = server.sockets[0].getsockname()[:2]

    worker_task = asyncio.create_task(
        run_worker(host, port, worker_id="worker-1", backend=DirectBackend())
    )
    await async_server.wait_for_workers(1)

    try:
        task = submit_call(scheduler, "double-21", "double", x=21)
        [response] = await wait_for_tasks({task.task_id})
        result = collect_call_result(response)
        print(f"double(21) = {result}")
    finally:
        worker_task.cancel()
        try:
            await worker_task
        except (asyncio.CancelledError, Exception):
            pass
        await stop_master()


if __name__ == "__main__":
    asyncio.run(main())
