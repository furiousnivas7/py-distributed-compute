"""Example: a serialized callable, executed by a real worker.

Run it directly:

    python examples/02_serialized_callable.py

Unlike a registered function (see 01_registered_function.py), a
serialized callable never has to be pre-registered or even importable by
the worker -- it's shipped as part of the task payload itself, using
cloudpickle. That makes it remote code execution BY DESIGN: only use it
on a cluster you fully trust. See README.md's "Security Considerations"
section, and worker/serialization.py's module docstring, before using
this pattern for anything beyond a trusted, cooperative cluster.
"""

import asyncio

from jobs import collect_call_result, submit_serialized_call
from master import async_server, scheduler, start_master, stop_master, wait_for_tasks
from worker import DirectBackend, run_worker


async def main():
    server = await start_master("127.0.0.1", 0)
    host, port = server.sockets[0].getsockname()[:2]

    worker_task = asyncio.create_task(
        run_worker(host, port, worker_id="worker-1", backend=DirectBackend())
    )
    await async_server.wait_for_workers(1)

    try:
        # A closure -- never registered anywhere, not even importable by
        # the worker process. cloudpickle ships it by value.
        multiplier = 3

        def scale(x):
            return x * multiplier

        task = submit_serialized_call(scheduler, "scale-14", scale, args=[14])
        [response] = await wait_for_tasks({task.task_id})
        result = collect_call_result(response)
        print(f"scale(14) = {result}")
    finally:
        worker_task.cancel()
        try:
            await worker_task
        except (asyncio.CancelledError, Exception):
            pass
        await stop_master()


if __name__ == "__main__":
    asyncio.run(main())
