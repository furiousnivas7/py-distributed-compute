"""Example: multiple workers sharing a job's tasks.

Run it directly:

    python examples/03_multiple_workers.py

Starts three workers against one master and submits more tasks than any
single worker could run at once -- the scheduler assigns each task to
whichever worker is IDLE, so the work is naturally spread across all
three. Prints which worker actually ran each task.
"""

import asyncio

from master import async_server, scheduler, start_master, stop_master, wait_for_tasks
from worker import DirectBackend, run_worker


async def main():
    server = await start_master("127.0.0.1", 0)
    host, port = server.sockets[0].getsockname()[:2]

    worker_ids = ["worker-1", "worker-2", "worker-3"]
    worker_tasks = [
        asyncio.create_task(run_worker(host, port, worker_id=wid, backend=DirectBackend()))
        for wid in worker_ids
    ]
    await async_server.wait_for_workers(len(worker_ids))

    try:
        tasks = [scheduler.submit_task(f"add-{i}", "ADD", {"a": i, "b": i}) for i in range(6)]
        responses = await wait_for_tasks({t.task_id for t in tasks})

        for response in sorted(responses, key=lambda r: r["payload"]["task_id"]):
            task_id = response["payload"]["task_id"]
            result = response["payload"]["result"]
            worker_id = scheduler.get_task(task_id).assigned_worker_id
            print(f"{task_id}: result={result}, ran on {worker_id}")
    finally:
        for t in worker_tasks:
            t.cancel()
        for t in worker_tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        await stop_master()


if __name__ == "__main__":
    asyncio.run(main())
