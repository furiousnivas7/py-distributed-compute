"""Example: a worker crashing, and the master retrying the task elsewhere.

Run it directly:

    python examples/06_fault_tolerance.py

The registered function crashes its OWN child process (via os._exit) the
first time it's ever called, then succeeds on every later call -- so
whichever worker gets the first attempt crashes, and the master's
existing worker/transport-failure handling (mark that worker FAILED,
requeue the task) hands the retry to the other, healthy worker, which
completes it. Both workers use MultiprocessingBackend specifically so
the crash only ever kills a real, disposable CHILD process -- never this
example script itself. See docs/architecture.md's "Task lifecycle"
section for the worker/transport-failure vs execution-failure
distinction this relies on.
"""

import asyncio
import os
import tempfile

from jobs import collect_call_result, submit_call
from master import async_server, rpc_handler, scheduler, start_master, stop_master, wait_for_tasks
from worker import MultiprocessingBackend, register_function, run_worker

# A file, not an in-memory flag, because the function below runs inside a
# CHILD process each time -- it needs state that survives across process
# boundaries to only crash once.
SENTINEL = os.path.join(tempfile.gettempdir(), "pydc_example_06_sentinel")


def crash_once_then_succeed(x):
    if not os.path.exists(SENTINEL):
        open(SENTINEL, "w").close()
        os._exit(1)  # simulate a hard crash -- kills this child process only
    return x + 100


async def main():
    if os.path.exists(SENTINEL):
        os.remove(SENTINEL)
    register_function("crash_once_then_succeed", crash_once_then_succeed)

    server = await start_master("127.0.0.1", 0)
    host, port = server.sockets[0].getsockname()[:2]

    worker_ids = ["worker-1", "worker-2"]
    worker_tasks = [
        asyncio.create_task(
            run_worker(host, port, worker_id=wid, backend=MultiprocessingBackend(max_workers=1))
        )
        for wid in worker_ids
    ]
    await async_server.wait_for_workers(len(worker_ids))

    try:
        task = submit_call(scheduler, "crash-once-1", "crash_once_then_succeed", x=1)
        [response] = await wait_for_tasks({task.task_id})
        result = collect_call_result(response)

        final = scheduler.get_task(task.task_id)
        print(f"Result: {result} (attempts: {final.attempt}, completed by: {final.assigned_worker_id})")

        crashed_worker = next(wid for wid in worker_ids if wid != final.assigned_worker_id)
        print(f"{crashed_worker} status: {rpc_handler.worker_manager.get_worker(crashed_worker).status}")
    finally:
        for t in worker_tasks:
            t.cancel()
        for t in worker_tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        await stop_master()
        if os.path.exists(SENTINEL):
            os.remove(SENTINEL)


if __name__ == "__main__":
    asyncio.run(main())
