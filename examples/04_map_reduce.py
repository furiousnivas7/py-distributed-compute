"""Example: a full Map -> Shuffle -> Reduce job.

Run it directly:

    python examples/04_map_reduce.py

Word count over a small in-memory list, using the built-in WORD_COUNT/SUM
operations -- no custom code needed. See docs/architecture.md's
"MapReduce data flow" section for how this actually works under the
hood.
"""

import asyncio

from jobs import run_map_reduce
from master import async_server, scheduler, start_master, stop_master
from worker import DirectBackend, run_worker

WORDS = ["apple", "banana", "apple", "orange", "banana", "apple"]


async def main():
    server = await start_master("127.0.0.1", 0)
    host, port = server.sockets[0].getsockname()[:2]

    worker_task = asyncio.create_task(
        run_worker(host, port, worker_id="worker-1", backend=DirectBackend())
    )
    await async_server.wait_for_workers(1)

    try:
        result = await run_map_reduce(
            scheduler,
            async_server.drain_tasks_for,
            "word-count-job",
            WORDS,
            map_operation="WORD_COUNT",
            reduce_operation="SUM",
            num_partitions=2,
        )
        print(f"Word counts: {result}")
    finally:
        worker_task.cancel()
        try:
            await worker_task
        except (asyncio.CancelledError, Exception):
            pass
        await stop_master()


if __name__ == "__main__":
    asyncio.run(main())
