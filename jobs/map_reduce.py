"""End-to-end MapReduce orchestration: wires the existing Map, Shuffle, and
Reduce layers together. No new scheduling, transport, or execution logic
lives here -- this module only calls into jobs/map.py, jobs/shuffle.py, and
jobs/reduce.py, which in turn call into the existing Scheduler.

`dispatch` is injected rather than imported (e.g. the caller passes
master.async_server.drain_tasks_for) so this stays agnostic to which
master implementation -- sync or async -- is actually running the tasks,
matching how build_map_job/build_reduce_job only depend on a Scheduler
rather than a specific transport.

Phase 8.8: `dispatch` now takes this job's own set of task_ids and must
only assign/report on tasks within that set (see
master.async_server.drain_tasks_for). Passing master.async_server.
drain_pending_tasks here (which ignores its argument and drains
everything) is exactly the bug this phase fixes: with more than one
MapReduce job -- or a job running alongside ordinary ad-hoc tasks -- an
unscoped dispatch can dispatch and "keep" a response that actually
belongs to someone else's task, corrupting both callers' results.
"""

from typing import Awaitable, Callable

from jobs.map import build_intermediate_results, build_map_job
from jobs.reduce import build_reduce_job, collect_reduce_results
from jobs.shuffle import shuffle
from master.scheduler import Scheduler

DispatchFn = Callable[[set[str]], Awaitable[list[dict]]]


async def run_map_reduce(
    scheduler: Scheduler,
    dispatch: DispatchFn,
    job_id: str,
    data: list,
    map_operation: str,
    reduce_operation: str,
    num_partitions: int,
) -> dict:
    """Run a full Map -> Shuffle -> Reduce job and return the final result.

    Map failures are tolerated: a failed/missing partition just contributes
    nothing to Shuffle (see jobs.map.build_intermediate_results and
    jobs.shuffle.shuffle). Reduce failures are NOT tolerated: this
    deliberately does not catch collect_reduce_results' ValueError -- a
    silently dropped key would corrupt the final result with no signal
    anything went wrong, so it propagates to the caller instead.

    `dispatch` is called once per phase, each time with exactly this job's
    own task_ids for that phase -- never the whole scheduler's pending
    queue -- so this job's dispatch can safely run concurrently with any
    other caller's (another job, ordinary ad-hoc tasks, or the failure
    monitor) without either one's responses ending up in the wrong place.
    """
    map_tasks = build_map_job(scheduler, job_id, map_operation, data, num_partitions)
    map_task_ids = {task.task_id for task in map_tasks}
    map_responses = await dispatch(map_task_ids)
    intermediate_results = build_intermediate_results(job_id, map_tasks, map_responses)

    grouped = shuffle(intermediate_results)

    reduce_tasks = build_reduce_job(scheduler, job_id, grouped, reduce_operation)
    reduce_task_ids = {task.task_id for task in reduce_tasks.values()}
    reduce_responses = await dispatch(reduce_task_ids)

    return collect_reduce_results(reduce_tasks, reduce_responses)
