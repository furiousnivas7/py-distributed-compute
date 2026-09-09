"""Public API for job submission and orchestration: single function calls,
Map/Reduce, and full MapReduce.

Stable import surface -- deep imports (``jobs.call``, ``jobs.map``, ...)
keep working exactly as before; this module only adds re-exports on top of
what already exists.

Typical usage -- a single call::

    from jobs import submit_call, collect_call_result
    from master import scheduler, wait_for_tasks

    task = submit_call(scheduler, "t1", "double", x=21)
    [response] = await wait_for_tasks({task.task_id})
    result = collect_call_result(response)  # 42

An arbitrary function, never registered anywhere (see the README's
"Security Considerations" section before using this on an untrusted
cluster)::

    from jobs import submit_serialized_call

    task = submit_serialized_call(scheduler, "t1", lambda x: x * 2, args=[21])

A full MapReduce job::

    from jobs import ExecutionSpec, run_map_reduce
    from master import scheduler
    from master.async_server import drain_tasks_for

    result = await run_map_reduce(
        scheduler, drain_tasks_for, "job-1", data,
        map_operation="WORD_COUNT", reduce_operation="SUM",
        num_partitions=4,
    )
    # or with an arbitrary callable instead of a built-in/registered name:
    # map_operation=ExecutionSpec.serialized(my_map_fn)
"""

from jobs.call import CallError, collect_call_result, submit_call, submit_registered_call, submit_serialized_call
from jobs.map import build_intermediate_results, build_map_job, collect_map_results
from jobs.map_reduce import run_map_reduce
from jobs.models import ExecutionSpec, IntermediateResult, IntermediateResultStore, ResultStatus
from jobs.reduce import build_reduce_job, collect_reduce_results, reduce_grouped
from jobs.shuffle import shuffle

__all__ = [
    "submit_call",
    "submit_registered_call",
    "submit_serialized_call",
    "collect_call_result",
    "CallError",
    "build_map_job",
    "collect_map_results",
    "build_intermediate_results",
    "build_reduce_job",
    "collect_reduce_results",
    "reduce_grouped",
    "shuffle",
    "run_map_reduce",
    "ExecutionSpec",
    "IntermediateResult",
    "IntermediateResultStore",
    "ResultStatus",
]
