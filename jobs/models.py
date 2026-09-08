"""Job-level data models (Map, and later Reduce/Shuffle).

These sit above common/models.py's Task and Worker: job_id and
partition_id are Map/Reduce-job concepts the scheduler and executor don't
need to know about, so they live here rather than on the generic Task
model.
"""

from dataclasses import dataclass
from typing import Callable, Union


class ResultStatus:
    SUCCESS = "success"
    ERROR = "error"


@dataclass(frozen=True)
class ExecutionSpec:
    """Phase 10.5: describes how to run a MAP or REDUCE operation, so the
    SAME orchestration API (jobs.map.build_map_job, jobs.reduce.
    build_reduce_job, jobs.map_reduce.run_map_reduce) works for a
    registered/built-in operation name and an arbitrary serialized
    callable -- no separate "_serialized" pipeline needed.

    Construct via .registered()/.serialized(), not the constructor
    directly -- ExecutionSpec.coerce() (what build_map_job/build_reduce_job
    call internally) also accepts a plain string as shorthand for
    .registered(that string), which is what makes every existing caller
    passing "WORD_COUNT" or "SUM" keep working completely unchanged.
    """

    execution_mode: str
    operation: str | None = None
    fn: Callable | None = None

    @classmethod
    def registered(cls, operation: str) -> "ExecutionSpec":
        return cls("registered", operation=operation)

    @classmethod
    def serialized(cls, fn: Callable) -> "ExecutionSpec":
        return cls("serialized_callable", fn=fn)

    @classmethod
    def coerce(cls, value: Union[str, "ExecutionSpec"]) -> "ExecutionSpec":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls.registered(value)
        raise TypeError(f"Expected a str or ExecutionSpec, got {type(value).__name__}")


@dataclass
class IntermediateResult:
    """One partition's output from a Map task -- an explicit intermediate
    representation, distinct from the raw TASK_RESULT response dict, so a
    later Shuffle/Reduce stage has something structured to consume instead
    of re-deriving job_id/partition_id/provenance from responses itself."""

    job_id: str
    partition_id: int
    task_id: str
    status: str
    data: list | None = None
    message: str | None = None
    worker_id: str | None = None
    attempt: int = 0


class IntermediateResultStore:
    """In-memory store of intermediate results, keyed by job_id.

    Keying by job_id is what keeps two Map jobs' results from mixing even
    if their partition_ids or task_id suffixes happen to overlap (e.g. two
    jobs both having a partition_id 0).
    """

    def __init__(self):
        self._results: dict[str, list[IntermediateResult]] = {}

    def store(self, job_id: str, results: list[IntermediateResult]) -> None:
        self._results[job_id] = list(results)

    def get(self, job_id: str) -> list[IntermediateResult]:
        return list(self._results.get(job_id, []))

    def has_job(self, job_id: str) -> bool:
        return job_id in self._results

    def clear(self) -> None:
        self._results.clear()
