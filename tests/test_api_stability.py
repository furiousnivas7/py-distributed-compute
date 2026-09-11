"""Phase 12.5.1: API stability audit.

tests/test_public_api.py (Phase 12.1) already continuously enforces "no
accidental public exports" (dir(pkg) == __all__ exactly) and "deep import
== curated re-export, same object" on every test run -- this file adds
what that one doesn't cover: a frozen snapshot of exactly which names
each package exports (so an accidental rename/removal fails loudly and
specifically, not just "the set changed"), and exception/async-sync
consistency checks across the whole public surface.
"""

import asyncio
import inspect

import common
import jobs
import master
import worker

# Frozen as of Phase 12.5.1 -- deliberately hardcoded rather than derived
# from the packages themselves, so a real accidental change (rename,
# removal, unreviewed addition) fails this test with a specific diff
# instead of silently becoming the new baseline.
EXPECTED_MASTER_API = {
    "scheduler", "start_master", "stop_master", "wait_for_tasks", "wait_for_workers",
    "Scheduler", "TaskNotFoundError", "NoAvailableWorkerError",
    "WorkerManager", "DuplicateWorkerError", "WorkerNotFoundError",
}
EXPECTED_WORKER_API = {
    "run_worker", "ExecutionBackend", "DirectBackend", "MultiprocessingBackend",
    "BackendMetrics", "BackendConfig", "resolve_backend_config", "build_backend",
    "ExecutionError", "ExecutionErrorCode", "MAP_OPERATIONS", "REDUCE_OPERATIONS",
    "register_function", "register", "get_function", "is_registered",
    "clear_registry", "FunctionAlreadyRegisteredError",
}
EXPECTED_JOBS_API = {
    "submit_call", "submit_registered_call", "submit_serialized_call", "collect_call_result",
    "CallError", "build_map_job", "collect_map_results", "build_intermediate_results",
    "build_reduce_job", "collect_reduce_results", "reduce_grouped", "shuffle", "run_map_reduce",
    "ExecutionSpec", "IntermediateResult", "IntermediateResultStore", "ResultStatus",
}
EXPECTED_COMMON_API = {"Task", "TaskStatus", "Worker", "WorkerStatus"}


def test_master_public_api_is_exactly_the_frozen_snapshot():
    assert set(master.__all__) == EXPECTED_MASTER_API


def test_worker_public_api_is_exactly_the_frozen_snapshot():
    assert set(worker.__all__) == EXPECTED_WORKER_API


def test_jobs_public_api_is_exactly_the_frozen_snapshot():
    assert set(jobs.__all__) == EXPECTED_JOBS_API


def test_common_public_api_is_exactly_the_frozen_snapshot():
    assert set(common.__all__) == EXPECTED_COMMON_API


# -- exception hierarchy consistency ---------------------------------------

EXPECTED_EXCEPTION_BASES = {
    "TaskNotFoundError": KeyError,
    "NoAvailableWorkerError": RuntimeError,
    "DuplicateWorkerError": ValueError,
    "WorkerNotFoundError": KeyError,
    "ExecutionError": Exception,
    "CallError": ValueError,
    "FunctionAlreadyRegisteredError": ValueError,
}


def test_every_public_exception_subclasses_a_sensible_builtin():
    """A caller doing `except ValueError` / `except KeyError` /
    `except RuntimeError` for a general case should also catch this
    project's own specific exceptions where that makes sense -- verified
    directly, not just asserted in a docstring."""
    for pkg in (master, worker, jobs):
        for name in pkg.__all__:
            obj = getattr(pkg, name)
            if isinstance(obj, type) and issubclass(obj, BaseException):
                assert name in EXPECTED_EXCEPTION_BASES, f"{name} is a new exception not in the audit table"
                assert issubclass(obj, EXPECTED_EXCEPTION_BASES[name])


def test_no_public_exception_is_a_bare_exception_or_baseexception():
    """A bare `raise Exception(...)` (or BaseException) gives a caller no
    way to catch it specifically without also catching everything else --
    every public exception here must be at least one level more specific."""
    for pkg in (master, worker, jobs):
        for name in pkg.__all__:
            obj = getattr(pkg, name)
            if isinstance(obj, type) and issubclass(obj, BaseException):
                assert obj not in (Exception, BaseException)


# -- sync vs async separation is explicit and consistent --------------------

EXPECTED_ASYNC_CALLABLES = {
    (master, "start_master"),
    (master, "stop_master"),
    (master, "wait_for_tasks"),
    (master, "wait_for_workers"),
    (worker, "run_worker"),
    (jobs, "run_map_reduce"),
}


def test_documented_async_functions_are_actually_coroutine_functions():
    for pkg, name in EXPECTED_ASYNC_CALLABLES:
        obj = getattr(pkg, name)
        assert asyncio.iscoroutinefunction(obj), f"{pkg.__name__}.{name} should be async def"


def test_every_other_public_callable_is_plain_sync():
    """Everything in the public API NOT in EXPECTED_ASYNC_CALLABLES above
    must be plain sync -- a caller should never have to guess whether a
    given function needs `await` by trial and error."""
    async_names = {name for _, name in EXPECTED_ASYNC_CALLABLES}
    for pkg in (master, worker, jobs):
        for name in pkg.__all__:
            if name in async_names:
                continue
            obj = getattr(pkg, name)
            if inspect.isfunction(obj) or inspect.ismethod(obj):
                assert not asyncio.iscoroutinefunction(obj), f"{pkg.__name__}.{name} is unexpectedly async"


# -- return-type spot checks (Task/Worker are always the documented type) ---


def test_scheduler_submit_task_returns_a_common_task():
    s = master.Scheduler(master.WorkerManager())
    result = s.submit_task("t1", "ADD", {"a": 1, "b": 1})
    assert isinstance(result, common.Task)


def test_worker_manager_register_worker_returns_a_common_worker():
    wm = master.WorkerManager()
    result = wm.register_worker("w1", "127.0.0.1", 6001)
    assert isinstance(result, common.Worker)


def test_backend_config_and_worker_runtime_config_are_plain_dataclasses():
    """Config objects are inert data, not something with hidden async
    behavior or side effects -- verified by construction, not assumption."""
    import dataclasses

    assert dataclasses.is_dataclass(worker.BackendConfig)
    from worker.config import WorkerRuntimeConfig

    assert dataclasses.is_dataclass(WorkerRuntimeConfig)
