"""Phase 12.1: the public API surface (master/__init__.py, worker/__init__.py,
jobs/__init__.py, common/__init__.py) -- these packages are purely additive
re-export layers on top of already-existing modules (see each __init__.py's
own docstring), so every test here is about the SURFACE itself: what's
importable, that it's exactly what __all__ declares (nothing accidental),
that it's the SAME object a deep import already gave you (no duplication/
divergence), and that the README's own "Getting Started" examples actually
run correctly end to end.
"""

import asyncio
import types

import pytest

import common
import jobs
import master
import worker
from worker import registry


@pytest.fixture(autouse=True)
def reset_state():
    registry.clear()
    master.scheduler.clear()
    from master import async_server, rpc_handler

    rpc_handler.worker_manager.clear()
    async_server.connections.clear()
    yield
    registry.clear()


# -- public-import tests ---------------------------------------------------


def _non_module_public_names(pkg):
    """dir(package) always includes every submodule Python has bound as an
    attribute once imported (e.g. `master.async_server`, set automatically
    the moment `from master.async_server import X` runs inside
    master/__init__.py) -- that's normal Python import-system behavior,
    not an accidental export, and every package with submodules has it.
    __all__ governs the CURATED surface (what `from package import *`
    pulls in, and what this test actually checks for "nothing accidental"
    beyond that): filter out submodules themselves before comparing."""
    return {
        name
        for name in dir(pkg)
        if not name.startswith("_") and not isinstance(getattr(pkg, name), types.ModuleType)
    }


def test_master_exposes_exactly_its_declared_all():
    assert _non_module_public_names(master) == set(master.__all__)


def test_worker_exposes_exactly_its_declared_all():
    assert _non_module_public_names(worker) == set(worker.__all__)


def test_jobs_exposes_exactly_its_declared_all():
    assert _non_module_public_names(jobs) == set(jobs.__all__)


def test_common_exposes_exactly_its_declared_all():
    assert _non_module_public_names(common) == set(common.__all__)


def test_every_declared_name_is_actually_importable():
    for name in master.__all__:
        assert hasattr(master, name)
    for name in worker.__all__:
        assert hasattr(worker, name)
    for name in jobs.__all__:
        assert hasattr(jobs, name)
    for name in common.__all__:
        assert hasattr(common, name)


# -- backward compatibility: public re-export IS the same object -----------


def test_master_scheduler_class_is_the_same_object_as_the_deep_import():
    from master.scheduler import Scheduler as DeepScheduler

    assert master.Scheduler is DeepScheduler


def test_master_singleton_scheduler_is_the_same_object_as_async_server_scheduler():
    from master import async_server

    assert master.scheduler is async_server.scheduler


def test_worker_direct_backend_is_the_same_object_as_the_deep_import():
    from worker.backend import DirectBackend as DeepDirectBackend

    assert worker.DirectBackend is DeepDirectBackend


def test_worker_multiprocessing_backend_is_the_same_object_as_the_deep_import():
    from worker.backend import MultiprocessingBackend as DeepMultiprocessingBackend

    assert worker.MultiprocessingBackend is DeepMultiprocessingBackend


def test_jobs_execution_spec_is_the_same_object_as_the_deep_import():
    from jobs.models import ExecutionSpec as DeepExecutionSpec

    assert jobs.ExecutionSpec is DeepExecutionSpec


def test_existing_deep_imports_still_work_unchanged():
    """Every deep-import style used throughout this project's own test
    suite (master.scheduler.Scheduler, worker.backend.DirectBackend, ...)
    must keep working exactly as before -- these __init__.py files are
    additive re-exports, never a move/rename."""
    from jobs.call import submit_call
    from master.scheduler import Scheduler
    from master.worker_manager import WorkerManager
    from worker.backend import DirectBackend
    from worker.registry import register_function

    assert callable(submit_call)
    assert callable(Scheduler)
    assert callable(WorkerManager)
    assert callable(DirectBackend)
    assert callable(register_function)


# -- API contract tests -----------------------------------------------------


def test_scheduler_has_the_documented_task_submission_contract():
    s = master.Scheduler(master.WorkerManager())
    assert hasattr(s, "submit_task")
    assert callable(s.submit_task)


def test_execution_backend_is_an_abstract_base_class():
    with pytest.raises(TypeError):
        worker.ExecutionBackend()


def test_direct_and_multiprocessing_backends_are_execution_backends():
    assert isinstance(worker.DirectBackend(), worker.ExecutionBackend)
    assert isinstance(worker.MultiprocessingBackend(), worker.ExecutionBackend)


def test_execution_spec_construction_contract():
    registered = jobs.ExecutionSpec.registered("SUM")
    serialized = jobs.ExecutionSpec.serialized(lambda x: x)
    assert registered.execution_mode == "registered"
    assert serialized.execution_mode == "serialized_callable"
    assert jobs.ExecutionSpec.coerce("SUM") == registered


# -- lifecycle + README doc-example tests ------------------------------


def test_getting_started_registered_function_example():
    """The README's first Getting Started snippet, executed for real:
    register a function, start a worker via DirectBackend, submit a call
    through the public jobs API, collect the result."""
    worker.register_function("double", lambda x: x * 2)

    async def scenario():
        server = await master.start_master("127.0.0.1", 0)
        try:
            host, port = server.sockets[0].getsockname()[:2]
            worker_task = asyncio.create_task(
                worker.run_worker(host, port, worker_id="worker-1", backend=worker.DirectBackend())
            )
            from master import async_server

            await async_server.wait_for_workers(1)

            task = jobs.submit_call(master.scheduler, "t1", "double", x=21)
            [response] = await master.wait_for_tasks({task.task_id})
            result = jobs.collect_call_result(response)

            worker_task.cancel()
            try:
                await worker_task
            except (asyncio.CancelledError, Exception):
                pass
            return result
        finally:
            await master.stop_master()

    result = asyncio.run(scenario())
    assert result == 42


def test_getting_started_serialized_callable_example():
    async def scenario():
        server = await master.start_master("127.0.0.1", 0)
        try:
            host, port = server.sockets[0].getsockname()[:2]
            worker_task = asyncio.create_task(
                worker.run_worker(host, port, worker_id="worker-1", backend=worker.DirectBackend())
            )
            from master import async_server

            await async_server.wait_for_workers(1)

            task = jobs.submit_serialized_call(master.scheduler, "t2", lambda x: x * x, args=[6])
            [response] = await master.wait_for_tasks({task.task_id})
            result = jobs.collect_call_result(response)

            worker_task.cancel()
            try:
                await worker_task
            except (asyncio.CancelledError, Exception):
                pass
            return result
        finally:
            await master.stop_master()

    result = asyncio.run(scenario())
    assert result == 36


def test_getting_started_map_reduce_example():
    async def scenario():
        server = await master.start_master("127.0.0.1", 0)
        try:
            host, port = server.sockets[0].getsockname()[:2]
            worker_task = asyncio.create_task(
                worker.run_worker(host, port, worker_id="worker-1", backend=worker.DirectBackend())
            )
            from master import async_server

            await async_server.wait_for_workers(1)

            result = await jobs.run_map_reduce(
                master.scheduler,
                async_server.drain_tasks_for,
                "job-1",
                data=["a", "b", "a", "c", "b", "c"],
                map_operation="WORD_COUNT",
                reduce_operation="SUM",
                num_partitions=2,
            )

            worker_task.cancel()
            try:
                await worker_task
            except (asyncio.CancelledError, Exception):
                pass
            return result
        finally:
            await master.stop_master()

    result = asyncio.run(scenario())
    assert result == {"a": 2, "b": 2, "c": 2}


def test_getting_started_multiprocessing_backend_example():
    worker.register_function("triple", lambda x: x * 3)

    async def scenario():
        server = await master.start_master("127.0.0.1", 0)
        try:
            host, port = server.sockets[0].getsockname()[:2]
            backend = worker.MultiprocessingBackend(max_workers=2, max_in_flight=2)
            worker_task = asyncio.create_task(
                worker.run_worker(host, port, worker_id="worker-1", backend=backend)
            )
            from master import async_server

            await async_server.wait_for_workers(1)

            task = jobs.submit_call(master.scheduler, "t3", "triple", x=5)
            [response] = await master.wait_for_tasks({task.task_id})
            result = jobs.collect_call_result(response)

            worker_task.cancel()
            try:
                await worker_task
            except (asyncio.CancelledError, Exception):
                pass
            return result
        finally:
            await master.stop_master()

    result = asyncio.run(scenario())
    assert result == 15
