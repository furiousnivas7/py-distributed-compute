"""Phase 11.6: backend diagnostics -- describe() (identity/capability
metadata) and structured logging (worker.backend's module logger) for
DirectBackend and MultiprocessingBackend. All of this is diagnostic-only:
these tests also confirm it never changes metrics, retry-relevant results,
or logs anything sensitive (payload contents, serialized callable bytes).
"""

import asyncio
import json
import logging
from unittest.mock import patch

import pytest

from worker import registry
from worker.backend import DirectBackend, ExecutionBackend, MultiprocessingBackend
from worker.executor import ExecutionErrorCode


@pytest.fixture(autouse=True)
def reset_registry():
    registry.clear()
    yield
    registry.clear()


def _raise_value_error():
    raise ValueError("deliberate failure")


def _crash():
    import os

    os._exit(1)


# -- describe() -----------------------------------------------------------


def test_describe_for_direct_backend():
    info = DirectBackend().describe()
    assert info == {
        "backend": "DirectBackend",
        "max_concurrency": None,
        "supports_serialized_callables": True,
    }


def test_describe_for_multiprocessing_backend():
    info = MultiprocessingBackend(max_workers=4, max_in_flight=8).describe()
    assert info == {
        "backend": "MultiprocessingBackend",
        "max_concurrency": 8,
        "supports_serialized_callables": True,
        "max_workers": 4,
        "max_in_flight": 8,
    }


def test_describe_is_json_serializable():
    json.dumps(DirectBackend().describe())
    json.dumps(MultiprocessingBackend(max_workers=2, max_in_flight=1).describe())


def test_base_execution_backend_describe_default():
    class Minimal(ExecutionBackend):
        async def execute(self, task_type: str, payload: dict) -> dict:
            return {"status": "success", "result": None}

    assert Minimal().describe() == {
        "backend": "Minimal",
        "max_concurrency": None,
        "supports_serialized_callables": True,
    }


# -- lifecycle logs on successful start/stop --------------------------


def test_lifecycle_logs_on_successful_start_and_stop(caplog):
    async def scenario():
        backend = MultiprocessingBackend(max_workers=2)
        await backend.start()
        await backend.stop()

    with caplog.at_level(logging.INFO, logger="worker.backend"):
        asyncio.run(scenario())

    messages = [r.getMessage() for r in caplog.records]
    assert any("backend_created" in m for m in messages)
    assert any("backend_started" in m for m in messages)
    assert any("backend_stopping" in m for m in messages)
    assert any("backend_stopped" in m for m in messages)


# -- startup failure logging --------------------------------------------


def test_startup_failure_is_logged_as_an_error(caplog):
    backend = MultiprocessingBackend(max_workers=2)

    async def scenario():
        with patch(
            "worker.backend.ProcessPoolExecutor",
            side_effect=RuntimeError("deliberate pool construction failure"),
        ):
            with pytest.raises(RuntimeError, match="deliberate pool construction failure"):
                await backend.start()

    with caplog.at_level(logging.INFO, logger="worker.backend"):
        asyncio.run(scenario())

    error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert any("backend_start_failed" in r.getMessage() for r in error_records)


# -- execution failure logging -------------------------------------------


def test_structured_execution_failure_is_logged_as_a_warning_direct_backend(caplog):
    async def scenario():
        backend = DirectBackend()
        return await backend.execute("does_not_exist", {})

    with caplog.at_level(logging.DEBUG, logger="worker.backend"):
        result = asyncio.run(scenario())

    assert result["code"] == ExecutionErrorCode.UNKNOWN_OPERATION
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("task_execution_failed" in r.getMessage() for r in warnings)
    assert any(ExecutionErrorCode.UNKNOWN_OPERATION in r.getMessage() for r in warnings)


def test_structured_execution_failure_is_logged_as_a_warning_multiprocessing_backend(caplog):
    registry.register_function("boom", _raise_value_error)

    async def scenario():
        backend = MultiprocessingBackend(max_workers=2)
        await backend.start()
        try:
            return await backend.execute("boom", {})
        finally:
            await backend.stop()

    with caplog.at_level(logging.DEBUG, logger="worker.backend"):
        result = asyncio.run(scenario())

    assert result["status"] == "error"
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("task_execution_failed" in r.getMessage() for r in warnings)


# -- process-pool failure logging -----------------------------------------


def test_process_pool_failure_is_logged_as_an_error_with_diagnostic_fields(caplog):
    registry.register_function("crash", _crash)

    async def scenario():
        backend = MultiprocessingBackend(max_workers=2, max_in_flight=4)
        await backend.start()
        try:
            with pytest.raises(Exception):
                await backend.execute("crash", {})
        finally:
            await backend.stop()

    with caplog.at_level(logging.DEBUG, logger="worker.backend"):
        asyncio.run(scenario())

    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    matching = [r for r in errors if "process_pool_failed" in r.getMessage()]
    assert matching, "expected a process_pool_failed error log"
    message = matching[0].getMessage()
    assert "task_type=crash" in message
    assert "max_workers=2" in message
    assert "max_in_flight=4" in message
    assert "error=" in message


# -- configuration errors include received values --------------------


def test_max_workers_error_includes_the_received_value():
    with pytest.raises(ValueError, match="received -2"):
        MultiprocessingBackend(max_workers=-2)


def test_max_in_flight_error_includes_the_received_value():
    with pytest.raises(ValueError, match="received 0"):
        MultiprocessingBackend(max_workers=2, max_in_flight=0)


def test_unsupported_mp_context_is_rejected_with_an_actionable_message():
    with pytest.raises(ValueError, match="mp_context"):
        MultiprocessingBackend(mp_context="not-a-real-context")


def test_unsupported_mp_context_error_includes_the_received_value():
    with pytest.raises(ValueError, match="not-a-real-context"):
        MultiprocessingBackend(mp_context="not-a-real-context")


def test_execution_after_shutdown_error_is_actionable():
    async def scenario():
        backend = MultiprocessingBackend(max_workers=2)
        await backend.start()
        await backend.stop()
        with pytest.raises(RuntimeError, match="after stop"):
            await backend.execute("ADD", {"a": 1, "b": 1})

    asyncio.run(scenario())


# -- logs do not include callable payloads / sensitive content --------


def test_logs_never_include_serialized_callable_bytes_or_payload_contents(caplog):
    from jobs.call import submit_serialized_call
    from master.scheduler import Scheduler
    from master.worker_manager import WorkerManager

    SENTINEL = "super-secret-argument-xyz123"

    def echo(x):
        return x

    async def scenario():
        backend = MultiprocessingBackend(max_workers=2)
        await backend.start()
        try:
            scheduler = Scheduler(WorkerManager())
            task = submit_serialized_call(scheduler, "t1", echo, args=[SENTINEL])
            return await backend.execute(task.task_type, task.payload)
        finally:
            await backend.stop()

    with caplog.at_level(logging.DEBUG, logger="worker.backend"):
        result = asyncio.run(scenario())

    assert result == {"status": "success", "result": SENTINEL}
    for record in caplog.records:
        assert SENTINEL not in record.getMessage()


def test_execution_failure_log_never_includes_the_error_message_body(caplog):
    """Only the structured `code` is logged, never `message` -- message
    text can echo caller-supplied argument values back (see
    execute_task's own error formatting)."""
    registry.register_function("boom", _raise_value_error)

    async def scenario():
        backend = DirectBackend()
        return await backend.execute("boom", {})

    with caplog.at_level(logging.DEBUG, logger="worker.backend"):
        result = asyncio.run(scenario())

    # The result itself DOES carry the exception text (existing execute_task
    # contract, unchanged) -- only the LOG must never repeat it.
    assert "deliberate failure" in result["message"]
    for record in caplog.records:
        assert "deliberate failure" not in record.getMessage()


# -- diagnostics do not alter metrics -------------------------------------


def test_diagnostics_do_not_alter_metrics(caplog):
    registry.register_function("noop", lambda: "ok")

    async def scenario():
        backend = MultiprocessingBackend(max_workers=2)
        await backend.start()
        try:
            await backend.execute("noop", {})
        finally:
            await backend.stop()
        return backend.get_metrics()

    with caplog.at_level(logging.DEBUG, logger="worker.backend"):
        metrics_with_logging = asyncio.run(scenario())

    metrics_without_logging = asyncio.run(scenario())

    assert metrics_with_logging.submitted == metrics_without_logging.submitted == 1
    assert metrics_with_logging.completed == metrics_without_logging.completed == 1


# -- diagnostics do not alter retry behavior -------------------------------


def test_diagnostics_do_not_alter_retry_relevant_result_shape(caplog):
    """describe()/logging must never change what execute() actually
    returns -- the master's retry decision is driven entirely by that
    result (or the absence of one, on a raise), never by anything logged."""
    registry.register_function("boom", _raise_value_error)

    async def scenario():
        backend = DirectBackend()
        return await backend.execute("boom", {})

    with caplog.at_level(logging.DEBUG, logger="worker.backend"):
        result_with_logging = asyncio.run(scenario())

    result_without_logging = asyncio.run(scenario())

    assert result_with_logging == result_without_logging
    assert result_with_logging["status"] == "error"
    assert result_with_logging["code"] == ExecutionErrorCode.USER_FUNCTION_ERROR
