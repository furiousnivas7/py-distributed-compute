"""Phase 11.1: the execution-backend abstraction (worker/backend.py),
in isolation from any real connection or worker process.
"""

import asyncio

import pytest

from worker import registry
from worker.backend import DirectBackend, ExecutionBackend
from worker.executor import execute_task


@pytest.fixture(autouse=True)
def reset_registry():
    registry.clear()
    yield
    registry.clear()


def test_execution_backend_cannot_be_instantiated_directly():
    """ABC with an abstract execute() -- enforces that every backend
    implementation actually provides one, at class-definition/instantiation
    time rather than failing later with an AttributeError mid-task."""
    with pytest.raises(TypeError):
        ExecutionBackend()


def test_a_backend_missing_execute_cannot_be_instantiated_either():
    class Incomplete(ExecutionBackend):
        pass

    with pytest.raises(TypeError):
        Incomplete()


def test_direct_backend_success_matches_execute_task_exactly():
    async def scenario():
        backend = DirectBackend()
        return await backend.execute("ADD", {"a": 2, "b": 3})

    result = asyncio.run(scenario())
    assert result == execute_task("ADD", {"a": 2, "b": 3})
    assert result == {"status": "success", "result": 5}


def test_direct_backend_failure_matches_execute_task_exactly():
    async def scenario():
        backend = DirectBackend()
        return await backend.execute("does_not_exist", {})

    result = asyncio.run(scenario())
    assert result == execute_task("does_not_exist", {})
    assert result["status"] == "error"
    assert result["code"] == "UNKNOWN_OPERATION"


def test_direct_backend_runs_registered_functions():
    registry.register_function("double", lambda x: x * 2)

    async def scenario():
        backend = DirectBackend()
        return await backend.execute("double", {"x": 21})

    result = asyncio.run(scenario())
    assert result == {"status": "success", "result": 42}


def test_direct_backend_start_and_stop_are_harmless_no_ops():
    async def scenario():
        backend = DirectBackend()
        await backend.start()
        result = await backend.execute("ADD", {"a": 1, "b": 1})
        await backend.stop()
        # Safe to call again -- no state to corrupt.
        await backend.start()
        await backend.stop()
        return result

    result = asyncio.run(scenario())
    assert result == {"status": "success", "result": 2}


def test_a_custom_backend_can_override_only_execute():
    """start()/stop() default to no-ops -- a minimal custom backend only
    needs to implement execute(), proving the interface doesn't force
    unnecessary boilerplate on a backend that has no setup/teardown."""

    class UppercaseEchoBackend(ExecutionBackend):
        async def execute(self, task_type: str, payload: dict) -> dict:
            return {"status": "success", "result": task_type.upper()}

    async def scenario():
        backend = UppercaseEchoBackend()
        await backend.start()
        result = await backend.execute("hello", {})
        await backend.stop()
        return result

    result = asyncio.run(scenario())
    assert result == {"status": "success", "result": "HELLO"}
