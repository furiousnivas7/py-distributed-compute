"""Phase 10.3: the EXECUTE task type's envelope (worker/executor.py's
execute_envelope), exercised through execute_task() -- the same entry
point every other task type goes through, distinguishing 'registered' and
'serialized_callable' execution_mode explicitly rather than inferring
which one a payload shape must mean.
"""

import pytest

from worker import registry, serialization
from worker.executor import execute_task


@pytest.fixture(autouse=True)
def reset_registry():
    registry.clear()
    yield
    registry.clear()


def _envelope(**payload):
    return execute_task("EXECUTE", payload)


# -- registered mode --------------------------------------------------


def test_registered_mode_dispatches_to_the_registry():
    registry.register_function("add", lambda a, b: a + b)
    result = _envelope(execution_mode="registered", operation="add", payload={"a": 2, "b": 3})
    assert result == {"status": "success", "result": 5}


def test_registered_mode_unknown_operation_is_a_controlled_error():
    result = _envelope(execution_mode="registered", operation="missing", payload={})
    assert result["status"] == "error"
    assert "missing" in result["message"]


def test_registered_mode_missing_operation_field_is_a_controlled_error():
    result = _envelope(execution_mode="registered", payload={})
    assert result["status"] == "error"
    assert "operation" in result["message"]


def test_registered_mode_defaults_payload_to_empty():
    registry.register_function("answer", lambda: 42)
    result = _envelope(execution_mode="registered", operation="answer")
    assert result == {"status": "success", "result": 42}


# -- serialized_callable mode ------------------------------------------


def _serialize(fn) -> str:
    return serialization.encode_for_wire(serialization.serialize_callable(fn))


def test_serialized_callable_mode_executes_a_simple_function():
    encoded = _serialize(lambda a, b: a + b)
    result = _envelope(execution_mode="serialized_callable", callable=encoded, args=[2, 3])
    assert result == {"status": "success", "result": 5}


def test_serialized_callable_mode_with_kwargs():
    encoded = _serialize(lambda name, greeting="Hello": f"{greeting}, {name}!")
    result = _envelope(
        execution_mode="serialized_callable", callable=encoded, args=["Ada"], kwargs={"greeting": "Hi"}
    )
    assert result == {"status": "success", "result": "Hi, Ada!"}


def test_serialized_callable_mode_captures_a_closure():
    multiplier = 10
    encoded = _serialize(lambda x: x * multiplier)
    result = _envelope(execution_mode="serialized_callable", callable=encoded, args=[3])
    assert result == {"status": "success", "result": 30}


def test_serialized_callable_mode_missing_callable_field_is_a_controlled_error():
    result = _envelope(execution_mode="serialized_callable", args=[])
    assert result["status"] == "error"
    assert "callable" in result["message"]


def test_serialized_callable_mode_malformed_callable_is_a_controlled_error():
    result = _envelope(execution_mode="serialized_callable", callable="not valid base64 or pickle data")
    assert result["status"] == "error"


def test_serialized_callable_mode_function_raising_an_exception_is_caught():
    encoded = _serialize(lambda a, b: a / b)
    result = _envelope(execution_mode="serialized_callable", callable=encoded, args=[1, 0])
    assert result["status"] == "error"
    assert "ZeroDivisionError" in result["message"]


def test_serialized_callable_mode_non_json_serializable_return_is_a_controlled_error():
    encoded = _serialize(lambda: object())
    result = _envelope(execution_mode="serialized_callable", callable=encoded)
    assert result["status"] == "error"
    assert "not JSON-serializable" in result["message"]


def test_serialized_callable_mode_args_must_be_a_list():
    encoded = _serialize(lambda: None)
    result = _envelope(execution_mode="serialized_callable", callable=encoded, args="not-a-list")
    assert result["status"] == "error"
    assert "args" in result["message"]


def test_serialized_callable_mode_kwargs_must_be_an_object():
    encoded = _serialize(lambda: None)
    result = _envelope(execution_mode="serialized_callable", callable=encoded, kwargs=["not", "a", "dict"])
    assert result["status"] == "error"
    assert "kwargs" in result["message"]


# -- envelope-level validation -------------------------------------------


def test_unknown_execution_mode_is_a_controlled_error():
    result = _envelope(execution_mode="teleport", payload={})
    assert result["status"] == "error"
    assert "teleport" in result["message"]


def test_missing_execution_mode_is_a_controlled_error():
    result = _envelope(operation="add")
    assert result["status"] == "error"
