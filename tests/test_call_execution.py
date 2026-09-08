"""Phase 10.1: the CALL task type's execution contract (worker/executor.py's
execute_call), exercised through the same execute_task() entry point every
other task type goes through.
"""

import pytest

from worker import registry
from worker.executor import execute_task


@pytest.fixture(autouse=True)
def reset_registry():
    registry.clear()
    yield
    registry.clear()


def test_call_with_positional_args():
    registry.register_function("add", lambda a, b: a + b)
    result = execute_task("CALL", {"function": "add", "args": [2, 3]})
    assert result == {"status": "success", "result": 5}


def test_call_with_keyword_args():
    registry.register_function("greet", lambda name, greeting="Hello": f"{greeting}, {name}!")
    result = execute_task("CALL", {"function": "greet", "args": ["Ada"], "kwargs": {"greeting": "Hi"}})
    assert result == {"status": "success", "result": "Hi, Ada!"}


def test_call_with_no_args():
    registry.register_function("answer", lambda: 42)
    result = execute_task("CALL", {"function": "answer"})
    assert result == {"status": "success", "result": 42}


def test_call_returns_complex_json_safe_structure():
    registry.register_function("build", lambda n: {"squares": [i * i for i in range(n)]})
    result = execute_task("CALL", {"function": "build", "args": [4]})
    assert result == {"status": "success", "result": {"squares": [0, 1, 4, 9]}}


def test_call_unknown_function_is_a_controlled_error():
    result = execute_task("CALL", {"function": "does_not_exist", "args": []})
    assert result["status"] == "error"
    assert "does_not_exist" in result["message"]


def test_call_missing_function_field_is_a_controlled_error():
    result = execute_task("CALL", {"args": [1, 2]})
    assert result["status"] == "error"
    assert "function" in result["message"]


def test_call_args_must_be_a_list():
    registry.register_function("noop", lambda: None)
    result = execute_task("CALL", {"function": "noop", "args": "not-a-list"})
    assert result["status"] == "error"
    assert "args" in result["message"]


def test_call_kwargs_must_be_an_object():
    registry.register_function("noop", lambda: None)
    result = execute_task("CALL", {"function": "noop", "kwargs": ["not", "a", "dict"]})
    assert result["status"] == "error"
    assert "kwargs" in result["message"]


def test_call_function_raising_an_exception_is_caught_and_reported():
    def divide(a, b):
        return a / b

    registry.register_function("divide", divide)
    result = execute_task("CALL", {"function": "divide", "args": [1, 0]})
    assert result["status"] == "error"
    assert "ZeroDivisionError" in result["message"]


def test_call_function_raising_a_custom_exception_reports_its_type_and_message():
    class MyDomainError(Exception):
        pass

    def picky(x):
        if x < 0:
            raise MyDomainError("x must be non-negative")
        return x

    registry.register_function("picky", picky)
    result = execute_task("CALL", {"function": "picky", "args": [-1]})
    assert result["status"] == "error"
    assert "MyDomainError" in result["message"]
    assert "x must be non-negative" in result["message"]


def test_call_non_json_serializable_return_value_is_a_controlled_error():
    registry.register_function("bad_return", lambda: object())
    result = execute_task("CALL", {"function": "bad_return", "args": []})
    assert result["status"] == "error"
    assert "not JSON-serializable" in result["message"]


def test_call_wrong_arity_is_caught_as_the_functions_own_exception():
    registry.register_function("needs_two", lambda a, b: a + b)
    result = execute_task("CALL", {"function": "needs_two", "args": [1]})
    assert result["status"] == "error"
    assert "TypeError" in result["message"]
