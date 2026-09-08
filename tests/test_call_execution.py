"""Phase 10.2: registered-function execution (worker/executor.py's
execute_registered, and its fallback wiring inside execute_task), exercised
through the same execute_task() entry point every other task type goes
through. A registered operation's task_type IS its registry name --
there's no wrapper task type -- and its payload IS its keyword arguments.
"""

import pytest

from worker import registry
from worker.executor import execute_task


@pytest.fixture(autouse=True)
def reset_registry():
    registry.clear()
    yield
    registry.clear()


def test_registered_function_dispatched_by_task_type():
    registry.register_function("add", lambda a, b: a + b)
    result = execute_task("add", {"a": 2, "b": 3})
    assert result == {"status": "success", "result": 5}


def test_registered_function_with_default_keyword_argument():
    registry.register_function("greet", lambda name, greeting="Hello": f"{greeting}, {name}!")
    result = execute_task("greet", {"name": "Ada"})
    assert result == {"status": "success", "result": "Hello, Ada!"}


def test_registered_function_with_no_arguments():
    registry.register_function("answer", lambda: 42)
    result = execute_task("answer", {})
    assert result == {"status": "success", "result": 42}


def test_registered_function_returns_complex_json_safe_structure():
    registry.register_function("build", lambda n: {"squares": [i * i for i in range(n)]})
    result = execute_task("build", {"n": 4})
    assert result == {"status": "success", "result": {"squares": [0, 1, 4, 9]}}


def test_unregistered_task_type_is_a_controlled_error():
    result = execute_task("does_not_exist", {})
    assert result["status"] == "error"
    assert "does_not_exist" in result["message"]


def test_registered_function_payload_must_be_an_object():
    registry.register_function("noop", lambda: None)
    result = execute_task("noop", "not-a-dict")
    assert result["status"] == "error"
    assert "object" in result["message"]


def test_registered_function_raising_an_exception_is_caught_and_reported():
    def divide(a, b):
        return a / b

    registry.register_function("divide", divide)
    result = execute_task("divide", {"a": 1, "b": 0})
    assert result["status"] == "error"
    assert "ZeroDivisionError" in result["message"]


def test_registered_function_raising_a_custom_exception_reports_its_type_and_message():
    class MyDomainError(Exception):
        pass

    def picky(x):
        if x < 0:
            raise MyDomainError("x must be non-negative")
        return x

    registry.register_function("picky", picky)
    result = execute_task("picky", {"x": -1})
    assert result["status"] == "error"
    assert "MyDomainError" in result["message"]
    assert "x must be non-negative" in result["message"]


def test_registered_function_non_json_serializable_return_value_is_a_controlled_error():
    registry.register_function("bad_return", lambda: object())
    result = execute_task("bad_return", {})
    assert result["status"] == "error"
    assert "not JSON-serializable" in result["message"]


def test_registered_function_missing_required_argument_is_caught_as_typeerror():
    registry.register_function("needs_two", lambda a, b: a + b)
    result = execute_task("needs_two", {"a": 1})
    assert result["status"] == "error"
    assert "TypeError" in result["message"]


def test_registered_function_unexpected_argument_is_caught_as_typeerror():
    registry.register_function("needs_one", lambda a: a)
    result = execute_task("needs_one", {"a": 1, "unexpected": 2})
    assert result["status"] == "error"
    assert "TypeError" in result["message"]


def test_built_in_operations_take_priority_over_the_registry():
    """ADD is a fixed built-in handler; even if something registered a
    function under that exact name, the built-in must still win -- the
    registry is a fallback for what HANDLERS doesn't already cover, not
    an override mechanism."""
    registry.register_function("ADD", lambda **kwargs: "registry-not-builtin")
    result = execute_task("ADD", {"a": 2, "b": 3})
    assert result == {"status": "success", "result": 5}


def test_registered_via_decorator_is_dispatched_the_same_way():
    @registry.register("cube")
    def cube(x):
        return x * x * x

    result = execute_task("cube", {"x": 3})
    assert result == {"status": "success", "result": 27}
