"""Phase 10.4: structured task-failure categories (worker/executor.py's
ExecutionErrorCode). Every execution failure -- from a built-in operation,
a registered function, or a serialized callable -- must carry a "code"
alongside its free-text "message", distinguishing WHICH KIND of failure
occurred rather than leaving a caller to string-match a message.
"""

import math

import pytest

from worker import registry, serialization
from worker.executor import ExecutionErrorCode, execute_task


@pytest.fixture(autouse=True)
def reset_registry():
    registry.clear()
    yield
    registry.clear()


def _serialize(fn) -> str:
    return serialization.encode_for_wire(serialization.serialize_callable(fn))


# -- UNKNOWN_OPERATION ---------------------------------------------------


def test_unregistered_task_type_is_unknown_operation():
    result = execute_task("does_not_exist", {})
    assert result["code"] == ExecutionErrorCode.UNKNOWN_OPERATION


def test_unsupported_map_operation_is_unknown_operation():
    result = execute_task("MAP", {"operation": "NOT_A_REAL_OP", "data": [1, 2]})
    assert result["code"] == ExecutionErrorCode.UNKNOWN_OPERATION


def test_unsupported_reduce_operation_is_unknown_operation():
    result = execute_task("REDUCE", {"operation": "NOT_A_REAL_OP", "values": [1, 2]})
    assert result["code"] == ExecutionErrorCode.UNKNOWN_OPERATION


def test_unknown_registered_operation_via_execute_envelope_is_unknown_operation():
    result = execute_task("EXECUTE", {"execution_mode": "registered", "operation": "missing", "payload": {}})
    assert result["code"] == ExecutionErrorCode.UNKNOWN_OPERATION


def test_unknown_execution_mode_is_unknown_operation():
    result = execute_task("EXECUTE", {"execution_mode": "teleport"})
    assert result["code"] == ExecutionErrorCode.UNKNOWN_OPERATION


# -- INVALID_ARGUMENTS ----------------------------------------------------


def test_add_missing_operand_is_invalid_arguments():
    result = execute_task("ADD", {"a": 1})
    assert result["code"] == ExecutionErrorCode.INVALID_ARGUMENTS


def test_registered_function_non_dict_payload_is_invalid_arguments():
    registry.register_function("noop", lambda: None)
    result = execute_task("noop", "not-a-dict")
    assert result["code"] == ExecutionErrorCode.INVALID_ARGUMENTS


def test_serialized_callable_missing_field_is_invalid_arguments():
    result = execute_task("EXECUTE", {"execution_mode": "serialized_callable"})
    assert result["code"] == ExecutionErrorCode.INVALID_ARGUMENTS


def test_serialized_callable_args_not_a_list_is_invalid_arguments():
    encoded = _serialize(lambda: None)
    result = execute_task(
        "EXECUTE", {"execution_mode": "serialized_callable", "callable": encoded, "args": "nope"}
    )
    assert result["code"] == ExecutionErrorCode.INVALID_ARGUMENTS


# -- DESERIALIZATION_FAILED ------------------------------------------------


def test_malformed_callable_is_deserialization_failed():
    result = execute_task(
        "EXECUTE", {"execution_mode": "serialized_callable", "callable": "not valid base64 or pickle"}
    )
    assert result["code"] == ExecutionErrorCode.DESERIALIZATION_FAILED


def test_valid_base64_but_not_a_pickled_callable_is_deserialization_failed():
    import base64

    garbage = base64.b64encode(b"definitely not a pickle stream").decode("ascii")
    result = execute_task("EXECUTE", {"execution_mode": "serialized_callable", "callable": garbage})
    assert result["code"] == ExecutionErrorCode.DESERIALIZATION_FAILED


# -- USER_FUNCTION_ERROR ----------------------------------------------------


def test_registered_function_raising_is_user_function_error():
    registry.register_function("boom", lambda: 1 / 0)
    result = execute_task("boom", {})
    assert result["code"] == ExecutionErrorCode.USER_FUNCTION_ERROR


def test_registered_function_wrong_arity_is_user_function_error():
    """A missing/extra argument surfaces as Python's own TypeError when
    calling fn(**payload) -- categorized the same as any other exception
    the function raises, since from the contract's perspective it's the
    SAME thing: the call to the user's function failed."""
    registry.register_function("needs_two", lambda a, b: a + b)
    result = execute_task("needs_two", {"a": 1})
    assert result["code"] == ExecutionErrorCode.USER_FUNCTION_ERROR


def test_serialized_callable_raising_is_user_function_error():
    encoded = _serialize(lambda a, b: a / b)
    result = execute_task(
        "EXECUTE", {"execution_mode": "serialized_callable", "callable": encoded, "args": [1, 0]}
    )
    assert result["code"] == ExecutionErrorCode.USER_FUNCTION_ERROR


# -- RESULT_SERIALIZATION_FAILED -------------------------------------------


def test_registered_function_returning_a_plain_object_is_result_serialization_failed():
    registry.register_function("bad", lambda: object())
    result = execute_task("bad", {})
    assert result["code"] == ExecutionErrorCode.RESULT_SERIALIZATION_FAILED


def test_serialized_callable_returning_a_set_is_result_serialization_failed():
    encoded = _serialize(lambda: {1, 2, 3})
    result = execute_task("EXECUTE", {"execution_mode": "serialized_callable", "callable": encoded})
    assert result["code"] == ExecutionErrorCode.RESULT_SERIALIZATION_FAILED


# -- UNSUPPORTED_RETURN_VALUE -----------------------------------------------


def test_registered_function_returning_nan_is_unsupported_return_value():
    registry.register_function("bad_float", lambda: math.nan)
    result = execute_task("bad_float", {})
    assert result["code"] == ExecutionErrorCode.UNSUPPORTED_RETURN_VALUE


def test_registered_function_returning_infinity_is_unsupported_return_value():
    registry.register_function("infinite", lambda: math.inf)
    result = execute_task("infinite", {})
    assert result["code"] == ExecutionErrorCode.UNSUPPORTED_RETURN_VALUE


def test_serialized_callable_returning_negative_infinity_is_unsupported_return_value():
    encoded = _serialize(lambda: -math.inf)
    result = execute_task("EXECUTE", {"execution_mode": "serialized_callable", "callable": encoded})
    assert result["code"] == ExecutionErrorCode.UNSUPPORTED_RETURN_VALUE


def test_nan_nested_inside_a_larger_structure_is_still_unsupported_return_value():
    registry.register_function("partial_nan", lambda: {"ok": 1, "bad": math.nan})
    result = execute_task("partial_nan", {})
    assert result["code"] == ExecutionErrorCode.UNSUPPORTED_RETURN_VALUE


# -- every error response carries both code and message --------------------


def test_every_error_response_has_both_code_and_message():
    result = execute_task("does_not_exist", {})
    assert result["status"] == "error"
    assert isinstance(result["code"], str) and result["code"]
    assert isinstance(result["message"], str) and result["message"]
