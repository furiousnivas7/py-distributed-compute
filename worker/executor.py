"""Executes tasks on the worker and produces a result payload."""

import json

from worker import registry, serialization

ADD = "ADD"
MULTIPLY = "MULTIPLY"
MAP = "MAP"
REDUCE = "REDUCE"
EXECUTE = "EXECUTE"

# Named operations only -- no arbitrary Python function serialization.
# Keeps the wire protocol a fixed, deterministic vocabulary rather than
# shipping code between processes.
NUMERIC_MAP_OPERATIONS = {
    "SQUARE": lambda x: x * x,
    "DOUBLE": lambda x: x * 2,
    "INCREMENT": lambda x: x + 1,
    "NEGATE": lambda x: -x,
}

# Key/value-emitting operations, for jobs that feed into Shuffle/Reduce
# (jobs/shuffle.py) rather than producing a flat transformed list.
KEY_VALUE_MAP_OPERATIONS = {
    "WORD_COUNT": lambda word: [word, 1],
}

MAP_OPERATIONS = {**NUMERIC_MAP_OPERATIONS, **KEY_VALUE_MAP_OPERATIONS}

REDUCE_OPERATIONS = {
    "SUM": sum,
    "COUNT": len,
    "MAX": max,
    "MIN": min,
}


class ExecutionErrorCode:
    """Phase 10.4: structured task-failure categories.

    Distinct from the transport-level error codes an ERROR message can
    carry (WORKER_UNREACHABLE, DUPLICATE_WORKER, ...) -- those describe
    the DISPATCH failing to reach a worker or complete an RPC at all; an
    ExecutionErrorCode describes a task that a worker DID receive and DID
    run to a (failed) conclusion, replying normally with TASK_RESULT
    {"status": "error", "code": ..., "message": ...}. A caller that wants
    "was this even attempted" vs "did the attempt fail, and how" can tell
    the two apart by which field carries the code (a bare ERROR message's
    payload vs a TASK_RESULT payload's "code"), not just by string-matching
    a free-text message.
    """

    UNKNOWN_OPERATION = "UNKNOWN_OPERATION"
    DESERIALIZATION_FAILED = "DESERIALIZATION_FAILED"
    INVALID_ARGUMENTS = "INVALID_ARGUMENTS"
    USER_FUNCTION_ERROR = "USER_FUNCTION_ERROR"
    RESULT_SERIALIZATION_FAILED = "RESULT_SERIALIZATION_FAILED"
    UNSUPPORTED_RETURN_VALUE = "UNSUPPORTED_RETURN_VALUE"


class ExecutionError(Exception):
    """Raised when a task's payload cannot be executed. Always carries a
    `code` (one of ExecutionErrorCode) alongside the human-readable
    message -- defaulting to INVALID_ARGUMENTS, the most common case
    (malformed/missing payload fields), so existing call sites that don't
    need a different category don't have to name one explicitly."""

    def __init__(self, message: str, code: str = ExecutionErrorCode.INVALID_ARGUMENTS):
        super().__init__(message)
        self.code = code
        self.message = message


def _check_result_is_json_safe(result, operation: str):
    """Shared by every execution path that produces a user-derived return
    value. Two distinct failure categories, deliberately not merged:

    - RESULT_SERIALIZATION_FAILED: the value's TYPE can't be represented
      in JSON at all (e.g. a plain object(), a set, a datetime) -- caught
      as the TypeError json.dumps raises for those.
    - UNSUPPORTED_RETURN_VALUE: the value IS JSON-encodable by Python's
      json module, but not by the JSON *specification* -- NaN/Infinity/
      -Infinity floats round-trip fine through json.dumps(allow_nan=True,
      the default) as the bare tokens NaN/Infinity, which aren't valid
      JSON per RFC 8259 and could break a stricter receiver on the other
      end. allow_nan=False turns that into a ValueError we catch
      separately, so a submitter gets told WHICH kind of "can't send
      this back" problem they have, not a single generic one.
    """
    try:
        json.dumps(result, allow_nan=False)
    except TypeError as exc:
        # Wrong type entirely (object(), a set, a datetime, ...) -- json
        # can't represent it under any circumstances.
        raise ExecutionError(
            f"Return value of {operation!r} is not JSON-serializable: {exc}",
            code=ExecutionErrorCode.RESULT_SERIALIZATION_FAILED,
        ) from exc
    except ValueError as exc:
        # allow_nan=False specifically: the value round-trips through
        # Python's json module but isn't valid per the JSON spec
        # (NaN/Infinity/-Infinity).
        raise ExecutionError(
            f"Return value of {operation!r} is not valid JSON (NaN/Infinity are not allowed): {exc}",
            code=ExecutionErrorCode.UNSUPPORTED_RETURN_VALUE,
        ) from exc


def execute_add(payload: dict):
    a, b = _require_numbers(payload)
    return a + b


def execute_multiply(payload: dict):
    a, b = _require_numbers(payload)
    return a * b


def execute_map(payload: dict):
    operation = payload.get("operation")
    data = payload.get("data")

    if operation not in MAP_OPERATIONS:
        raise ExecutionError(f"Unsupported MAP operation: {operation}", code=ExecutionErrorCode.UNKNOWN_OPERATION)

    if not isinstance(data, list):
        raise ExecutionError("payload must contain a list field 'data'")

    fn = MAP_OPERATIONS[operation]

    if operation in KEY_VALUE_MAP_OPERATIONS:
        mapped = []
        for value in data:
            if not isinstance(value, str) or not value:
                raise ExecutionError(f"{operation} data must contain only non-empty strings")
            mapped.append(fn(value))
        return mapped

    mapped = []
    for value in data:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ExecutionError("MAP data must contain only numeric values")
        mapped.append(fn(value))
    return mapped


def execute_reduce(payload: dict):
    operation = payload.get("operation")
    values = payload.get("values")

    if operation not in REDUCE_OPERATIONS:
        raise ExecutionError(f"Unsupported REDUCE operation: {operation}", code=ExecutionErrorCode.UNKNOWN_OPERATION)

    if not isinstance(values, list) or not values:
        raise ExecutionError("payload must contain a non-empty list field 'values'")

    for value in values:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ExecutionError("REDUCE values must contain only numeric values")

    return REDUCE_OPERATIONS[operation](values)


def execute_registered(operation: str, payload: dict):
    """Phase 10.2: invoke a pre-registered function (see worker/registry.py)
    by operation name, calling it as fn(**payload) -- the task's payload
    IS the function's keyword arguments, directly off the JSON-decoded
    wire message, no positional-args/kwargs wrapper to design or get
    wrong.

    Every failure mode is reported the same controlled way (ExecutionError
    -> {"status": "error", "code": ..., ...}, never an unhandled exception
    that could crash the worker process), each with its own
    ExecutionErrorCode (Phase 10.4):
      - UNKNOWN_OPERATION: no function registered under this name
      - INVALID_ARGUMENTS: payload isn't an object (can't be spread as
        keyword arguments)
      - USER_FUNCTION_ERROR: the function's OWN exception (wrapped with
        its type name so the caller can tell "my function raised
        ValueError" from "the contract itself was violated") -- this also
        naturally covers wrong/missing arguments, which surface as
        Python's own TypeError
      - RESULT_SERIALIZATION_FAILED / UNSUPPORTED_RETURN_VALUE: a return
        value that can't survive the JSON wire protocol -- caught HERE,
        on the worker, rather than failing confusingly deep inside
        send_message on the way out.
    """
    fn = registry.get_function(operation)
    if fn is None:
        raise ExecutionError(f"Unsupported task type: {operation}", code=ExecutionErrorCode.UNKNOWN_OPERATION)

    if not isinstance(payload, dict):
        raise ExecutionError("payload must be an object of keyword arguments")

    try:
        result = fn(**payload)
    except Exception as exc:
        raise ExecutionError(
            f"{type(exc).__name__}: {exc}", code=ExecutionErrorCode.USER_FUNCTION_ERROR
        ) from exc

    _check_result_is_json_safe(result, operation)
    return result


def execute_envelope(payload: dict):
    """Phase 10.3: the unified, explicit entry point for function-execution
    tasks (task_type EXECUTE). Distinguishes two execution modes by an
    explicit 'execution_mode' discriminator rather than inferring which
    one a payload shape must mean:

        {"execution_mode": "registered", "operation": "add",
         "payload": {"a": 1, "b": 2}}

        {"execution_mode": "serialized_callable", "callable": "<base64>",
         "args": [...], "kwargs": {...}}

    'registered' delegates straight to execute_registered (Phase 10.2) --
    same lookup, same fn(**payload) calling convention, just reached via
    an explicit envelope instead of task_type doubling as the operation
    name. 'serialized_callable' is the genuinely new capability this
    phase adds: an arbitrary Python callable, serialized by the SUBMITTER
    (see worker/serialization.py and jobs/call.py's submit_serialized_call),
    shipped over the wire, and executed here. See worker/serialization.py's
    module docstring for the trust-boundary implications of that -- this
    function itself does no sandboxing; it only isolates the (de)serialize
    step through that module rather than calling cloudpickle directly.
    """
    mode = payload.get("execution_mode")

    if mode == "registered":
        operation = payload.get("operation")
        if not isinstance(operation, str) or not operation:
            raise ExecutionError("payload must contain a non-empty string field 'operation'")
        return execute_registered(operation, payload.get("payload", {}))

    if mode == "serialized_callable":
        return _execute_serialized_callable(payload)

    raise ExecutionError(f"Unsupported execution_mode: {mode!r}", code=ExecutionErrorCode.UNKNOWN_OPERATION)


def _execute_serialized_callable(payload: dict):
    """See execute_registered's docstring for the shared ExecutionErrorCode
    categories; this path additionally has DESERIALIZATION_FAILED, for a
    'callable' field that isn't valid base64 or doesn't decode back into a
    callable object (see worker/serialization.py)."""
    encoded = payload.get("callable")
    args = payload.get("args", [])
    kwargs = payload.get("kwargs", {})

    if not isinstance(encoded, str) or not encoded:
        raise ExecutionError("payload must contain a non-empty string field 'callable'")
    if not isinstance(args, list):
        raise ExecutionError("payload field 'args' must be a list")
    if not isinstance(kwargs, dict):
        raise ExecutionError("payload field 'kwargs' must be an object")

    try:
        fn = serialization.deserialize_callable(serialization.decode_from_wire(encoded))
    except serialization.SerializationError as exc:
        raise ExecutionError(str(exc), code=ExecutionErrorCode.DESERIALIZATION_FAILED) from exc

    try:
        result = fn(*args, **kwargs)
    except Exception as exc:
        raise ExecutionError(
            f"{type(exc).__name__}: {exc}", code=ExecutionErrorCode.USER_FUNCTION_ERROR
        ) from exc

    _check_result_is_json_safe(result, "serialized_callable")
    return result


def _require_numbers(payload: dict):
    a = payload.get("a")
    b = payload.get("b")
    for value in (a, b):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ExecutionError("payload must contain numeric fields 'a' and 'b'")
    return a, b


HANDLERS = {
    ADD: execute_add,
    MULTIPLY: execute_multiply,
    MAP: execute_map,
    REDUCE: execute_reduce,
    EXECUTE: execute_envelope,
}


def execute_task(task_type: str, payload: dict) -> dict:
    """Built-in operations (HANDLERS) take priority; a task_type that
    isn't one of them falls back to worker.registry -- a task_type IS an
    operation name either way, whether it's a fixed vocabulary entry or a
    user-registered function (Phase 10.2), so both resolve through this
    one entry point rather than needing a caller to know which kind of
    operation they're submitting.

    An error result always carries a "code" (Phase 10.4, ExecutionErrorCode)
    alongside "message" -- a structured task failure, not a generic
    transport error: the worker received the task, ran it, and is
    reporting exactly what kind of failure it was, as opposed to an ERROR
    message like WORKER_UNREACHABLE, which means the task was never even
    attempted.
    """
    handler = HANDLERS.get(task_type)

    try:
        if handler is not None:
            result = handler(payload)
        else:
            result = execute_registered(task_type, payload)
    except ExecutionError as exc:
        return {"status": "error", "code": exc.code, "message": str(exc)}

    return {"status": "success", "result": result}
