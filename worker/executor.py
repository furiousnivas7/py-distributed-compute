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


class ExecutionError(Exception):
    """Raised when a task's payload cannot be executed."""


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
        raise ExecutionError(f"Unsupported MAP operation: {operation}")

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
        raise ExecutionError(f"Unsupported REDUCE operation: {operation}")

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

    Three distinct failure modes, all reported the same controlled way
    (ExecutionError -> {"status": "error", ...}, never an unhandled
    exception that could crash the worker process):
      - payload isn't an object (can't be spread as keyword arguments)
      - the function's OWN exception (wrapped with its type name so the
        caller can tell "my function raised ValueError" from "the
        contract itself was violated") -- this also naturally covers
        wrong/missing arguments, which surface as Python's own TypeError
      - a return value that can't survive the JSON wire protocol -- caught
        HERE, on the worker, rather than failing confusingly deep inside
        send_message on the way out.
    """
    fn = registry.get_function(operation)
    if fn is None:
        raise ExecutionError(f"Unsupported task type: {operation}")

    if not isinstance(payload, dict):
        raise ExecutionError("payload must be an object of keyword arguments")

    try:
        result = fn(**payload)
    except Exception as exc:
        raise ExecutionError(f"{type(exc).__name__}: {exc}") from exc

    try:
        json.dumps(result)
    except (TypeError, ValueError) as exc:
        raise ExecutionError(f"Return value of {operation!r} is not JSON-serializable: {exc}") from exc

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

    raise ExecutionError(f"Unsupported execution_mode: {mode!r}")


def _execute_serialized_callable(payload: dict):
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
        raise ExecutionError(str(exc)) from exc

    try:
        result = fn(*args, **kwargs)
    except Exception as exc:
        raise ExecutionError(f"{type(exc).__name__}: {exc}") from exc

    try:
        json.dumps(result)
    except (TypeError, ValueError) as exc:
        raise ExecutionError(f"Return value is not JSON-serializable: {exc}") from exc

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
    """
    handler = HANDLERS.get(task_type)

    try:
        if handler is not None:
            result = handler(payload)
        else:
            result = execute_registered(task_type, payload)
    except ExecutionError as exc:
        return {"status": "error", "message": str(exc)}

    return {"status": "success", "result": result}
