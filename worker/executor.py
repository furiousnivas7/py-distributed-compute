"""Executes tasks on the worker and produces a result payload."""

ADD = "ADD"
MULTIPLY = "MULTIPLY"
MAP = "MAP"
REDUCE = "REDUCE"

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
}


def execute_task(task_type: str, payload: dict) -> dict:
    handler = HANDLERS.get(task_type)
    if handler is None:
        return {"status": "error", "message": f"Unsupported task type: {task_type}"}

    try:
        result = handler(payload)
    except ExecutionError as exc:
        return {"status": "error", "message": str(exc)}

    return {"status": "success", "result": result}
