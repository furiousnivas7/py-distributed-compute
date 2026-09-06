"""Executes tasks on the worker and produces a result payload."""

ADD = "ADD"
MULTIPLY = "MULTIPLY"
MAP = "MAP"

# Named operations only -- no arbitrary Python function serialization.
# Keeps the wire protocol a fixed, deterministic vocabulary rather than
# shipping code between processes.
MAP_OPERATIONS = {
    "SQUARE": lambda x: x * x,
    "DOUBLE": lambda x: x * 2,
    "INCREMENT": lambda x: x + 1,
    "NEGATE": lambda x: -x,
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
    mapped = []
    for value in data:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ExecutionError("MAP data must contain only numeric values")
        mapped.append(fn(value))
    return mapped


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
