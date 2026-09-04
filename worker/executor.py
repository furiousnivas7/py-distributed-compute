"""Executes tasks on the worker and produces a result payload."""

ADD = "ADD"
MULTIPLY = "MULTIPLY"


class ExecutionError(Exception):
    """Raised when a task's payload cannot be executed."""


def execute_add(payload: dict):
    a, b = _require_numbers(payload)
    return a + b


def execute_multiply(payload: dict):
    a, b = _require_numbers(payload)
    return a * b


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
