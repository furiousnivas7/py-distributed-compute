from worker.executor import execute_task


def test_add():
    result = execute_task("ADD", {"a": 10, "b": 20})
    assert result == {"status": "success", "result": 30}


def test_multiply():
    result = execute_task("MULTIPLY", {"a": 4, "b": 5})
    assert result == {"status": "success", "result": 20}


def test_add_floats():
    result = execute_task("ADD", {"a": 1.5, "b": 2.5})
    assert result == {"status": "success", "result": 4.0}


def test_unknown_task_type():
    result = execute_task("SUBTRACT", {"a": 1, "b": 2})
    assert result["status"] == "error"
    assert "SUBTRACT" in result["message"]


def test_missing_operand():
    result = execute_task("ADD", {"a": 1})
    assert result["status"] == "error"


def test_non_numeric_operand():
    result = execute_task("MULTIPLY", {"a": "x", "b": 2})
    assert result["status"] == "error"


def test_boolean_operand_rejected():
    result = execute_task("ADD", {"a": True, "b": 2})
    assert result["status"] == "error"
