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


def test_map_square():
    result = execute_task("MAP", {"operation": "SQUARE", "data": [1, 2, 3, 4]})
    assert result == {"status": "success", "result": [1, 4, 9, 16]}


def test_map_double():
    result = execute_task("MAP", {"operation": "DOUBLE", "data": [1, 2, 3]})
    assert result == {"status": "success", "result": [2, 4, 6]}


def test_map_increment():
    result = execute_task("MAP", {"operation": "INCREMENT", "data": [0, 1, -1]})
    assert result == {"status": "success", "result": [1, 2, 0]}


def test_map_negate():
    result = execute_task("MAP", {"operation": "NEGATE", "data": [1, -2, 3]})
    assert result == {"status": "success", "result": [-1, 2, -3]}


def test_map_empty_data():
    result = execute_task("MAP", {"operation": "SQUARE", "data": []})
    assert result == {"status": "success", "result": []}


def test_map_unknown_operation():
    result = execute_task("MAP", {"operation": "CUBE", "data": [1, 2, 3]})
    assert result["status"] == "error"
    assert "CUBE" in result["message"]


def test_map_data_not_a_list():
    result = execute_task("MAP", {"operation": "SQUARE", "data": 5})
    assert result["status"] == "error"


def test_map_non_numeric_element_rejected():
    result = execute_task("MAP", {"operation": "SQUARE", "data": [1, "x", 3]})
    assert result["status"] == "error"


def test_map_boolean_element_rejected():
    result = execute_task("MAP", {"operation": "SQUARE", "data": [1, True, 3]})
    assert result["status"] == "error"
