import pytest

from jobs.reduce import build_reduce_job, collect_reduce_results, reduce_grouped
from master.scheduler import Scheduler
from master.worker_manager import WorkerManager


def test_reduce_grouped_sum():
    grouped = {"apple": [1, 1, 1, 1], "banana": [1, 1, 1], "orange": [1, 1]}
    assert reduce_grouped(grouped, "SUM") == {"apple": 4, "banana": 3, "orange": 2}


def test_reduce_grouped_count():
    grouped = {"apple": [1, 1, 1, 1], "banana": [1, 1, 1]}
    assert reduce_grouped(grouped, "COUNT") == {"apple": 4, "banana": 3}


def test_reduce_grouped_max():
    grouped = {"apple": [3, 7, 2], "banana": [10]}
    assert reduce_grouped(grouped, "MAX") == {"apple": 7, "banana": 10}


def test_reduce_grouped_min():
    grouped = {"apple": [3, 7, 2], "banana": [10]}
    assert reduce_grouped(grouped, "MIN") == {"apple": 2, "banana": 10}


def test_reduce_grouped_empty_input():
    assert reduce_grouped({}, "SUM") == {}


def test_reduce_grouped_unknown_operation_raises():
    with pytest.raises(ValueError):
        reduce_grouped({"apple": [1, 2]}, "AVERAGE")


def test_reduce_grouped_empty_values_for_key_raises():
    with pytest.raises(ValueError):
        reduce_grouped({"apple": []}, "SUM")


def test_reduce_grouped_non_numeric_value_raises():
    with pytest.raises(ValueError):
        reduce_grouped({"apple": [1, "x"]}, "SUM")


def _response(task_id: str, status: str, result=None, message=None) -> dict:
    payload = {"task_id": task_id, "status": status}
    if result is not None:
        payload["result"] = result
    if message is not None:
        payload["message"] = message
    return {"type": "TASK_RESULT", "payload": payload}


def test_build_reduce_job_submits_one_task_per_key():
    scheduler = Scheduler(WorkerManager())
    grouped = {"apple": [1, 1, 1, 1], "banana": [1, 1, 1]}

    tasks = build_reduce_job(scheduler, "job-1", grouped, "SUM")

    assert set(tasks.keys()) == {"apple", "banana"}
    assert tasks["apple"].task_id == "job-1-reduce-apple"
    assert tasks["apple"].task_type == "REDUCE"
    assert tasks["apple"].payload == {"operation": "SUM", "key": "apple", "values": [1, 1, 1, 1]}


def test_build_reduce_job_empty_grouped_submits_nothing():
    scheduler = Scheduler(WorkerManager())
    assert build_reduce_job(scheduler, "job-1", {}, "SUM") == {}


def test_collect_reduce_results_success():
    scheduler = Scheduler(WorkerManager())
    grouped = {"apple": [1, 1, 1, 1], "banana": [1, 1, 1]}
    tasks = build_reduce_job(scheduler, "job-1", grouped, "SUM")

    responses = [
        _response("job-1-reduce-apple", "success", result=4),
        _response("job-1-reduce-banana", "success", result=3),
    ]

    assert collect_reduce_results(tasks, responses) == {"apple": 4, "banana": 3}


def test_collect_reduce_results_raises_on_failed_key():
    """The important asymmetry with shuffle(): a failed key must raise,
    never be silently dropped from the final reduced result."""
    scheduler = Scheduler(WorkerManager())
    tasks = build_reduce_job(scheduler, "job-1", {"apple": [1, 2]}, "SUM")

    responses = [_response("job-1-reduce-apple", "error", message="boom")]

    with pytest.raises(ValueError):
        collect_reduce_results(tasks, responses)


def test_collect_reduce_results_raises_on_missing_key():
    scheduler = Scheduler(WorkerManager())
    tasks = build_reduce_job(scheduler, "job-1", {"apple": [1], "banana": [2]}, "SUM")

    responses = [_response("job-1-reduce-apple", "success", result=1)]

    with pytest.raises(ValueError):
        collect_reduce_results(tasks, responses)


def test_collect_reduce_results_one_bad_key_does_not_silently_drop_it_from_a_partial_result():
    """Even if every OTHER key succeeded, one failure must still raise
    rather than return a dict that's silently missing that key."""
    scheduler = Scheduler(WorkerManager())
    tasks = build_reduce_job(scheduler, "job-1", {"apple": [1], "banana": [2], "cherry": [3]}, "SUM")

    responses = [
        _response("job-1-reduce-apple", "success", result=1),
        _response("job-1-reduce-banana", "error", message="boom"),
        _response("job-1-reduce-cherry", "success", result=3),
    ]

    with pytest.raises(ValueError):
        collect_reduce_results(tasks, responses)
