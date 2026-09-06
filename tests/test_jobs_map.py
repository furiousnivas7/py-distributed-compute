import pytest

from common.models import TaskStatus
from jobs.map import build_intermediate_results, build_map_job, collect_map_results, partition
from jobs.models import ResultStatus
from master.scheduler import Scheduler
from master.worker_manager import WorkerManager


def test_partition_even_split():
    assert partition([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]


def test_partition_uneven_split_front_loads_remainder():
    assert partition([1, 2, 3, 4, 5], 2) == [[1, 2, 3], [4, 5]]


def test_partition_single_partition():
    assert partition([1, 2, 3], 1) == [[1, 2, 3]]


def test_partition_more_partitions_than_items():
    # Never more chunks than items -- no empty chunks.
    assert partition([1, 2], 5) == [[1], [2]]


def test_partition_empty_data():
    assert partition([], 4) == []


def test_partition_preserves_order():
    chunks = partition(list(range(10)), 3)
    assert [x for chunk in chunks for x in chunk] == list(range(10))


def test_partition_invalid_num_partitions_raises():
    with pytest.raises(ValueError):
        partition([1, 2, 3], 0)


def test_build_map_job_submits_one_task_per_partition():
    scheduler = Scheduler(WorkerManager())
    tasks = build_map_job(scheduler, "job-1", "SQUARE", [1, 2, 3, 4], num_partitions=2)

    assert len(tasks) == 2
    assert [t.task_id for t in tasks] == ["job-1-map-0", "job-1-map-1"]
    assert all(t.task_type == "MAP" for t in tasks)
    assert all(t.status == TaskStatus.PENDING for t in tasks)
    assert tasks[0].payload == {"operation": "SQUARE", "data": [1, 2]}
    assert tasks[1].payload == {"operation": "SQUARE", "data": [3, 4]}


def test_build_map_job_empty_data_submits_nothing():
    scheduler = Scheduler(WorkerManager())
    tasks = build_map_job(scheduler, "job-1", "SQUARE", [], num_partitions=4)
    assert tasks == []


def _response(task_id: str, status: str, result=None, message=None) -> dict:
    payload = {"task_id": task_id, "status": status}
    if result is not None:
        payload["result"] = result
    if message is not None:
        payload["message"] = message
    return {"type": "TASK_RESULT", "payload": payload}


def test_collect_map_results_in_partition_order_regardless_of_response_order():
    scheduler = Scheduler(WorkerManager())
    tasks = build_map_job(scheduler, "job-1", "SQUARE", [1, 2, 3, 4], num_partitions=2)

    # Responses arrive out of partition order (partition 1 finished first).
    responses = [
        _response("job-1-map-1", "success", result=[9, 16]),
        _response("job-1-map-0", "success", result=[1, 4]),
    ]

    combined = collect_map_results(tasks, responses)
    assert combined == [1, 4, 9, 16]


def test_collect_map_results_empty_job():
    assert collect_map_results([], []) == []


def test_collect_map_results_raises_on_failed_task():
    scheduler = Scheduler(WorkerManager())
    tasks = build_map_job(scheduler, "job-1", "SQUARE", [1, 2], num_partitions=1)

    responses = [_response("job-1-map-0", "error", message="boom")]

    with pytest.raises(ValueError):
        collect_map_results(tasks, responses)


def test_collect_map_results_raises_on_missing_response():
    scheduler = Scheduler(WorkerManager())
    tasks = build_map_job(scheduler, "job-1", "SQUARE", [1, 2, 3, 4], num_partitions=2)

    responses = [_response("job-1-map-0", "success", result=[1, 4])]

    with pytest.raises(ValueError):
        collect_map_results(tasks, responses)


def test_build_intermediate_results_success():
    scheduler = Scheduler(WorkerManager())
    tasks = build_map_job(scheduler, "job-1", "SQUARE", [1, 2, 3, 4], num_partitions=2)
    tasks[0].assigned_worker_id = "worker-1"
    tasks[0].attempt = 1
    tasks[1].assigned_worker_id = "worker-2"
    tasks[1].attempt = 1

    responses = [
        _response("job-1-map-0", "success", result=[1, 4]),
        _response("job-1-map-1", "success", result=[9, 16]),
    ]

    results = build_intermediate_results("job-1", tasks, responses)

    assert len(results) == 2
    assert [r.partition_id for r in results] == [0, 1]
    assert results[0].job_id == "job-1"
    assert results[0].task_id == "job-1-map-0"
    assert results[0].status == ResultStatus.SUCCESS
    assert results[0].data == [1, 4]
    assert results[0].worker_id == "worker-1"
    assert results[1].data == [9, 16]
    assert results[1].worker_id == "worker-2"


def test_build_intermediate_results_preserves_partition_order_regardless_of_response_order():
    scheduler = Scheduler(WorkerManager())
    tasks = build_map_job(scheduler, "job-1", "SQUARE", [1, 2, 3, 4], num_partitions=2)

    # Responses arrive out of partition order.
    responses = [
        _response("job-1-map-1", "success", result=[9, 16]),
        _response("job-1-map-0", "success", result=[1, 4]),
    ]

    results = build_intermediate_results("job-1", tasks, responses)

    assert [r.partition_id for r in results] == [0, 1]
    assert results[0].data == [1, 4]
    assert results[1].data == [9, 16]


def test_build_intermediate_results_records_failure_without_raising():
    scheduler = Scheduler(WorkerManager())
    tasks = build_map_job(scheduler, "job-1", "SQUARE", [1, 2], num_partitions=1)
    tasks[0].assigned_worker_id = "worker-1"

    responses = [_response("job-1-map-0", "error", message="bad data")]

    results = build_intermediate_results("job-1", tasks, responses)

    assert len(results) == 1
    assert results[0].status == ResultStatus.ERROR
    assert results[0].message == "bad data"
    assert results[0].data is None
    assert results[0].worker_id == "worker-1"


def test_build_intermediate_results_records_missing_response_without_raising():
    scheduler = Scheduler(WorkerManager())
    tasks = build_map_job(scheduler, "job-1", "SQUARE", [1, 2, 3, 4], num_partitions=2)

    responses = [_response("job-1-map-0", "success", result=[1, 4])]

    results = build_intermediate_results("job-1", tasks, responses)

    assert len(results) == 2
    assert results[0].status == ResultStatus.SUCCESS
    assert results[1].status == ResultStatus.ERROR
    assert results[1].message == "No response received"


def test_build_intermediate_results_empty_job():
    assert build_intermediate_results("job-1", [], []) == []
