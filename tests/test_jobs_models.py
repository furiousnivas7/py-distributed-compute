from jobs.models import IntermediateResult, IntermediateResultStore, ResultStatus


def test_intermediate_result_defaults():
    result = IntermediateResult(
        job_id="job-1",
        partition_id=0,
        task_id="job-1-map-0",
        status=ResultStatus.SUCCESS,
        data=[1, 4, 9],
    )
    assert result.job_id == "job-1"
    assert result.partition_id == 0
    assert result.task_id == "job-1-map-0"
    assert result.status == ResultStatus.SUCCESS
    assert result.data == [1, 4, 9]
    assert result.message is None
    assert result.worker_id is None
    assert result.attempt == 0


def test_intermediate_result_error_shape():
    result = IntermediateResult(
        job_id="job-1",
        partition_id=1,
        task_id="job-1-map-1",
        status=ResultStatus.ERROR,
        message="boom",
        worker_id="worker-1",
        attempt=2,
    )
    assert result.status == ResultStatus.ERROR
    assert result.data is None
    assert result.message == "boom"
    assert result.worker_id == "worker-1"
    assert result.attempt == 2


def test_store_and_get_round_trip():
    store = IntermediateResultStore()
    results = [
        IntermediateResult("job-1", 0, "job-1-map-0", ResultStatus.SUCCESS, data=[1, 4]),
        IntermediateResult("job-1", 1, "job-1-map-1", ResultStatus.SUCCESS, data=[9, 16]),
    ]
    store.store("job-1", results)

    assert store.has_job("job-1") is True
    fetched = store.get("job-1")
    assert fetched == results
    # get() returns a copy -- mutating it must not affect the stored list.
    fetched.append(IntermediateResult("job-1", 2, "job-1-map-2", ResultStatus.SUCCESS, data=[]))
    assert len(store.get("job-1")) == 2


def test_get_unknown_job_returns_empty_list():
    store = IntermediateResultStore()
    assert store.get("missing-job") == []
    assert store.has_job("missing-job") is False


def test_multiple_jobs_do_not_mix_results():
    store = IntermediateResultStore()
    job1_results = [IntermediateResult("job-1", 0, "job-1-map-0", ResultStatus.SUCCESS, data=[1, 2])]
    job2_results = [IntermediateResult("job-2", 0, "job-2-map-0", ResultStatus.SUCCESS, data=[100, 200])]

    store.store("job-1", job1_results)
    store.store("job-2", job2_results)

    assert store.get("job-1") == job1_results
    assert store.get("job-2") == job2_results
    assert store.get("job-1") != store.get("job-2")


def test_clear_removes_all_jobs():
    store = IntermediateResultStore()
    store.store("job-1", [IntermediateResult("job-1", 0, "job-1-map-0", ResultStatus.SUCCESS, data=[1])])
    store.clear()
    assert store.get("job-1") == []
    assert store.has_job("job-1") is False


def test_storing_again_overwrites_previous_results_for_same_job():
    store = IntermediateResultStore()
    store.store("job-1", [IntermediateResult("job-1", 0, "job-1-map-0", ResultStatus.SUCCESS, data=[1])])
    store.store("job-1", [IntermediateResult("job-1", 0, "job-1-map-0", ResultStatus.SUCCESS, data=[2])])

    results = store.get("job-1")
    assert len(results) == 1
    assert results[0].data == [2]
