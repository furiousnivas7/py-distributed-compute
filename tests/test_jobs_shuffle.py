from jobs.models import IntermediateResult, ResultStatus
from jobs.shuffle import shuffle


def _result(job_id, partition_id, data=None, status=ResultStatus.SUCCESS, message=None):
    return IntermediateResult(
        job_id=job_id,
        partition_id=partition_id,
        task_id=f"{job_id}-map-{partition_id}",
        status=status,
        data=data,
        message=message,
    )


def test_basic_key_value_grouping():
    results = [_result("job-1", 0, data=[("apple", 1), ("banana", 1)])]
    assert shuffle(results) == {"apple": [1], "banana": [1]}


def test_multiple_partitions_are_all_grouped():
    results = [
        _result("job-1", 0, data=[("apple", 1), ("banana", 1)]),
        _result("job-1", 1, data=[("apple", 1), ("orange", 1)]),
        _result("job-1", 2, data=[("banana", 1), ("apple", 1)]),
    ]
    assert shuffle(results) == {
        "apple": [1, 1, 1],
        "banana": [1, 1],
        "orange": [1],
    }


def test_duplicate_keys_within_one_partition():
    results = [_result("job-1", 0, data=[("apple", 1), ("apple", 1), ("apple", 1)])]
    assert shuffle(results) == {"apple": [1, 1, 1]}


def test_keys_appearing_in_different_partitions_are_merged():
    results = [
        _result("job-1", 0, data=[("apple", 1)]),
        _result("job-1", 1, data=[("apple", 1)]),
        _result("job-1", 2, data=[("apple", 1)]),
    ]
    assert shuffle(results) == {"apple": [1, 1, 1]}


def test_empty_intermediate_results_list():
    assert shuffle([]) == {}


def test_empty_partition_data_contributes_nothing():
    results = [_result("job-1", 0, data=[])]
    assert shuffle(results) == {}


def test_multiple_value_types():
    results = [
        _result("job-1", 0, data=[("count", 1), ("label", "x"), ("ratio", 1.5), ("flag", True)]),
    ]
    grouped = shuffle(results)
    assert grouped == {"count": [1], "label": ["x"], "ratio": [1.5], "flag": [True]}


def test_value_ordering_is_preserved_partition_then_within_partition():
    results = [
        _result("job-1", 0, data=[("k", "a"), ("k", "b")]),
        _result("job-1", 1, data=[("k", "c"), ("k", "d")]),
    ]
    assert shuffle(results) == {"k": ["a", "b", "c", "d"]}


def test_failed_intermediate_result_is_skipped():
    results = [
        _result("job-1", 0, data=[("apple", 1)]),
        _result("job-1", 1, status=ResultStatus.ERROR, message="boom", data=None),
    ]
    assert shuffle(results) == {"apple": [1]}


def test_missing_intermediate_result_is_skipped():
    # jobs.map.build_intermediate_results represents a missing response as
    # an ERROR result with data=None -- shuffle must not choke on that.
    results = [
        _result("job-1", 0, data=[("apple", 1)]),
        _result("job-1", 1, status=ResultStatus.ERROR, message="No response received", data=None),
    ]
    assert shuffle(results) == {"apple": [1]}


def test_all_failed_results_produce_empty_grouping():
    results = [
        _result("job-1", 0, status=ResultStatus.ERROR, message="boom", data=None),
        _result("job-1", 1, status=ResultStatus.ERROR, message="boom", data=None),
    ]
    assert shuffle(results) == {}


def test_multiple_jobs_shuffled_independently_do_not_mix():
    job1_results = [_result("job-1", 0, data=[("apple", 1), ("banana", 1)])]
    job2_results = [_result("job-2", 0, data=[("apple", 100)])]

    grouped1 = shuffle(job1_results)
    grouped2 = shuffle(job2_results)

    assert grouped1 == {"apple": [1], "banana": [1]}
    assert grouped2 == {"apple": [100]}


def test_deterministic_output_across_repeated_calls():
    results = [
        _result("job-1", 0, data=[("banana", 1), ("apple", 1)]),
        _result("job-1", 1, data=[("apple", 1), ("cherry", 1)]),
    ]

    first = shuffle(results)
    second = shuffle(results)

    assert first == second
    assert list(first.keys()) == list(second.keys())
    assert list(first.values()) == list(second.values())
