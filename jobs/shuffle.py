"""Shuffle: group Map intermediate results by key, ready for Reduce.

Pure data transformation -- no networking, no scheduler involvement, no
partitioned network shuffle (that belongs to a later, real distributed
Reduce phase). Takes whatever IntermediateResult objects a Map job
produced (jobs/map.py, jobs/models.py) and groups their (key, value) pairs
by key.
"""

from jobs.models import IntermediateResult, ResultStatus


def shuffle(results: list[IntermediateResult]) -> dict:
    """Group all (key, value) pairs across `results` by key.

    Only SUCCESS results contribute -- ERROR results (a failed Map task) or
    ones missing entirely (see jobs.map.build_intermediate_results) are
    skipped rather than raising, since a partially-failed Map job may still
    be worth shuffling whatever succeeded. A caller that wants to fail fast
    instead should check result statuses before calling this.

    Value ordering within each key's list is deterministic as long as
    `results` is given in a stable order (jobs.map.build_intermediate_results
    already returns results in partition order): values land in the order
    their source partition appears in `results`, then in the order they
    appeared within that partition's data.
    """
    grouped: dict = {}

    for result in results:
        if result.status != ResultStatus.SUCCESS or result.data is None:
            continue

        for pair in result.data:
            key, value = pair
            grouped.setdefault(key, []).append(value)

    return grouped
