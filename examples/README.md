# Examples

Each script below is standalone and runnable with just:

```bash
python examples/<name>.py
```

No separate master/worker process needed — every example starts its own
in-process master + worker(s), exactly like `pydc run` does (see
[`../docs/api.md`](../docs/api.md#pydc-cli)). All of them use only the
public API (`master`, `worker`, `jobs`, `common` — see
[`../docs/api.md`](../docs/api.md)), and every one is run automatically
by `tests/test_examples.py` as part of the normal test suite, so they
can't silently rot.

| File | Demonstrates |
|---|---|
| [`01_registered_function.py`](01_registered_function.py) | the basic case: register a function, run a worker, submit a call |
| [`02_serialized_callable.py`](02_serialized_callable.py) | an arbitrary function shipped as data (cloudpickle), never registered |
| [`03_multiple_workers.py`](03_multiple_workers.py) | several workers sharing one job's tasks |
| [`04_map_reduce.py`](04_map_reduce.py) | a full Map → Shuffle → Reduce job with built-in operations |
| [`05_multiprocessing_backend.py`](05_multiprocessing_backend.py) | CPU-bound work in a real child process, without blocking the worker's event loop |
| [`06_fault_tolerance.py`](06_fault_tolerance.py) | a worker crashing mid-job and the master retrying the task on another worker |
| [`07_cli_usage.sh`](07_cli_usage.sh) | the same ideas via the `pydc` CLI instead of Python code (illustrative — not auto-run; see the script's own header) |

See [`../docs/architecture.md`](../docs/architecture.md) for how the
pieces these examples use actually fit together, and
[`../docs/development.md`](../docs/development.md) for running the test
suite these examples are validated by.
