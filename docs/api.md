# Public API Reference

The stable, documented import surface (Phase 12.1) — `master`, `worker`,
`jobs`, `common`. Each package's `__init__.py` re-exports exactly the
names below via `__all__`; deep imports (`from master.scheduler import
Scheduler`, etc.) work identically and are used throughout this
project's own test suite. This page is a map, not a duplicate of every
docstring — follow the file:line pointers for the full contract on any
given function.

For a narrative walkthrough with runnable code, see
[`README.md`'s "Getting Started"](../README.md#getting-started-python-api)
and [`examples/`](../examples/). For how these pieces fit together, see
[`architecture.md`](architecture.md).

## `master`

Process-lifecycle and task-scheduling. `master.scheduler` is a
**process-wide singleton** `Scheduler` instance — every task submitted
through it is what `start_master`'s dispatcher and `wait_for_tasks`
actually operate on. `master/async_server.py`'s docstring is the
authoritative source on the dispatch contract; `wait_for_tasks` is the
one production-supported way to submit work and get results back.

| Name | Kind | Signature | Notes |
|---|---|---|---|
| `scheduler` | singleton | — | the shared `Scheduler` instance every `jobs.*` helper submits through by default |
| `start_master` | async fn | `(host=HOST, port=PORT) -> asyncio.Server` | idempotent — returns the existing server if already running |
| `stop_master` | async fn | `() -> None` | idempotent; closes connections, stops the dispatcher, tears down cleanly |
| `wait_for_tasks` | async fn | `(task_ids: set[str], timeout=30.0) -> list[dict]` | starts the dispatcher if needed, awaits each task's terminal `TASK_RESULT`/`ERROR` |
| `wait_for_workers` | async fn | `(count: int, poll_interval=0.01) -> None` | blocks until at least `count` workers are registered |
| `Scheduler` | class | `Scheduler(worker_manager: WorkerManager)` | `submit_task`/`assign_task`/`start_task`/`complete_task`/`fail_task`/`requeue_tasks_for_worker` — see `master/scheduler.py` |
| `Scheduler.submit_task` | method | `(task_id: str, task_type: str, payload: dict) -> Task` | raises `ValueError` on a duplicate `task_id` or invalid arguments |
| `TaskNotFoundError` | exception | `KeyError` subclass | an unknown `task_id` |
| `NoAvailableWorkerError` | exception | `RuntimeError` subclass | no `IDLE` worker when `assign_task` needs one |
| `WorkerManager` | class | `WorkerManager()` | `register_worker`/`get_worker`/`get_all_workers`/`update_status`/`record_heartbeat`/`get_stale_workers` — see `master/worker_manager.py` |
| `DuplicateWorkerError` | exception | `ValueError` subclass | re-registering an already-`IDLE`/`BUSY` worker_id |
| `WorkerNotFoundError` | exception | `KeyError` subclass | an unknown `worker_id` |

**Sync vs async**: `Scheduler`/`WorkerManager` methods are all plain sync
(no `await`) — every state mutation runs to completion atomically with
respect to other coroutines on the same event loop, deliberately, so no
locking is needed. `start_master`/`stop_master`/`wait_for_tasks`/
`wait_for_workers` are `async def` and must be awaited from a running
event loop (`asyncio.run(...)`).

## `worker`

Execution: how a worker actually runs a task, and how it's configured.

| Name | Kind | Signature | Notes |
|---|---|---|---|
| `run_worker` | async fn | `(master_host, master_port, worker_id=..., worker_host=..., worker_port=..., heartbeat_interval=..., shutdown_event=None, backend=None) -> None` | runs until the connection ends; `backend=None` defaults to `DirectBackend()` |
| `ExecutionBackend` | ABC | — | `execute(task_type, payload) -> dict` (abstract, async), `start()`/`stop()` (async, default no-op), `get_metrics()`, `describe()` |
| `DirectBackend` | class | `DirectBackend()` | runs in-process; the default |
| `MultiprocessingBackend` | class | `MultiprocessingBackend(max_workers=None, mp_context="fork", max_in_flight=None)` | raises `ValueError` for a non-positive `max_workers`/`max_in_flight` or an unsupported `mp_context` |
| `BackendMetrics` | dataclass | `submitted`/`running`/`completed`/`failed`/`backend_errors`/`total_execution_time`/`total_wait_time` | a snapshot from `backend.get_metrics()`, never a live reference |
| `BackendConfig` | dataclass | `backend`/`max_workers`/`max_in_flight`/`mp_context` | resolved config, not yet a backend instance |
| `resolve_backend_config` | fn | `(argv=None, env=None) -> BackendConfig` | CLI args > env vars > defaults; raises `ValueError` with the actual received value on anything invalid |
| `build_backend` | fn | `(config: BackendConfig) -> ExecutionBackend` | constructs the backend a config describes |
| `ExecutionError` | exception | `Exception` subclass, `.code` | raised internally by `worker.executor`, always caught before crossing `execute_task`'s boundary |
| `ExecutionErrorCode` | class of constants | `UNKNOWN_OPERATION`, `INVALID_ARGUMENTS`, `DESERIALIZATION_FAILED`, `USER_FUNCTION_ERROR`, `RESULT_SERIALIZATION_FAILED`, `UNSUPPORTED_RETURN_VALUE` | the `code` field of every structured error result |
| `MAP_OPERATIONS` / `REDUCE_OPERATIONS` | dict | `{name: callable}` | built-in Map/Reduce operations (`WORD_COUNT`, `SUM`, `COUNT`, `MAX`, `MIN`, ...) |
| `register_function` | fn | `(name: str, fn: Callable, *, replace=False) -> None` | raises `FunctionAlreadyRegisteredError` unless `replace=True` |
| `register` | decorator factory | `(name: str)` | `@register("my_op")` form of `register_function` |
| `get_function` | fn | `(name: str) -> Callable \| None` | |
| `is_registered` | fn | `(name: str) -> bool` | |
| `clear_registry` | fn | `() -> None` | resets the registry (mostly for tests) |
| `FunctionAlreadyRegisteredError` | exception | `ValueError` subclass | |

**Sync vs async**: `run_worker` and `ExecutionBackend.execute`/`start`/`stop`
are `async def`. Everything else in this package is plain sync.

**Two failure categories through `execute()`** (see `architecture.md`):
a normal execution failure never raises — it's a `{"status": "error",
...}` result; a backend failure (a dead child process) *does* raise, and
is allowed to propagate, tearing down the worker's connection (the
master's existing crash-recovery path takes it from there).

## `jobs`

Submitting work: single calls, Map/Reduce, and full MapReduce.

| Name | Kind | Signature | Notes |
|---|---|---|---|
| `submit_call` | fn | `(scheduler, task_id, operation: str, **kwargs) -> Task` | `task_type` IS `operation`; `kwargs` become the payload directly |
| `submit_registered_call` | fn | `(scheduler, task_id, operation: str, **kwargs) -> Task` | same result, via the explicit `EXECUTE`/`"registered"` envelope |
| `submit_serialized_call` | fn | `(scheduler, task_id, fn, args=None, kwargs=None) -> Task` | ships `fn` itself (cloudpickle) — see the README's Security Considerations |
| `collect_call_result` | fn | `(response: dict)` | returns the result, or raises `CallError` |
| `CallError` | exception | `ValueError` subclass, `.code`/`.message` | |
| `build_map_job` | fn | `(scheduler, job_id, operation: str \| ExecutionSpec, data: list, num_partitions: int) -> list[Task]` | one `MAP` task per partition, in partition order |
| `collect_map_results` | fn | `(tasks, responses) -> list` | |
| `build_intermediate_results` | fn | `(job_id, tasks, responses) -> list[IntermediateResult]` | tolerates failed/missing partitions |
| `build_reduce_job` | fn | `(scheduler, job_id, grouped: dict, operation: str \| ExecutionSpec) -> dict[str, Task]` | one `REDUCE` task per key |
| `collect_reduce_results` | fn | `(tasks, responses) -> dict` | raises `ValueError` on any missing/failed key — never silently drops one |
| `reduce_grouped` | fn | `(grouped: dict, operation: str) -> dict` | the same computation, purely locally, no dispatch |
| `shuffle` | fn | `(results: list[IntermediateResult]) -> dict` | groups Map output by key |
| `run_map_reduce` | async fn | `(scheduler, dispatch, job_id, data, map_operation, reduce_operation, num_partitions) -> dict` | full Map → Shuffle → Reduce; `dispatch` is injected (e.g. `master.async_server.drain_tasks_for`) |
| `ExecutionSpec` | frozen dataclass | `.registered(name)` / `.serialized(fn)` / `.coerce(value)` | unifies registered-function and serialized-callable dispatch for `build_map_job`/`build_reduce_job`/`run_map_reduce` |
| `IntermediateResult` | dataclass | `job_id`/`partition_id`/`task_id`/`status`/`data`/`message`/`worker_id`/`attempt` | one partition's Map output |
| `IntermediateResultStore` | class | `store`/`get`/`has_job`/`clear` | in-memory, keyed by `job_id` |
| `ResultStatus` | class of constants | `SUCCESS`, `ERROR` | |

**Sync vs async**: only `run_map_reduce` is `async def` (it awaits
`dispatch`). Every submission/collection helper is plain sync — they
just call `scheduler.submit_task` and read `Task`/response objects.

## `common`

Shared data models every other package's public surface returns or
accepts.

| Name | Kind | Notes |
|---|---|---|
| `Task` | dataclass | `task_id`/`task_type`/`payload`/`status`/`assigned_worker_id`/`attempt` |
| `TaskStatus` | class of constants | `PENDING`, `ASSIGNED`, `RUNNING`, `COMPLETED`, `FAILED` |
| `Worker` | dataclass | `worker_id`/`host`/`port`/`status`/`last_heartbeat`/`generation` |
| `WorkerStatus` | class of constants | `REGISTERED`, `IDLE`, `BUSY`, `FAILED`, `DRAINING`, `STOPPED` |

## `pydc` CLI

Registered via `pyproject.toml`'s `[project.scripts]`, implemented in
`client.py`. See `pydc --help` / `pydc <command> --help` for the full,
current flag list (this table is a summary, the `--help` output is
authoritative).

| Command | What it does |
|---|---|
| `pydc master [--host] [--port]` | starts a master, serves until Ctrl+C/SIGTERM |
| `pydc worker [--backend ...] [--master-host] [--worker-id] ...` | starts a worker, connecting to a master |
| `pydc run call OPERATION [--arg k=v] [--import mod:fn:name]` | in-process master + one local worker, one call, prints JSON, exits |
| `pydc run mapreduce --data FILE --map OP --reduce OP [--partitions N]` | same, for a full MapReduce job |

`pydc run` is entirely local/in-process — see `architecture.md`'s note
on why there's no `pydc submit` for an already-running remote master.

## Everything *not* re-exported here

`worker.serialization` (raw cloudpickle helpers — go through
`jobs.submit_serialized_call`/`ExecutionSpec.serialized` instead),
`master.server`/`worker.worker` (the earlier synchronous/threaded
implementations, superseded but kept for their own tests), and `rpc/`
(wire-protocol/transport internals) are all still importable via their
full module paths, but are intentionally outside the curated public
surface.
