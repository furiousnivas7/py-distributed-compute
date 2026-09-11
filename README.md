# Python Distributed Compute Engine

A distributed computing engine built from scratch in Python.

The system allows a large data-processing job to be divided into smaller tasks and executed by multiple worker processes or machines. A central master coordinates the workers, distributes tasks, collects results, and automatically retries failed tasks.

This project is designed to explore the fundamentals of distributed systems, custom network protocols, asynchronous programming, parallel processing, and fault tolerance.

**Documentation:** this README covers installation, the Python API, the
CLI, and configuration. For more depth, see
[`docs/architecture.md`](docs/architecture.md) (how the pieces fit
together — RPC protocol, task lifecycle, scheduling, MapReduce data flow,
execution backends), [`docs/api.md`](docs/api.md) (the full public API
reference), [`docs/development.md`](docs/development.md) (dev setup,
testing, troubleshooting), [`docs/security.md`](docs/security.md) (the
current trust boundaries — read this before deploying anywhere beyond a
trusted network), [`docs/performance_baseline.md`](docs/performance_baseline.md)
(measured throughput/latency numbers), and [`examples/`](examples/)
(runnable scripts, one per feature).

## Features

- Master-worker architecture, one persistent TCP connection per worker
- Custom TCP-based RPC protocol (`rpc/protocol.py`)
- Worker registration, heartbeat monitoring, and generation-based reconnection after a crash
- Fully asynchronous master and worker (`asyncio`)
- Parallel task execution via a `multiprocessing` execution backend (opt-in; `DirectBackend` in-process execution is the default)
- MapReduce programming model (Map → Shuffle → Reduce)
- Remote execution of registered functions and serialized Python callables
- Task scheduling, dispatch, and automatic retry on worker/transport failure (never on a deterministic execution failure — see "Retry Semantics" below)
- Structured logging (`logging`, not `print`) for lifecycle and diagnostic events
- `pydc` command-line interface (`master`/`worker`/`run` subcommands)
- Installable via `pip` (editable install or a built wheel)

## Architecture

```text
              Your code, in the SAME process as the master
              (scheduler.submit_task / jobs.submit_call / ...)
                           |
                           v
                    +--------------+
                    |    Master    |
                    |              |
                    | Scheduler    |
                    | WorkerManager|
                    | Dispatcher   |
                    +------+-------+
                           |
                     Custom TCP RPC
                    (one persistent connection
                       per worker)
                           |
             +-------------+-------------+
             |             |             |
             v             v             v
       +-----------+ +-----------+ +-----------+
       | Worker 1  | | Worker 2  | | Worker 3  |
       | Execution | | Execution | | Execution |
       |  Backend  | |  Backend  | |  Backend  |
       +-----------+ +-----------+ +-----------+
             |             |             |
             +-------------+-------------+
                           |
                           v
                    Final Result
```

There is no separate network-facing "client" protocol — job submission
happens by calling the master's public API directly (see "Getting
Started" below). For the full picture (the wire protocol, task lifecycle
and retry semantics, worker generations, MapReduce data flow, and the
execution-backend abstraction), see
[`docs/architecture.md`](docs/architecture.md).

## How MapReduce Works

This is the MapReduce path specifically — the engine also supports a
single function call with no Map/Reduce involved at all (see "Example
Jobs" below). For a MapReduce job:

```text
Submit Job
    |
    v
Split Input Data
    |
    v
Create Tasks
    |
    v
Assign Tasks to Workers
    |
    v
Execute Map Function
    |
    v
Shuffle Intermediate Results
    |
    v
Execute Reduce Function
    |
    v
Combine Final Results
    |
    v
Return Output
```

If a worker fails during execution, the master detects the failure and reschedules its unfinished tasks to another available worker.

## Project Status

**`v0.2.0` — a hardened engineering prototype, not a `v1.0.0` release.**
See [`docs/release_decision.md`](docs/release_decision.md) for the full
reasoning: the engine, retry/fault-tolerance, and test coverage
(600+ tests, including end-to-end, concurrency/stress, failure-injection,
and resource-leak suites) are solid, but there's no authentication/
encryption/authorization, no remote job submission to an already-running
master, and no state persistence — real gaps for a "1.0," not oversights.
See [`docs/security.md`](docs/security.md) for the trust boundary this
implies.

Developed incrementally, in small reviewed phases, each with its own
tests and validation before moving on.

## Technology Stack

- **Language:** Python 3.10+
- **Networking:** raw TCP sockets, a custom length-prefixed JSON RPC protocol (`rpc/`)
- **Concurrency:** `asyncio` (the master and worker are both fully async)
- **Parallelism:** `multiprocessing`, via an opt-in execution backend (`worker/backend.py`)
- **Serialization:** `cloudpickle`, for arbitrary-callable execution (the one third-party runtime dependency)
- **Testing:** `pytest`
- **Logging:** Python `logging`
- **Packaging:** `setuptools`/`pyproject.toml` (PEP 621)
- **Version control:** Git

## Project Structure

```text
py-distributed-compute/
│
├── master/            async master: scheduler, worker registry, dispatch
│   ├── async_server.py    the real implementation (Phase 8.9+)
│   ├── config.py          host/port CLI+env resolution
│   ├── scheduler.py
│   ├── worker_manager.py
│   ├── rpc_handler.py
│   └── server.py           earlier synchronous implementation (kept for its own tests)
│
├── worker/             async worker: execution, backends, registry
│   ├── async_worker.py    the real implementation (Phase 9+)
│   ├── backend.py         ExecutionBackend / DirectBackend / MultiprocessingBackend
│   ├── config.py          backend + runtime CLI+env resolution
│   ├── executor.py        task_type -> result, the execution contract
│   ├── registry.py        register_function/register
│   ├── serialization.py   cloudpickle isolation boundary
│   └── worker.py           earlier synchronous implementation (kept for its own tests)
│
├── rpc/                wire protocol and transport
│   ├── protocol.py
│   ├── async_connection.py / async_rpc.py
│   └── connection.py       synchronous counterpart
│
├── jobs/               job submission and orchestration
│   ├── call.py             single registered/serialized calls
│   ├── map.py / reduce.py / shuffle.py / map_reduce.py
│   └── models.py           ExecutionSpec, IntermediateResult
│
├── common/             shared data models
│   ├── models.py           Task, Worker, their status enums
│   └── env.py              env-var names shared by master/worker CLIs
│
├── docs/               architecture.md, api.md, development.md,
│                       security.md, performance_baseline.md,
│                       release_decision.md
├── examples/           runnable, tested example scripts
├── scripts/            benchmark.py (performance baseline)
├── tests/              600+ tests
│
├── client.py           pydc CLI entry point
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Installation

Clone the repository (replace the URL below with wherever you cloned
this from):

```bash
git clone <your-fork-or-clone-url>/py-distributed-compute.git
cd py-distributed-compute
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

Make sure `pip` is reasonably current first -- a venv's `ensurepip`-bundled
pip can be too old to support this project's `pyproject.toml`-only
editable installs (PEP 660):

```bash
pip install --upgrade pip
```

Install the project in editable mode, plus development dependencies
(`pytest`) needed to run the test suite:

```bash
pip install -e .[dev]
```

If you'll also be running `python -m build` locally (e.g. to run
`tests/test_packaging.py`), install a new enough `setuptools` in this same
venv too -- the test suite builds with `--no-isolation` for speed, which
means it uses whatever `setuptools` this venv already has instead of
fetching a fresh one into a throwaway build environment every time:

```bash
pip install --upgrade "setuptools>=61.0" wheel build
```

(`pip install -r requirements.txt` does exactly the same thing -- kept as
the familiar entry point.) For a runtime-only install with no dev
dependencies, drop the extra: `pip install -e .` -- or `pip install .`
for a normal (non-editable) install. `cloudpickle` is the only runtime
dependency (see `pyproject.toml`); it's installed automatically either
way.

### Building and installing a wheel

```bash
pip install build
python -m build
pip install dist/py_distributed_compute-0.2.0-py3-none-any.whl
```

The public API (`master`, `worker`, `jobs`, `common` -- see "Getting
Started" below) then works from any directory, with no dependency on the
repository being on `PYTHONPATH` or the current working directory.

## Getting Started (Python API)

This is the minimal path through the public API -- see `master/__init__.py`,
`worker/__init__.py`, and `jobs/__init__.py` for the complete stable
surface (deep imports like `master.scheduler.Scheduler` keep working too;
these packages just add convenient top-level re-exports).

Register a function, start a worker (in-process, `DirectBackend`), and
submit a call to it:

```python
import asyncio
from master import scheduler, start_master, stop_master, wait_for_tasks
from worker import DirectBackend, register_function, run_worker
from jobs import submit_call, collect_call_result

register_function("double", lambda x: x * 2)

async def main():
    server = await start_master("127.0.0.1", 0)
    host, port = server.sockets[0].getsockname()[:2]
    worker_task = asyncio.create_task(
        run_worker(host, port, worker_id="worker-1", backend=DirectBackend())
    )

    task = submit_call(scheduler, "t1", "double", x=21)
    [response] = await wait_for_tasks({task.task_id})
    print(collect_call_result(response))  # 42

    worker_task.cancel()
    await stop_master()

asyncio.run(main())
```

A serialized callable -- an arbitrary function that was never registered
anywhere (see "Security Considerations" below before using this on a
cluster you don't fully trust):

```python
from jobs import submit_serialized_call

task = submit_serialized_call(scheduler, "t2", lambda x: x * x, args=[6])
[response] = await wait_for_tasks({task.task_id})
print(collect_call_result(response))  # 36
```

A full MapReduce job:

```python
from jobs import run_map_reduce
from master.async_server import drain_tasks_for

result = await run_map_reduce(
    scheduler, drain_tasks_for, "job-1",
    data=["a", "b", "a", "c", "b", "c"],
    map_operation="WORD_COUNT",
    reduce_operation="SUM",
    num_partitions=2,
)
# {"a": 2, "b": 2, "c": 2}
```

A `MultiprocessingBackend` worker instead of `DirectBackend` (see
"Execution Backend Configuration" below for CLI/environment-driven
selection):

```python
from worker import MultiprocessingBackend

backend = MultiprocessingBackend(max_workers=4, max_in_flight=8)
worker_task = asyncio.create_task(run_worker(host, port, backend=backend))
```

## Usage

`pydc` (installed with the project -- see Installation above; run
`pydc --help` for the full command list) is the command-line entry point.

Start the master, and keep it running (Ctrl+C to stop):

```bash
pydc master
```

Start a worker in another terminal, pointed at that master:

```bash
pydc worker
```

Start additional workers in more terminals the same way -- each gets a
randomly generated `--worker-id` unless you set one explicitly:

```bash
pydc worker --worker-id worker-2
```

Run one job without a separate master/worker process at all -- starts an
in-process master + one local worker, submits the job, prints the JSON
result, and exits:

```bash
pydc run call ADD --arg a=10 --arg b=32
# 42

pydc run mapreduce --data words.json --map WORD_COUNT --reduce SUM --partitions 4
```

See "Execution Backend Configuration" below for `--backend`/`--max-workers`/
etc. (accepted by both `pydc worker` and `pydc run`), and `pydc worker
--help`/`pydc master --help`/`pydc run --help` for every flag.

> **Why isn't there a `pydc submit` that talks to an already-running
> remote master?** `master.scheduler` is an in-process singleton, and the
> wire protocol has no client-facing job-submission message type -- only
> worker<->master messages. `pydc run` is the closest equivalent that
> needs no protocol change: everything happens in one process. Submitting
> to a separate, already-running master from a different process is a
> real gap, left for a future phase that explicitly adds a client-facing
> RPC endpoint.

## Execution Backend Configuration

A worker started via `worker.async_worker` (`pydc worker`)
selects and configures its `ExecutionBackend` (see `worker/backend.py`) at
startup, without any application code changes -- see `worker/config.py`.
View all available options:

```bash
pydc worker --help
```

### Options

| CLI flag | Environment variable | Default | Meaning |
|---|---|---|---|
| `--backend` | `PY_DISTRIBUTED_BACKEND` | `direct` | `direct` or `multiprocessing` |
| `--max-workers` | `PY_DISTRIBUTED_MAX_WORKERS` | pool default (`os.cpu_count()`) | process pool size (`multiprocessing` only) |
| `--max-in-flight` | `PY_DISTRIBUTED_MAX_IN_FLIGHT` | unbounded | cap on concurrently in-flight executions (`multiprocessing` only) |
| `--mp-context` | `PY_DISTRIBUTED_MP_CONTEXT` | `fork` | `fork`, `spawn`, or `forkserver` (`multiprocessing` only) |

`pydc worker` also accepts, same CLI/env/defaults precedence, resolved
independently of the backend options above:

| CLI flag | Environment variable | Default | Meaning |
|---|---|---|---|
| `--master-host` | `PY_DISTRIBUTED_MASTER_HOST` | `127.0.0.1` | host to connect to |
| `--master-port` | `PY_DISTRIBUTED_MASTER_PORT` | `5000` | port to connect to |
| `--worker-id` | `PY_DISTRIBUTED_WORKER_ID` | a random id | reported to the master on REGISTER |
| `--worker-host` | `PY_DISTRIBUTED_WORKER_HOST` | `127.0.0.1` | reported to the master as metadata only |
| `--worker-port` | `PY_DISTRIBUTED_WORKER_PORT` | `6001` | reported to the master as metadata only |

`pydc master` accepts `--host`/`--port` (same `PY_DISTRIBUTED_MASTER_HOST`/
`PY_DISTRIBUTED_MASTER_PORT` environment variables -- a worker uses them to
know where to *connect*; a master started via its own CLI uses the same
two variables to know where to *bind*, so one pair of env vars covers a
typical deployment either way).

### Precedence

**CLI arguments > environment variables > defaults.** A field omitted from
the CLI falls back to its environment variable; a field set in neither
falls back to its default -- resolved independently per field, so e.g.
`--max-workers` on the CLI and `PY_DISTRIBUTED_MAX_IN_FLIGHT` in the
environment can both apply to the same worker at once.

### Examples

Direct backend (the default -- runs tasks in-process, identical to every
worker before Phase 11):

```bash
pydc worker
```

Multiprocessing backend via CLI flags:

```bash
pydc worker --backend multiprocessing --max-workers 4 --max-in-flight 8
```

Multiprocessing backend via environment variables:

```bash
export PY_DISTRIBUTED_BACKEND=multiprocessing
export PY_DISTRIBUTED_MAX_WORKERS=4
export PY_DISTRIBUTED_MAX_IN_FLIGHT=8
export PY_DISTRIBUTED_MP_CONTEXT=spawn
pydc worker
```

Invalid configuration is rejected immediately with an actionable message,
e.g. `max_workers must be a positive integer or None; received -2` or
`backend must be one of ('direct', 'multiprocessing'); received 'gpu'` --
never a generic error or a silent fallback to a working value.

The resolved configuration is visible at startup: a `Selected backend: ...`
line is printed, and `backend_config_resolved`/`worker_starting` are logged
at INFO (see `worker/backend.py` and `worker/config.py`'s module loggers --
`pydc worker` configures logging to stdout by default).

See `worker/backend.py`'s `MultiprocessingBackend` docstring for what
`mp_context` actually changes (fork vs. spawn/forkserver registered-function
visibility) and `README.md`'s Security Considerations section for what a
worker trusts regardless of backend.

## Example Jobs

See [`examples/`](examples/) for runnable scripts covering every pattern
below — each one is also run by `tests/test_examples.py`, so they're
guaranteed to still work against the current code.

Built-in operations, no custom code needed (`worker/executor.py`):

- **Single call**: `ADD`, `MULTIPLY`
- **Map**: `WORD_COUNT`, and the rest of `MAP_OPERATIONS`
- **Reduce**: `SUM`, `COUNT`, `MAX`, `MIN`, and the rest of `REDUCE_OPERATIONS`

Word count, end to end (`examples/04_map_reduce.py`):

```text
Input:  ["apple", "banana", "apple", "orange", "banana", "apple"]
Map:    WORD_COUNT   (each word -> [word, 1])
Shuffle: group by word
Reduce: SUM           (sum each word's 1s)
Output: {"apple": 3, "banana": 2, "orange": 1}
```

Beyond the built-ins: a **registered function** (`worker.register_function`,
`examples/01_registered_function.py`) for code the worker operator
already trusts, or a **serialized callable**
(`jobs.submit_serialized_call`, `examples/02_serialized_callable.py`)
for an arbitrary function shipped as data — see "Security
Considerations" below for the trust boundary between the two.

## Security Considerations

**Serialized callables are executable code, not data.** The engine supports
submitting an arbitrary Python function (`jobs.call.submit_serialized_call`,
or `jobs.map.build_map_job`/`jobs.reduce.build_reduce_job` given an
`ExecutionSpec.serialized(fn)` operation)
by serializing it with `cloudpickle` and shipping it to a worker, which
deserializes and calls it. A worker that accepts and runs one is trusting
whatever produced it exactly as much as it trusts its own source code --
this is remote code execution by design, not a bug or an oversight.

That is an acceptable, deliberate trade-off for a cooperative cluster
where every master and worker is operated by the same trusted party (this
project's scope throughout). It is **not** safe to expose to, or accept
submissions from, an untrusted client or network. If a deployment ever
needs to run code from a submitter it doesn't fully trust, that requires
actual process/OS-level sandboxing of the worker -- not a check anywhere
in this codebase, and not something planned here. See
`worker/serialization.py`'s module docstring for the full rationale.

The registered-function path (`worker.registry`, `jobs.call.submit_call`)
does not have this exposure: only functions a worker operator already
chose to import and register can run, regardless of what a task's
payload contains -- a task can select *which* registered function runs
and *what arguments* it receives, never *what code* runs.

## Fault Tolerance

The master periodically receives heartbeat messages from workers.

If a worker stops responding:

```text
Worker 2 → No heartbeat
              |
              v
        Failure detected
              |
              v
      Unfinished task found
              |
              v
       Task assigned again
              |
              v
        Task completed
```

This prevents a single worker failure from causing the entire job to fail.

### Retry Semantics

Retry is a **policy decision**, not an automatic property of every
failure -- the two kinds of failure a task can have are handled
differently, deliberately:

| Failure kind | Example | Retried? |
|---|---|---|
| Worker/transport failure | connection dies, heartbeat times out | Yes -- requeued and reassigned to another worker (up to `MAX_TASK_ATTEMPTS`) |
| Execution failure | the function itself raises, bad arguments, non-serializable result | No -- the task is marked `FAILED` at its current attempt |

The reasoning: a worker/transport failure is usually transient (a crash,
a network blip) and unrelated to the task itself, so retrying on a
different worker is likely to succeed. An execution failure is
deterministic -- the same function called with the same arguments on any
worker fails the same way (see `worker/executor.py`'s `ExecutionErrorCode`
categories) -- so retrying it would just reproduce the identical failure
and waste a worker slot. This distinction currently only has two buckets;
a future retry policy could reasonably subdivide "execution failure"
further (e.g. a transient error inside a registered function, which
*would* benefit from a retry, versus a deterministic bug, which wouldn't)
-- not implemented today, and worth tracking before relying on retries
for anything beyond worker/transport failures.

## Learning Objectives

This project is intended to develop practical knowledge of:

- Client-server networking
- TCP communication
- Protocol design
- Distributed system architecture
- Asynchronous programming
- Process-based parallelism
- Task scheduling
- MapReduce algorithms
- Failure detection
- Fault tolerance
- Distributed state management
- Logging and observability
- Performance analysis
- Software testing
- Packaging and CLI design

## Future Improvements

Deliberately not built yet — some explicitly deferred by name in past
phase reviews, kept here as a single honest list rather than scattered
across docstrings:

- Remote job submission to an already-running master (see
  `docs/architecture.md`'s note on why `pydc run` is local-only today —
  this needs a new client-facing wire-protocol message)
- Persistent job metadata (everything is in-memory; a master restart
  loses all task/worker state)
- Worker resource tracking (CPU/memory), task prioritization, and
  scheduler-level backpressure (explicitly out of scope through Phase
  11.4 — see `worker/backend.py`'s concurrency-controls docstrings)
- Dynamic worker discovery / data locality
- Intermediate-result storage beyond one job's in-memory `IntermediateResultStore`
- Job cancellation
- Authentication between workers and master (the wire protocol has none
  today — see "Security Considerations")
- Web-based monitoring dashboard
- Distributed file storage
- Docker-based deployment
- A namespace migration for the top-level package names (`master`,
  `worker`, etc. are generic and could collide with an unrelated
  package in a shared environment — a known, deliberately deferred
  issue from the packaging phase)
- More advanced scheduling algorithms

## Contributing

Contributions, suggestions, and improvements are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Implement your changes.
4. Add or update tests.
5. Submit a pull request.

## License

This project is licensed under the MIT License.

## Author

**Nivash Sharma**

Built as a practical exploration of distributed systems and advanced Python engineering.