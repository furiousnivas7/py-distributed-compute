# Python Distributed Compute Engine

A distributed computing engine built from scratch in Python.

The system allows a large data-processing job to be divided into smaller tasks and executed by multiple worker processes or machines. A central master coordinates the workers, distributes tasks, collects results, and automatically retries failed tasks.

This project is designed to explore the fundamentals of distributed systems, custom network protocols, asynchronous programming, parallel processing, and fault tolerance.

## Features

- Master-worker architecture
- Custom TCP-based RPC protocol
- Worker registration and management
- Asynchronous communication using `asyncio`
- Parallel task execution using `multiprocessing`
- MapReduce programming model
- Remote execution of registered functions and serialized Python callables
- Task scheduling and load distribution
- Worker heartbeat monitoring
- Failed-task detection and retry
- Job and task status tracking
- Command-line interface
- Structured logging
- Performance benchmarking
- Docker-based multi-worker deployment

## Architecture

```text
                         Client
                           |
                           v
                    +--------------+
                    |    Master    |
                    |              |
                    | Job Manager  |
                    | Scheduler   |
                    | Worker Pool |
                    | Fault Check |
                    +------+-------+
                           |
                     Custom TCP RPC
                           |
             +-------------+-------------+
             |             |             |
             v             v             v
       +-----------+ +-----------+ +-----------+
       | Worker 1  | | Worker 2  | | Worker 3  |
       |           | |           | |           |
       | Process 1 | | Process 1 | | Process 1 |
       | Process 2 | | Process 2 | | Process 2 |
       +-----------+ +-----------+ +-----------+
             |             |             |
             +-------------+-------------+
                           |
                           v
                    Final Result
```

## How It Works

The system follows this workflow:

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

## Example

For an input file containing:

```text
apple banana apple
orange apple banana
```

The Map stage produces:

```text
apple  -> 1
banana -> 1
apple  -> 1
orange -> 1
apple  -> 1
banana -> 1
```

The Reduce stage produces:

```text
apple  -> 3
banana -> 2
orange -> 1
```

## Project Status

This project is being developed incrementally.

### Development phases

- [ ] Phase 1: Basic TCP client and server
- [ ] Phase 2: Custom RPC protocol
- [ ] Phase 3: Worker registration
- [ ] Phase 4: Worker heartbeat monitoring
- [ ] Phase 5: Task scheduler
- [ ] Phase 6: Basic MapReduce execution
- [ ] Phase 7: Async communication with `asyncio`
- [ ] Phase 8: Parallel execution with `multiprocessing`
- [ ] Phase 9: Task retry and fault tolerance
- [ ] Phase 10: CLI and monitoring
- [ ] Phase 11: Testing and benchmarking
- [ ] Phase 12: Docker deployment

## Technology Stack

- **Language:** Python 3.11+
- **Networking:** TCP sockets
- **Communication:** Custom JSON-based RPC
- **Concurrency:** `asyncio`
- **Parallelism:** `multiprocessing`
- **Testing:** `pytest`
- **Logging:** Python `logging`
- **Containerization:** Docker
- **Version control:** Git and GitHub

## Planned Project Structure

```text
py-distributed-compute/
│
├── master/
│   ├── server.py
│   ├── scheduler.py
│   └── worker_manager.py
│
├── worker/
│   ├── worker.py
│   └── executor.py
│
├── rpc/
│   ├── protocol.py
│   └── connection.py
│
├── jobs/
│   ├── map.py
│   └── reduce.py
│
├── common/
│   ├── models.py
│   └── logger.py
│
├── tests/
│
├── client.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/py-distributed-compute.git
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
pip install dist/py_distributed_compute-0.1.0-py3-none-any.whl
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

The engine will initially support:

- Word count
- Line count
- Character frequency
- Log analysis
- CSV aggregation
- Numerical data processing

Example:

```text
Input:
server.log

Map:
Extract HTTP status codes

Reduce:
Count each status code

Output:
200: 15420
404: 832
500: 47
```

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
- Docker deployment

## Future Improvements

- Persistent job metadata
- Worker resource tracking
- Task prioritization
- Dynamic worker discovery
- Data locality
- Intermediate-result storage
- Job cancellation
- Authentication between workers and master
- Web-based monitoring dashboard
- Distributed file storage
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