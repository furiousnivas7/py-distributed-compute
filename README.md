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

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Start the master:

```bash
python -m master.server
```

Start a worker:

```bash
python -m worker.worker
```

Start additional workers in separate terminals:

```bash
python -m worker.worker
```

Submit a job:

```bash
python client.py submit wordcount data.txt
```

> The exact commands may change during development.

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
`jobs.map.build_map_job_serialized`, `jobs.reduce.build_reduce_job_serialized`)
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