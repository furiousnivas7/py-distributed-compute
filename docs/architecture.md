# Architecture

This document describes how the engine actually works, written from the
code (`master/`, `worker/`, `rpc/`, `jobs/`, `common/`), not from an
early plan. For the public API you'd actually import, see
[`api.md`](api.md); for running things locally, see
[`development.md`](development.md).

## Master ↔ worker, at a glance

```text
        +----------------------------+
        |          Master            |
        |  (master/async_server.py)  |
        |                            |
        |   Scheduler   WorkerManager|
        |  (master/scheduler.py,     |
        |   worker_manager.py)       |
        +-------------+--------------+
                      |
           one persistent TCP connection
             per worker (rpc/protocol.py)
                      |
        +-------------+--------------+
        |             |              |
        v             v              v
  +----------+  +----------+  +----------+
  | Worker A |  | Worker B |  | Worker C |
  |(worker/  |  |          |  |          |
  | async_   |  |          |  |          |
  | worker.py)| |          |  |          |
  +----------+  +----------+  +----------+
        |             |              |
        v             v              v
  ExecutionBackend (worker/backend.py)
  DirectBackend or MultiprocessingBackend
```

There is exactly **one persistent TCP connection per worker**, opened by
the worker (the master never dials out). Every message type that
connection ever carries — `REGISTER`, `HEARTBEAT`, `TASK`, `TASK_RESULT`,
`PING`, `SHUTDOWN` — flows over that same connection, multiplexed by
`request_id`. This is a deliberate simplification the async master makes
over the older threaded one (`master/server.py`, `worker/worker.py`,
kept only for their own tests): a thread-based implementation needed a
*second* port and a dedicated heartbeat-listener thread specifically to
avoid two OS threads calling `recv()` on the same socket at once; asyncio
has exactly one coroutine reading a given connection at a time by
construction, so that problem doesn't exist here.

**There is no client → master RPC message type.** Job submission
(`scheduler.submit_task(...)`) happens by calling Python functions
directly against an in-process `Scheduler` object — normally the module
singleton `master.scheduler` a master process holds. `pydc run` (see
[`api.md`](api.md#pydc-cli)) is the closest thing to "submit a job from
a separate process" this engine supports today: it starts master +
worker in one process, submits, and exits. Submitting to an
already-running, separate master process is a real gap, not yet built.

## The wire protocol

Defined in `rpc/protocol.py`. Every message is a JSON object with three
fields:

```json
{"type": "TASK", "request_id": "task-t1:1", "payload": {...}}
```

| Type | Direction | Meaning |
|---|---|---|
| `REGISTER` / `REGISTER_ACK` | worker → master | a worker joins the cluster |
| `HEARTBEAT` / `HEARTBEAT_ACK` | worker → master | liveness, sent on a timer |
| `TASK` / `TASK_RESULT` | master → worker → master | dispatch and its outcome |
| `PING` / `PONG` | either | basic reachability check |
| `SHUTDOWN` / `SHUTDOWN_ACK` | worker → master | "I intend to stop accepting new work" |
| `ERROR` | master → worker (or vice versa) | malformed message, or a `TASK` that could never be delivered (`WORKER_UNREACHABLE`) |

Messages are length-prefixed over the raw TCP stream (`rpc/async_connection.py`)
— there's no HTTP, no framing library, just a 4-byte length header before
each JSON payload.

## Task lifecycle

`common/models.py`'s `Task`:

```text
PENDING → ASSIGNED → RUNNING → COMPLETED
                             ↘ FAILED
```

- **PENDING**: submitted (`Scheduler.submit_task`), not yet given to a worker.
- **ASSIGNED**: `Scheduler.assign_task` picked an `IDLE` worker for it, bumped `task.attempt`, and marked that worker `BUSY`.
- **RUNNING**: the master actually sent the `TASK` message and is awaiting a reply (`Scheduler.start_task`).
- **COMPLETED** / **FAILED**: a terminal outcome (`Scheduler.complete_task` / `fail_task`).

### Attempts: two failure categories, two different outcomes

Every assignment increments `task.attempt`; the wire-level `request_id`
for that specific try is `f"task-{task_id}:{attempt}"`
(`async_server.attempt_request_id`) — not just `task_id`. That distinction
exists so a late reply to an *earlier* attempt (say, a worker that was slow
to answer attempt 1 after the master already gave up and dispatched
attempt 2 elsewhere) can never be mistaken for the reply to the attempt
that's actually still in flight.

A task can fail two structurally different ways, and this engine treats
them differently on purpose (see the README's "Retry Semantics" for the
policy rationale):

| Failure | Detected by | Outcome |
|---|---|---|
| **Worker/transport failure** — connection dies, heartbeat times out | `requeue_tasks_for_worker` (called from a dead connection or the failure monitor) | requeued to `PENDING` and reassigned, up to `MAX_TASK_ATTEMPTS` (3) |
| **Execution failure** — the function raised, bad arguments, non-serializable result | `dispatch_assigned_task` sees `{"status": "error", ...}` in the `TASK_RESULT` | `FAILED` immediately, **never retried** — deterministic, retrying would just reproduce it |

### Worker generations: surviving reconnection

A worker that crashes and comes back reuses its `worker_id`, but the
master needs to make sure a *stale* connection for that same id can never
mutate state on behalf of the new one. `Worker.generation` (bumped on
every successful re-registration, `common/models.py`) plus `WorkerLink`
identity comparison in `connections[worker_id]` is what enforces this:
when a worker re-registers on a new connection, the old connection (if
still technically open) is proactively closed, and a narrow race — a
message from the old connection that was already in-flight in the OS
buffer at the moment of closing — is closed off by checking
`connections.get(worker_id) is link` before processing *anything* on a
read loop (see `handle_worker_connection`'s own extensive comments in
`master/async_server.py`). Once superseded, an old connection's read
loop can only ever hit that check and exit — it can never again touch
`worker_manager`/`scheduler` state.

## Scheduling and dispatch

`master/scheduler.py`'s `Scheduler` owns task/worker state transitions;
it never talks to the network directly. `master/async_server.py`'s
**dispatcher** is what actually drives it:

```text
scheduler.submit_task(...)
        |
        v
wait_for_tasks(task_ids)        <- the one supported public entry point
        |
        v
ensure_dispatcher_running()     <- idempotent; starts dispatcher_loop
        |
        v
   dispatcher_loop               <- the ONLY caller of
        |                           scheduler.assign_next_pending_task()
        v                           once it's running
     scheduler
        |
        v
dispatch_assigned_task           <- records the terminal outcome into
        |                            the response registry
        v
      worker
```

`wait_for_tasks(task_ids)` is the one production entry point for
"submit work and get results back" — it starts the dispatcher if it
isn't already running and awaits each task's terminal response via a
per-task-id `asyncio.Future`, so concurrent callers (two different jobs,
or a job running alongside ad-hoc tasks) never see each other's results.

`drain_tasks_for` / `drain_pending_tasks` are **legacy**: they predate
the centralized dispatcher and run their own independent assign-and-
dispatch loop directly against the shared scheduler. They're kept
because a large share of the test suite drives dispatch manually this
way, and because `jobs.map_reduce.run_map_reduce` takes a `dispatch`
function as a parameter specifically so it can be pointed at either —
but they are not safe to call concurrently with each other or with the
dispatcher, and new production code should use `wait_for_tasks` instead.

## MapReduce data flow

`jobs/map.py` → `jobs/shuffle.py` → `jobs/reduce.py`, wired together by
`jobs/map_reduce.py::run_map_reduce`:

```text
data (a list)
     |
     v
partition(data, num_partitions)          jobs/map.py
     |
     v
one MAP task per partition   ----->  worker.executor.execute_map
     |                                  (built-in, registered, or
     v                                  serialized-callable operation)
IntermediateResult per partition        jobs/map.py
     |
     v
shuffle(results) -> {key: [values]}     jobs/shuffle.py
     |
     v
one REDUCE task per key      ----->  worker.executor.execute_reduce
     |
     v
{key: reduced_value}                    jobs/reduce.py
```

Map failures are tolerated — a failed/missing partition just contributes
nothing to the shuffle. Reduce failures are **not** tolerated:
`collect_reduce_results` raises rather than silently dropping a key,
since a missing key would corrupt the final result with no signal
anything went wrong.

`map_operation`/`reduce_operation` each independently accept a plain
string (a built-in like `WORD_COUNT`/`SUM`, or the name of a function
registered via `worker.registry`) or `ExecutionSpec.serialized(fn)` for
an arbitrary callable (`jobs/models.py`) — all four combinations
(registered+registered, serialized+serialized, and the two mixed cases)
go through the exact same code path.

## The execution-backend abstraction

Separates **what** gets executed (a `task_type` + `payload` — the same
contract regardless of backend) from **how**: `worker/backend.py`'s
`ExecutionBackend` ABC, with two implementations today:

- **`DirectBackend`** — runs `worker.executor.execute_task` in the same
  process and coroutine that calls it. The default; identical to how
  every worker behaved before this abstraction existed.
- **`MultiprocessingBackend`** — runs each task in a child process from
  a `concurrent.futures.ProcessPoolExecutor`, so a CPU-bound task doesn't
  block the worker's own event loop (and therefore its heartbeat, or its
  ability to notice a `SHUTDOWN`). Configurable `max_workers` (pool
  size), `max_in_flight` (a semaphore-based cap distinct from pool size),
  and `mp_context` (`fork`/`spawn`/`forkserver`).

`execute()` is deliberately `async`, even though `DirectBackend`'s
implementation underneath is a plain blocking call — a backend that
hands work to a process pool needs to be able to `await` that work
without blocking the event loop, and defining the interface as async
from the start avoided a breaking change later.

**Two failure categories, handled differently, exactly mirroring the
task-lifecycle table above:** an execution failure (the function itself
raises) must *never* raise out of `execute()` — it comes back as a
normal `{"status": "error", ...}` result. A **backend** failure (the
child process itself died — `BrokenProcessPool`) is allowed to raise;
`serve_tasks()` has no special handling for it, so it propagates and
tears down the worker's connection to the master, which the master
already treats exactly like any other worker crash (mark `FAILED`,
requeue the task) — reusing the existing connection-death retry path
instead of inventing a second one.

Backend selection/configuration is a worker **runtime** concern, not a
scheduler one — see `worker/config.py` and the CLI docs in
[`api.md`](api.md#pydc-cli).

## Execution modes: built-ins, registered functions, serialized callables

Every task's actual computation goes through `worker/executor.py::execute_task`,
regardless of which backend runs it. Three ways a task can specify *what*
to run:

1. **Built-in operations** — `ADD`, `MULTIPLY`, and the `MAP_OPERATIONS`/
   `REDUCE_OPERATIONS` tables (`WORD_COUNT`, `SUM`, `COUNT`, `MAX`, `MIN`, ...).
2. **Registered functions** (`worker.registry`) — a worker operator
   explicitly imports and registers a function under a name; a task can
   select *which* registered function runs and *what arguments* it gets,
   never *what code* runs. No remote-code-execution exposure.
3. **Serialized callables** (`worker.serialization`, cloudpickle) — an
   arbitrary function shipped as part of the task payload itself. This
   **is** remote code execution by design — see the README's "Security
   Considerations" section before using it on anything but a fully
   trusted cluster.

All three produce the exact same result shape:
`{"status": "success", "result": ...}` or
`{"status": "error", "code": ..., "message": ...}` — see `worker/executor.py`'s
`ExecutionErrorCode` for the full taxonomy of structured failure codes.
