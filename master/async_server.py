"""Async TCP master: registers workers, then schedules and dispatches tasks
to them, all on a single asyncio event loop instead of one OS thread per
connection / per dispatch / for the heartbeat listener / for the failure
monitor (see master/server.py for the threaded implementation this mirrors).

Architectural simplification enabled by asyncio (see WorkerLink below): a
worker's REGISTER, HEARTBEAT, and TASK_RESULT messages all flow over the
SAME persistent connection, multiplexed by request_id. The threaded
implementation needed a second port and a dedicated heartbeat-listener
thread specifically to avoid two OS threads calling recv() on the same
socket; asyncio has exactly one coroutine reading a given connection at a
time by construction, so that whole problem doesn't exist here.

DISPATCH CONTRACT (Phase 8.9)
------------------------------
This module has exactly one PUBLIC, concurrency-safe way to run tasks and
get their results:

    submit task(s) via scheduler.submit_task(...)
                  |
                  v
         wait_for_tasks(task_ids)   <-- the only supported entry point
                  |
                  v
    ensure_dispatcher_running()  (idempotent, starts dispatcher_loop)
                  |
                  v
             dispatcher_loop        <-- the ONLY caller of
                  |                     scheduler.assign_next_pending_task()
                  v                     once it's running
              scheduler
                  |
                  v
           dispatch_assigned_task    <-- records into the response
                  |                      registry (record_if_terminal)
                  v
                worker

Everything below `dispatch_assigned_task` in that diagram (scheduler,
worker links, the response registry) is internal machinery. New
production code that needs to run tasks and wait for results should call
`wait_for_tasks` and nothing else in this module.

`drain_tasks_for` / `drain_pending_tasks` are LEGACY: they predate the
centralized dispatcher and run their own independent assign-and-dispatch
loop directly against the shared scheduler. They are kept only because a
large share of the pre-8.9 test suite drives dispatch manually (submit a
task, then call one of these to pump it through) and depends on that
precise manual control -- an always-on dispatcher would race with it.
They remain exactly as unsafe for concurrent callers as they always were
(see drain_tasks_for's own docstring) and must not be used by new
production code, or by two callers at once. Do not build new features on
top of them; migrate a call site off them entirely rather than adding to
it.
"""

import asyncio

from common.models import TaskStatus, WorkerStatus
from master import rpc_handler
from master.scheduler import Scheduler
from rpc import protocol
from rpc.async_connection import AsyncConnection
from rpc.async_rpc import receive_message, send_message
from rpc.protocol import ProtocolError, build_message

HOST = "127.0.0.1"
PORT = 5000
EXPECTED_WORKERS = 2
HEARTBEAT_TIMEOUT = 5.0
FAILURE_CHECK_INTERVAL = 1.0

# rpc_handler.handle_register/handle_heartbeat are hardcoded to mutate
# rpc_handler's own module-level WorkerManager, so this has to be the same
# object rather than an independent instance -- there's no way to reuse
# rpc_handler.handle_request otherwise. The Scheduler, by contrast, isn't
# referenced anywhere inside rpc_handler, so this module gets its own,
# separate from master.server.scheduler.
worker_manager = rpc_handler.worker_manager
scheduler = Scheduler(worker_manager)

# worker_id -> WorkerLink, populated once a connection's REGISTER succeeds.
connections: dict[str, "WorkerLink"] = {}

# Phase 8.9 -- centralized task ownership and response routing.
#
# Every dispatch, regardless of which code path triggered it (the legacy
# drain_pending_tasks()/drain_tasks_for() loops, a test calling
# dispatch_assigned_task() directly, or the dispatcher_loop below), writes
# its task's terminal outcome here exactly once. Keyed by task_id, so any
# caller can retrieve or await a specific task's result without needing to
# be the one that happened to dispatch it -- this is what makes
# wait_for_tasks() safe for concurrent callers without job-scoping the
# assignment step at all (contrast with drain_tasks_for()'s task_ids
# filter, Phase 8.8's fix, which is still available and still used by
# tests that call it directly).
_task_responses: dict[str, dict] = {}
_task_futures: dict[str, asyncio.Future] = {}

_dispatcher_task: asyncio.Task | None = None
_dispatcher_stop_event: asyncio.Event | None = None
_dispatcher_event_loop: asyncio.AbstractEventLoop | None = None


def _record_task_response(task_id: str, response: dict) -> None:
    """Record a task's terminal outcome and resolve anyone awaiting it.

    Response payloads always carry the attempt that produced them
    (`payload["attempt"]`), so even though this registry is keyed by
    task_id alone -- a caller wants "the final answer for this task",
    not "attempt N's answer" -- which attempt actually won is never lost.
    """
    _task_responses[task_id] = response
    future = _task_futures.get(task_id)
    if future is not None and not future.done():
        future.set_result(response)


def clear_dispatch_registry() -> None:
    """Reset the response registry and any pending futures.

    Module-level state, exactly like rpc_handler.worker_manager and this
    module's own `scheduler` -- tests must clear it between runs (see
    tests/conftest.py) or a stale, already-resolved response left over
    from an earlier test reusing the same task_id could resolve a later,
    unrelated wait_for_tasks() call immediately with the wrong data.
    """
    _task_responses.clear()
    for future in _task_futures.values():
        if not future.done():
            future.cancel()
    _task_futures.clear()


class WorkerLink:
    """Owns the single read loop for one worker's persistent connection.

    Only one coroutine may safely read from an AsyncConnection at a time, so
    a coroutine that wants to send a TASK and await its TASK_RESULT can't
    just read the reply itself -- another message (a HEARTBEAT, say) could
    legitimately arrive first. Instead it registers a Future here, keyed by
    the TASK's request_id, and awaits that; the connection's one read loop
    (see handle_worker_connection) resolves it when the matching reply
    shows up, or fails every pending Future if the connection dies first.
    """

    def __init__(self, conn: AsyncConnection):
        self.conn = conn
        self._pending: dict[str, asyncio.Future] = {}

    async def send_task(self, task_id: str, task_type: str, task_payload: dict, attempt: int) -> dict:
        request_id = f"task-{task_id}"
        request = build_message(
            protocol.TASK,
            request_id,
            {"task_id": task_id, "task_type": task_type, "task_payload": task_payload, "attempt": attempt},
        )

        future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        try:
            await send_message(self.conn, request)
            return await future
        finally:
            self._pending.pop(request_id, None)

    def resolve(self, request_id: str, message: dict) -> bool:
        """Resolve a pending send_task() future if `request_id` matches one. Returns whether it did."""
        future = self._pending.get(request_id)
        if future is None or future.done():
            return False
        future.set_result(message)
        return True

    def fail_pending(self, exc: Exception) -> None:
        for future in self._pending.values():
            if not future.done():
                future.set_exception(exc)


async def handle_worker_connection(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Per-connection coroutine registered with asyncio.start_server.

    Reads every message on this connection for its whole lifetime. A
    message whose request_id matches an in-flight send_task() call resolves
    that call's Future; anything else (PING, REGISTER, HEARTBEAT, or an
    unrecognized type) is dispatched through the existing, unmodified
    rpc_handler.handle_request and replied to inline -- exactly like the
    synchronous master's per-connection loop, just without needing a
    separate phase for "registration" vs. "everything after."
    """
    conn = AsyncConnection(reader, writer)
    link = WorkerLink(conn)
    worker_id = None

    try:
        while True:
            try:
                message = await receive_message(conn)
            except ConnectionError:
                return
            except ProtocolError as exc:
                error = build_message(protocol.ERROR, "unknown", {"code": "INVALID_MESSAGE", "message": str(exc)})
                await send_message(conn, error)
                continue

            request_id = message.get("request_id")
            if request_id and link.resolve(request_id, message):
                continue

            response = rpc_handler.handle_request(message)
            await send_message(conn, response)

            if message["type"] == protocol.REGISTER and response["type"] == protocol.REGISTER_ACK:
                worker_id = message["payload"]["worker_id"]
                connections[worker_id] = link
    finally:
        if worker_id is not None:
            connections.pop(worker_id, None)
        link.fail_pending(ConnectionError("Worker connection closed"))
        await conn.close()


async def dispatch_assigned_task(task) -> dict:
    """Send an ASSIGNED task to its worker and update scheduler state from the result.

    Mirrors master.server.dispatch_assigned_task's capture-then-recheck
    pattern (worker_id and attempt are captured before the await, and
    rechecked after it) for the same reason: the failure monitor could
    requeue and reassign this task to someone else while this coroutine is
    suspended awaiting the reply. No lock is needed here the way the
    threaded version needed one: none of scheduler's or worker_manager's
    methods contain an await, so each call already runs to completion
    atomically with respect to every other coroutine on this loop. A lock
    would only matter for a *sequence* of such calls spanning an await --
    which is exactly what the capture/recheck below replaces.
    """
    link = connections.get(task.assigned_worker_id)
    worker_id = task.assigned_worker_id
    attempt = task.attempt

    scheduler.start_task(task.task_id)

    def is_current_attempt() -> bool:
        return task.assigned_worker_id == worker_id and task.attempt == attempt

    def record_if_terminal(response: dict, was_current: bool) -> None:
        # Only the call whose (worker_id, attempt) still matched the
        # task's state at the moment it detected the failure/result gets to
        # decide whether that state is terminal -- a stale/late call for an
        # attempt that's since moved on must never overwrite what a later,
        # winning attempt already recorded. `was_current` must be captured
        # BEFORE any mutation (e.g. mark_worker_failed_and_requeue, below)
        # that could itself flip is_current_attempt() -- re-checking it
        # AFTER such a mutation would make this call look "stale" against
        # state it just changed itself, even though nothing else ran.
        if was_current and task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
            _record_task_response(task.task_id, response)

    def mark_worker_failed_and_requeue(was_current: bool) -> None:
        if was_current:
            worker_manager.update_status(worker_id, WorkerStatus.FAILED)
            scheduler.requeue_tasks_for_worker(worker_id)

    if link is None:
        was_current = is_current_attempt()
        mark_worker_failed_and_requeue(was_current)
        response = build_message(
            protocol.ERROR,
            f"task-{task.task_id}",
            {"task_id": task.task_id, "attempt": attempt, "code": "WORKER_UNREACHABLE", "message": "no connection"},
        )
        record_if_terminal(response, was_current)
        return response

    try:
        response = await link.send_task(task.task_id, task.task_type, task.payload, attempt)
    except (ConnectionError, ProtocolError) as exc:
        # Worker died mid-dispatch -- as strong a failure signal as a
        # heartbeat timeout, so it converges on the same recovery action.
        was_current = is_current_attempt()
        mark_worker_failed_and_requeue(was_current)
        response = build_message(
            protocol.ERROR,
            f"task-{task.task_id}",
            {"task_id": task.task_id, "attempt": attempt, "code": "WORKER_UNREACHABLE", "message": str(exc)},
        )
        record_if_terminal(response, was_current)
        return response

    if response["type"] != protocol.TASK_RESULT:
        was_current = is_current_attempt()
        if was_current:
            scheduler.fail_task(task.task_id)
        record_if_terminal(response, was_current)
        return response

    if not is_current_attempt():
        return response

    result = response["payload"]
    if result["status"] == "success":
        scheduler.complete_task(task.task_id)
    else:
        scheduler.fail_task(task.task_id)

    record_if_terminal(response, True)
    return response


async def drain_tasks_for(task_ids: set[str] | None, poll_interval: float = 0.01, timeout: float = 30.0) -> list[dict]:
    """LEGACY / TEST-ONLY -- see the module docstring's "DISPATCH CONTRACT"
    section. Predates the Phase 8.9 centralized dispatcher; new production
    code must use wait_for_tasks() instead. Kept only because a large share
    of the pre-8.9 test suite depends on driving dispatch this way. Do not
    add new callers.

    Repeatedly assign and dispatch PENDING tasks to IDLE workers until none remain.

    Each round assigns one task per currently-IDLE worker (pure in-memory,
    synchronous, no await) before dispatching that whole round concurrently
    via asyncio.gather -- the async equivalent of the threaded
    implementation's one-thread-per-task dispatch.

    If `task_ids` is given, only tasks in that set are ever assigned or
    reported on -- this is the fix for a real concurrency hazard: this
    function (or failure_monitor, which calls it internally) can be
    in-flight from more than one caller at once, and assign_next_pending_task
    mutates a scheduler shared by all of them. Without scoping, whichever
    caller's loop happens to run first can "steal" a task a DIFFERENT
    caller submitted, and that task's response then lands in the stealing
    caller's own `responses` list instead of ever reaching the caller that
    actually needed it (see Phase 8.8 notes / tests/test_concurrent_dispatch.py
    for the exact scenario). Passing this job's own task_ids means each
    caller's assignment loop only ever touches its own tasks -- concurrent
    callers no longer compete for the same task, so no lock is needed here:
    every individual Scheduler/WorkerManager call is already atomic (none
    of them contain an await), and disjoint task_id sets mean there's
    nothing left to race over between two job-scoped callers.

    Scoping introduces one wrinkle a global drain never had to deal with:
    an idle-worker *shortage* is genuinely ambiguous for a scoped caller.
    "No task was assignable this round" could mean "every one of my tasks
    is done" (stop) or "my tasks are still pending but another concurrent
    caller currently holds every idle worker" (wait, don't give up) --
    workers are a resource shared across every job-scoped caller, by
    design. So when task_ids is given and a round assigns nothing, this
    checks whether any of `task_ids` are still non-terminal
    (PENDING/ASSIGNED/RUNNING); if so it waits briefly and tries again,
    bounded by `timeout` as a safety net against a genuinely-starved
    workload (e.g. every worker has failed with none left to replace
    them) hanging forever.

    `drain_pending_tasks()` (task_ids=None) keeps the original global
    behavior for callers that don't care about scoping -- the one-off
    manual demo in run_server(), and any test driving a single job/worker
    set in isolation. There, "nothing assignable" unambiguously means
    "everything pending has been dispatched" (a single caller sees the
    whole scheduler), so no wait/retry is needed and none is added. It
    remains just as unsafe to call from multiple concurrent callers as it
    always was; use this function with an explicit scope instead when more
    than one caller may be dispatching at once.
    """
    responses = []
    deadline = None

    while True:
        batch = []
        while True:
            task = scheduler.assign_next_pending_task(task_ids)
            if task is None:
                break
            batch.append(task)

        if batch:
            responses.extend(await asyncio.gather(*(dispatch_assigned_task(task) for task in batch)))
            continue

        if task_ids is None:
            break

        outstanding = {
            task.task_id
            for task in scheduler.get_all_tasks()
            if task.task_id in task_ids and task.status not in (TaskStatus.COMPLETED, TaskStatus.FAILED)
        }
        if not outstanding:
            break

        loop = asyncio.get_running_loop()
        if deadline is None:
            deadline = loop.time() + timeout
        elif loop.time() >= deadline:
            break

        await asyncio.sleep(poll_interval)

    return responses


async def drain_pending_tasks() -> list[dict]:
    """LEGACY / TEST-ONLY -- see the module docstring's "DISPATCH CONTRACT"
    section and drain_tasks_for()'s docstring. New production code must use
    wait_for_tasks() instead (run_server()'s demo already does).

    Drain every currently-PENDING task, regardless of who submitted it.
    Kept for backward compatibility (failure_monitor's pre-8.9 fallback
    sweep, and any test driving a single job/worker set in isolation) --
    but NOT safe to call from multiple concurrent callers. Use
    wait_for_tasks() (or, in a test that must drive dispatch manually,
    drain_tasks_for() with an explicit task_ids scope) instead whenever
    more than one caller may be dispatching at the same time.
    """
    return await drain_tasks_for(None)


async def dispatcher_loop(stop_event: asyncio.Event, poll_interval: float = 0.01) -> None:
    """The single authoritative loop for assigning PENDING tasks to IDLE
    workers and dispatching them, once started (see ensure_dispatcher_running).

    Unlike drain_tasks_for(), this never stops on its own and needs no
    task_ids scope: since it's the ONLY thing that ever calls
    scheduler.assign_next_pending_task() while it's running, there's no
    other caller left to race with over who gets to assign a given
    PENDING task -- every task, regardless of who submitted it, is
    eventually picked up by this same loop, and every dispatch's outcome
    lands in the shared _task_responses registry (see
    dispatch_assigned_task's record_if_terminal), addressable by any
    caller via wait_for_tasks() regardless of dispatch timing.

    Each dispatch runs as its own background task so a slow/stuck worker
    never blocks this loop from noticing and assigning other pending work.
    """
    while True:
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=poll_interval)
            return
        except asyncio.TimeoutError:
            pass

        while True:
            task = scheduler.assign_next_pending_task()
            if task is None:
                break
            asyncio.create_task(dispatch_assigned_task(task))


def is_dispatcher_running() -> bool:
    """Whether the centralized dispatcher is active on the CURRENT event loop.

    Each test (and run_server()) runs its own asyncio.run(), so a
    dispatcher task created by an earlier loop is meaningless here --
    comparing against the loop it was actually created on is what makes
    this safe to call from a fresh loop without confusing a stale
    reference for "still running".
    """
    return (
        _dispatcher_task is not None
        and _dispatcher_event_loop is asyncio.get_running_loop()
        and not _dispatcher_task.done()
    )


def ensure_dispatcher_running() -> None:
    """Idempotently start the centralized dispatcher on the current event
    loop. Safe to call repeatedly (wait_for_tasks() calls this itself) --
    a no-op if it's already running here."""
    global _dispatcher_task, _dispatcher_stop_event, _dispatcher_event_loop
    if is_dispatcher_running():
        return
    _dispatcher_stop_event = asyncio.Event()
    _dispatcher_event_loop = asyncio.get_running_loop()
    _dispatcher_task = asyncio.create_task(dispatcher_loop(_dispatcher_stop_event))


async def stop_dispatcher() -> None:
    """Stop the dispatcher started by ensure_dispatcher_running(), if any is running here."""
    global _dispatcher_task
    if not is_dispatcher_running():
        return
    _dispatcher_stop_event.set()
    await _dispatcher_task
    _dispatcher_task = None


async def wait_for_tasks(task_ids: set[str], timeout: float = 30.0) -> list[dict]:
    """Wait for every task in `task_ids` to reach a terminal state and
    return their responses (order matches iteration of `task_ids`, not
    completion order -- callers that care about a specific order, like
    jobs.map's partition order, already re-sort by task_id themselves).

    Starts the centralized dispatcher if it isn't already running, then
    relies entirely on it to actually assign and dispatch these tasks --
    this function itself never calls assign_next_pending_task or
    dispatch_assigned_task. That's what makes it safe for any number of
    concurrent callers (multiple MapReduce jobs, ordinary ad-hoc tasks,
    and the failure monitor's recovery work all resolve through the same
    dispatcher and the same _task_responses registry): there is no
    caller-specific response list for a response to go missing from.
    """
    ensure_dispatcher_running()

    responses: dict[str, dict] = {}
    pending_ids = []
    for task_id in task_ids:
        if task_id in _task_responses:
            responses[task_id] = _task_responses[task_id]
        else:
            pending_ids.append(task_id)

    if pending_ids:
        futures = []
        for task_id in pending_ids:
            future = _task_futures.get(task_id)
            if future is None or future.done():
                future = asyncio.get_running_loop().create_future()
                _task_futures[task_id] = future
            futures.append(future)

        results = await asyncio.wait_for(asyncio.gather(*futures), timeout=timeout)
        for task_id, result in zip(pending_ids, results):
            responses[task_id] = result

    return [responses[task_id] for task_id in task_ids]


async def failure_monitor(stop_event: asyncio.Event) -> None:
    """Every FAILURE_CHECK_INTERVAL seconds, mark workers FAILED if they've
    gone HEARTBEAT_TIMEOUT seconds without a heartbeat, requeue their tasks,
    and make sure something will pick those tasks back up.

    If the centralized dispatcher (see dispatcher_loop) is already running,
    this deliberately does NOT dispatch anything itself -- the dispatcher
    will assign the newly-PENDING tasks on its own very next tick, and
    dispatching them here too would recreate exactly the race Phase 8.8
    fixed (two independent callers competing over the same scheduler,
    one's response landing in the other's discarded result). This is what
    "the failure monitor submits recovery work to the dispatcher rather
    than independently draining tasks" means in practice: it does nothing
    beyond requeuing once a dispatcher exists to notice.

    When no dispatcher is running (every test and code path predating
    Phase 8.9, which drives dispatch manually or via drain_pending_tasks/
    drain_tasks_for), this falls back to the original behavior so that
    existing recovery tests keep working without needing to adopt the
    dispatcher.
    """
    while True:
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=FAILURE_CHECK_INTERVAL)
            return
        except asyncio.TimeoutError:
            pass

        stale_workers = worker_manager.get_stale_workers(HEARTBEAT_TIMEOUT)
        if not stale_workers:
            continue

        for worker in stale_workers:
            requeued_tasks = scheduler.requeue_tasks_for_worker(worker.worker_id)
            print(f"Worker failed: {worker.worker_id}")
            for task in requeued_tasks:
                print(f"Requeued task: {task.task_id}")

        if is_dispatcher_running():
            continue

        await drain_pending_tasks()


async def wait_for_workers(count: int, poll_interval: float = 0.01) -> None:
    """Wait until at least `count` workers are connected."""
    while len(connections) < count:
        await asyncio.sleep(poll_interval)


async def run_server() -> None:
    server = await asyncio.start_server(handle_worker_connection, HOST, PORT)
    print(f"Async master started on {HOST}:{PORT}")

    stop_event = asyncio.Event()
    monitor_task = asyncio.create_task(failure_monitor(stop_event))

    async with server:
        await wait_for_workers(EXPECTED_WORKERS)

        demo_tasks = [
            ("task-1", "ADD", {"a": 10, "b": 20}),
            ("task-2", "MULTIPLY", {"a": 3, "b": 4}),
            ("task-3", "ADD", {"a": 5, "b": 5}),
            ("task-4", "MULTIPLY", {"a": 2, "b": 2}),
        ]
        task_ids = set()
        for task_id, task_type, task_payload in demo_tasks:
            scheduler.submit_task(task_id, task_type, task_payload)
            task_ids.add(task_id)

        await wait_for_tasks(task_ids)
        await stop_dispatcher()

        print("Final task states:")
        for task in scheduler.get_all_tasks():
            print(f"  {task.task_id}: {task.status} (worker={task.assigned_worker_id})")

        stop_event.set()
        await monitor_task
        server.close()
        await server.wait_closed()


def main() -> None:
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
