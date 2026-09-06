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
"""

import asyncio

from common.models import WorkerStatus
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

    def mark_worker_failed_and_requeue() -> None:
        if is_current_attempt():
            worker_manager.update_status(worker_id, WorkerStatus.FAILED)
            scheduler.requeue_tasks_for_worker(worker_id)

    if link is None:
        mark_worker_failed_and_requeue()
        return build_message(
            protocol.ERROR,
            f"task-{task.task_id}",
            {"task_id": task.task_id, "code": "WORKER_UNREACHABLE", "message": "no connection"},
        )

    try:
        response = await link.send_task(task.task_id, task.task_type, task.payload, attempt)
    except (ConnectionError, ProtocolError) as exc:
        # Worker died mid-dispatch -- as strong a failure signal as a
        # heartbeat timeout, so it converges on the same recovery action.
        mark_worker_failed_and_requeue()
        return build_message(
            protocol.ERROR,
            f"task-{task.task_id}",
            {"task_id": task.task_id, "code": "WORKER_UNREACHABLE", "message": str(exc)},
        )

    if response["type"] != protocol.TASK_RESULT:
        if is_current_attempt():
            scheduler.fail_task(task.task_id)
        return response

    if not is_current_attempt():
        return response

    result = response["payload"]
    if result["status"] == "success":
        scheduler.complete_task(task.task_id)
    else:
        scheduler.fail_task(task.task_id)

    return response


async def drain_pending_tasks() -> list[dict]:
    """Repeatedly assign and dispatch PENDING tasks to IDLE workers until none remain.

    Each round assigns one task per currently-IDLE worker (pure in-memory,
    synchronous, no await) before dispatching that whole round concurrently
    via asyncio.gather -- the async equivalent of the threaded
    implementation's one-thread-per-task dispatch.
    """
    responses = []

    while True:
        batch = []
        while True:
            task = scheduler.assign_next_pending_task()
            if task is None:
                break
            batch.append(task)

        if not batch:
            break

        responses.extend(await asyncio.gather(*(dispatch_assigned_task(task) for task in batch)))

    return responses


async def failure_monitor(stop_event: asyncio.Event) -> None:
    """Every FAILURE_CHECK_INTERVAL seconds, mark workers FAILED if they've
    gone HEARTBEAT_TIMEOUT seconds without a heartbeat, requeue their tasks,
    and try to hand those tasks to whoever's still idle."""
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
        for task_id, task_type, task_payload in demo_tasks:
            scheduler.submit_task(task_id, task_type, task_payload)

        await drain_pending_tasks()

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
