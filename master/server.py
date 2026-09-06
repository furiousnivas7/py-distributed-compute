"""TCP master server: registers workers, then schedules and dispatches tasks to them."""

import socket
import threading

from common.models import WorkerStatus
from master import rpc_handler
from master.scheduler import Scheduler
from rpc import protocol
from rpc.connection import Connection
from rpc.protocol import ProtocolError, build_message

HOST = "127.0.0.1"
PORT = 5000
# Heartbeats get their own listening port rather than sharing PORT. If both
# accept_and_register()'s loop and heartbeat_listener() called accept() on
# the SAME listening socket from different threads, the kernel could hand a
# brand-new worker's registration connection to whichever thread happened to
# be waiting in accept() — if that's the heartbeat listener, it would treat
# the connection as a single fire-and-forget request and close it after one
# message, breaking that worker's PING/REGISTER handshake. Separate sockets
# make that race impossible.
HEARTBEAT_PORT = 5001
EXPECTED_WORKERS = 2
HEARTBEAT_TIMEOUT = 5.0
FAILURE_CHECK_INTERVAL = 1.0

worker_manager = rpc_handler.worker_manager
scheduler = Scheduler(worker_manager)

# Dispatching multiple tasks concurrently (Phase 6.3) runs one thread per
# in-flight task. The network I/O (send/recv on each worker's own socket)
# is safe to run in parallel, but the Scheduler/WorkerManager dicts are
# shared mutable state, so every state transition is serialized through
# this lock. Only the blocking network call happens outside it — that's
# what lets two workers actually run at the same time.
_state_lock = threading.Lock()


def accept_and_register(server_sock: socket.socket) -> tuple[Connection, str]:
    """Accept one connection and handle requests until the worker registers.

    Returns the open connection and the registered worker_id.
    """
    client_sock, addr = server_sock.accept()
    print("Worker connected")
    conn = Connection(client_sock)

    while True:
        raw = conn.recv_bytes()

        try:
            request = protocol.decode_message(raw)
        except ProtocolError as exc:
            error = build_message(protocol.ERROR, "unknown", {"code": "INVALID_MESSAGE", "message": str(exc)})
            conn.send_bytes(protocol.encode_message(error))
            continue

        print(f"RPC received: {request['type']}")
        response = rpc_handler.handle_request(request)
        print(f"RPC response: {response['type']}")
        conn.send_bytes(protocol.encode_message(response))

        if request["type"] == protocol.REGISTER and response["type"] == protocol.REGISTER_ACK:
            return conn, request["payload"]["worker_id"]


def accept_and_handle_one(server_sock: socket.socket) -> tuple[dict, dict]:
    """Accept a connection, handle exactly one request on it, then close it.

    Used for short-lived, single-purpose connections such as a HEARTBEAT
    ping — kept separate from the long-lived per-worker connection so a
    worker's periodic heartbeats can never race with the master reading a
    TASK_RESULT on that other connection.
    """
    client_sock, addr = server_sock.accept()
    conn = Connection(client_sock)

    try:
        request = protocol.decode_message(conn.recv_bytes())
        response = rpc_handler.handle_request(request)
        conn.send_bytes(protocol.encode_message(response))
        return request, response
    finally:
        conn.close()


def handle_heartbeat_connection(conn: Connection) -> None:
    """Serve exactly one request on an accepted heartbeat connection, then close it."""
    try:
        request = protocol.decode_message(conn.recv_bytes())
        response = rpc_handler.handle_request(request)
        conn.send_bytes(protocol.encode_message(response))
    finally:
        conn.close()


def heartbeat_listener(heartbeat_sock: socket.socket, stop_event: threading.Event) -> None:
    """Continuously accept short-lived HEARTBEAT connections on their own port.

    Runs on a background thread for the master's whole lifetime, since
    workers connect fresh for every heartbeat rather than keeping one open.
    A short accept() timeout lets this loop notice stop_event promptly
    instead of blocking in accept() forever.
    """
    heartbeat_sock.settimeout(0.5)

    while not stop_event.is_set():
        try:
            client_sock, addr = heartbeat_sock.accept()
        except socket.timeout:
            continue
        except OSError:
            return

        conn = Connection(client_sock)
        threading.Thread(target=handle_heartbeat_connection, args=(conn,), daemon=True).start()


def failure_monitor(stop_event: threading.Event) -> None:
    """Every FAILURE_CHECK_INTERVAL seconds, mark workers FAILED if they've
    gone HEARTBEAT_TIMEOUT seconds without a heartbeat, and requeue whatever
    tasks they were running so another worker can pick them up.

    Marking a worker FAILED and requeuing its tasks happens under
    _state_lock as one atomic step — the same lock dispatch_assigned_task
    uses — so a task can never be seen as both ASSIGNED-to-a-dead-worker
    and PENDING at the same time from another thread's point of view.
    """
    while not stop_event.wait(FAILURE_CHECK_INTERVAL):
        with _state_lock:
            stale_workers = worker_manager.get_stale_workers(HEARTBEAT_TIMEOUT)

            for worker in stale_workers:
                requeued_tasks = scheduler.requeue_tasks_for_worker(worker.worker_id)

                print(f"Worker failed: {worker.worker_id}")
                for task in requeued_tasks:
                    print(f"Requeued task: {task.task_id}")


def dispatch_task(conn: Connection, task_id: str, task_type: str, task_payload: dict, attempt: int = 1) -> dict:
    """Send a TASK request to a worker and return its decoded TASK_RESULT response."""
    request = build_message(
        protocol.TASK,
        request_id=f"task-{task_id}",
        payload={"task_id": task_id, "task_type": task_type, "task_payload": task_payload, "attempt": attempt},
    )
    print(f"RPC sent: {request['type']}")
    conn.send_bytes(protocol.encode_message(request))

    response = protocol.decode_message(conn.recv_bytes())
    print(f"RPC received: {response['type']}")
    return response


def dispatch_assigned_task(connections: dict[str, Connection], task) -> dict:
    """Send an ASSIGNED task to its worker and update scheduler state from the result.

    `connections` maps worker_id -> Connection, since the scheduler only knows
    which worker_id a task was assigned to, not how to reach it over the network.
    Safe to call from multiple threads at once, each with a different task:
    the network call runs unlocked (so workers genuinely run concurrently),
    and only the surrounding scheduler state changes are locked.

    The worker_id and attempt are captured up front, before the blocking
    network call. If the failure monitor requeues (and someone else later
    reassigns) this same task while we're waiting on this worker's reply,
    task.assigned_worker_id/task.attempt will have moved on by the time the
    reply arrives — so a late result from a dead attempt can never overwrite
    a newer one, even though both share the same underlying Task object.
    """
    conn = connections[task.assigned_worker_id]
    worker_id = task.assigned_worker_id
    attempt = task.attempt

    with _state_lock:
        scheduler.start_task(task.task_id)

    def is_current_attempt() -> bool:
        return task.assigned_worker_id == worker_id and task.attempt == attempt

    try:
        response = dispatch_task(conn, task.task_id, task.task_type, task.payload, attempt)
    except (ConnectionError, OSError, ProtocolError) as exc:
        # The worker died (or the connection otherwise broke) mid-dispatch —
        # this is direct evidence of failure, just as strong as a heartbeat
        # timeout, so it converges on the same recovery action: mark the
        # worker FAILED and requeue its tasks for retry, not a terminal
        # fail_task. That way it doesn't matter whether this connection-level
        # detection or the heartbeat-based failure_monitor notices first —
        # both paths land on "worker FAILED, task back to PENDING." Only do
        # this if it's still the current attempt — if the failure monitor
        # already requeued and reassigned this task while we were blocked
        # here, this dead attempt must not touch the new one's state.
        with _state_lock:
            if is_current_attempt():
                worker_manager.update_status(worker_id, WorkerStatus.FAILED)
                scheduler.requeue_tasks_for_worker(worker_id)
        return build_message(
            protocol.ERROR,
            f"task-{task.task_id}",
            {"code": "WORKER_UNREACHABLE", "message": str(exc)},
        )

    if response["type"] != protocol.TASK_RESULT:
        with _state_lock:
            if is_current_attempt():
                scheduler.fail_task(task.task_id)
        return response

    result = response["payload"]

    with _state_lock:
        if not is_current_attempt():
            return response

        if result["status"] == "success":
            scheduler.complete_task(task.task_id)
        else:
            scheduler.fail_task(task.task_id)

    return response


def dispatch_concurrently(connections: dict[str, Connection], tasks: list) -> list[dict]:
    """Dispatch several already-ASSIGNED tasks to their workers in parallel.

    Each task runs on its own thread so the master isn't blocked waiting on
    worker 1's TASK_RESULT before it can even send worker 2 its task.
    """
    responses: list[dict | None] = [None] * len(tasks)

    def run(index: int, task) -> None:
        responses[index] = dispatch_assigned_task(connections, task)

    threads = [threading.Thread(target=run, args=(i, task)) for i, task in enumerate(tasks)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    return responses


def submit_and_dispatch_task(
    connections: dict[str, Connection],
    task_id: str,
    task_type: str,
    task_payload: dict,
) -> dict:
    """Submit a task and, if a worker is free right now, assign and dispatch it."""
    with _state_lock:
        scheduler.submit_task(task_id, task_type, task_payload)
        task = scheduler.assign_next_pending_task()

    if task is None:
        return {"status": "pending", "task_id": task_id}

    return dispatch_assigned_task(connections, task)


def drain_pending_tasks(connections: dict[str, Connection]) -> list[dict]:
    """Repeatedly assign and dispatch PENDING tasks to IDLE workers until none remain.

    Each round assigns one task per currently-IDLE worker (a pure in-memory
    step, no network I/O), then dispatches that whole round concurrently —
    one thread per task — instead of waiting for each worker in turn.
    """
    responses = []

    while True:
        batch = []
        with _state_lock:
            while True:
                task = scheduler.assign_next_pending_task()
                if task is None:
                    break
                batch.append(task)

        if not batch:
            break

        responses.extend(dispatch_concurrently(connections, batch))

    return responses


def main():
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((HOST, PORT))
    server_sock.listen(EXPECTED_WORKERS)

    heartbeat_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    heartbeat_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    heartbeat_sock.bind((HOST, HEARTBEAT_PORT))
    heartbeat_sock.listen(EXPECTED_WORKERS)

    print(f"Master started on {HOST}:{PORT} (heartbeats on {HEARTBEAT_PORT})")

    stop_event = threading.Event()
    heartbeat_thread = threading.Thread(target=heartbeat_listener, args=(heartbeat_sock, stop_event), daemon=True)
    monitor_thread = threading.Thread(target=failure_monitor, args=(stop_event,), daemon=True)
    heartbeat_thread.start()
    monitor_thread.start()

    connections: dict[str, Connection] = {}
    for _ in range(EXPECTED_WORKERS):
        conn, worker_id = accept_and_register(server_sock)
        connections[worker_id] = conn

    demo_tasks = [
        ("task-1", "ADD", {"a": 10, "b": 20}),
        ("task-2", "MULTIPLY", {"a": 3, "b": 4}),
        ("task-3", "ADD", {"a": 5, "b": 5}),
        ("task-4", "MULTIPLY", {"a": 2, "b": 2}),
    ]

    try:
        # Submit everything up front so assignment picks whichever workers are
        # IDLE right now (task-1/2 -> worker-1/2), leaving task-3/4 PENDING.
        for task_id, task_type, task_payload in demo_tasks:
            scheduler.submit_task(task_id, task_type, task_payload)

        # Dispatches task-1/2, then as each finishes its worker frees up and
        # this drains task-3/4 onto it too.
        drain_pending_tasks(connections)

        print("Final task states:")
        for task in scheduler.get_all_tasks():
            print(f"  {task.task_id}: {task.status} (worker={task.assigned_worker_id})")
    finally:
        stop_event.set()
        heartbeat_thread.join(timeout=2)
        monitor_thread.join(timeout=2)
        for conn in connections.values():
            conn.close()
        server_sock.close()
        heartbeat_sock.close()


if __name__ == "__main__":
    main()
