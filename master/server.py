"""TCP master server: registers workers, then schedules and dispatches tasks to them."""

import socket

from master import rpc_handler
from master.scheduler import Scheduler
from rpc import protocol
from rpc.connection import Connection
from rpc.protocol import ProtocolError, build_message

HOST = "127.0.0.1"
PORT = 5000
EXPECTED_WORKERS = 2

worker_manager = rpc_handler.worker_manager
scheduler = Scheduler(worker_manager)


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


def dispatch_task(conn: Connection, task_id: str, task_type: str, task_payload: dict) -> dict:
    """Send a TASK request to a worker and return its decoded TASK_RESULT response."""
    request = build_message(
        protocol.TASK,
        request_id=f"task-{task_id}",
        payload={"task_id": task_id, "task_type": task_type, "task_payload": task_payload},
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
    """
    conn = connections[task.assigned_worker_id]

    scheduler.start_task(task.task_id)
    response = dispatch_task(conn, task.task_id, task.task_type, task.payload)

    if response["type"] != protocol.TASK_RESULT:
        scheduler.fail_task(task.task_id)
        return response

    result = response["payload"]

    if result["status"] == "success":
        scheduler.complete_task(task.task_id)
    else:
        scheduler.fail_task(task.task_id)

    return response


def submit_and_dispatch_task(
    connections: dict[str, Connection],
    task_id: str,
    task_type: str,
    task_payload: dict,
) -> dict:
    """Submit a task and, if a worker is free right now, assign and dispatch it."""
    scheduler.submit_task(task_id, task_type, task_payload)

    task = scheduler.assign_next_pending_task()

    if task is None:
        return {"status": "pending", "task_id": task_id}

    return dispatch_assigned_task(connections, task)


def drain_pending_tasks(connections: dict[str, Connection]) -> list[dict]:
    """Repeatedly assign and dispatch PENDING tasks to IDLE workers until none remain.

    Each round assigns one task per currently-IDLE worker (a pure in-memory
    step, no network I/O) before dispatching any of them. Dispatching blocks
    until a TASK_RESULT comes back, so assigning a whole round up front is
    what lets multiple idle workers each get a task instead of whichever
    worker finishes first grabbing every task in the queue.
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

        for task in batch:
            responses.append(dispatch_assigned_task(connections, task))

    return responses


def main():
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((HOST, PORT))
    server_sock.listen(EXPECTED_WORKERS)
    print(f"Master started on {HOST}:{PORT}")

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
        for conn in connections.values():
            conn.close()
        server_sock.close()


if __name__ == "__main__":
    main()
