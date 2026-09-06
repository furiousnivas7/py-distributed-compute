"""TCP worker client: registers with the master, then executes TASK requests it sends."""

import socket
import threading
import uuid

from rpc import protocol
from rpc.connection import Connection
from rpc.protocol import build_message
from worker.executor import execute_task

MASTER_HOST = "127.0.0.1"
MASTER_PORT = 5000
MASTER_HEARTBEAT_PORT = 5001

WORKER_ID = "worker-1"
WORKER_HOST = "127.0.0.1"
WORKER_PORT = 6001

# Must stay well below master.server.HEARTBEAT_TIMEOUT (5.0s). A live worker's
# first heartbeat doesn't fire until one full interval after it starts, so an
# interval too close to the timeout can make a healthy worker look FAILED
# just from scheduling/network jitter around that first send.
HEARTBEAT_INTERVAL_SECONDS = 1.5


def new_request_id() -> str:
    return f"req-{uuid.uuid4().hex[:8]}"


def send_rpc(conn: Connection, msg_type: str, payload: dict | None = None) -> dict:
    request = build_message(msg_type, new_request_id(), payload)
    print(f"Sent RPC: {msg_type}")
    conn.send_bytes(protocol.encode_message(request))

    response = protocol.decode_message(conn.recv_bytes())
    print(f"Received RPC: {response['type']}")
    return response


def register(conn: Connection, worker_id: str, worker_host: str, worker_port: int) -> dict:
    return send_rpc(
        conn,
        protocol.REGISTER,
        {"worker_id": worker_id, "host": worker_host, "port": worker_port},
    )


def send_heartbeat(master_host: str, master_port: int, worker_id: str) -> dict:
    """Report liveness on a fresh, short-lived connection, then close it.

    Uses its own connection rather than the long-lived task-dispatch one so
    a periodic heartbeat can never race with the master reading a
    TASK_RESULT reply on that connection.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((master_host, master_port))
    conn = Connection(sock)

    try:
        return send_rpc(conn, protocol.HEARTBEAT, {"worker_id": worker_id})
    finally:
        conn.close()


def start_heartbeat_loop(
    master_host: str,
    master_port: int,
    worker_id: str,
    stop_event: threading.Event,
    interval: float = HEARTBEAT_INTERVAL_SECONDS,
) -> None:
    """Send a HEARTBEAT every `interval` seconds until stop_event is set."""
    while not stop_event.wait(interval):
        try:
            send_heartbeat(master_host, master_port, worker_id)
        except OSError:
            return


def serve_tasks(conn: Connection) -> None:
    """Wait for TASK requests from the master, execute them, and reply with TASK_RESULT."""
    while True:
        try:
            raw = conn.recv_bytes()
        except ConnectionError:
            return

        request = protocol.decode_message(raw)
        print(f"Received RPC: {request['type']}")

        if request["type"] != protocol.TASK:
            continue

        task_payload = request["payload"]
        task_id = task_payload["task_id"]
        task_type = task_payload["task_type"]
        task_args = task_payload["task_payload"]

        result = execute_task(task_type, task_args)
        print(f"Executed task {task_id}: {result}")

        response = build_message(protocol.TASK_RESULT, request["request_id"], {"task_id": task_id, **result})
        conn.send_bytes(protocol.encode_message(response))
        print(f"Sent RPC: {protocol.TASK_RESULT}")


def run_worker(
    master_host: str,
    master_port: int,
    worker_id: str = WORKER_ID,
    worker_host: str = WORKER_HOST,
    worker_port: int = WORKER_PORT,
    heartbeat_port: int | None = None,
) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((master_host, master_port))
    print("Connected to master")
    conn = Connection(sock)

    stop_heartbeat = threading.Event()
    heartbeat_thread = None

    try:
        ping_response = send_rpc(conn, protocol.PING)
        print(f"Status: {ping_response['payload'].get('status')}")

        register_response = register(conn, worker_id, worker_host, worker_port)
        print(f"Status: {register_response['payload'].get('status')}")

        if heartbeat_port is not None:
            heartbeat_thread = threading.Thread(
                target=start_heartbeat_loop,
                args=(master_host, heartbeat_port, worker_id, stop_heartbeat),
                daemon=True,
            )
            heartbeat_thread.start()

        serve_tasks(conn)
    finally:
        stop_heartbeat.set()
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=2)
        conn.close()


def main():
    run_worker(MASTER_HOST, MASTER_PORT, heartbeat_port=MASTER_HEARTBEAT_PORT)


if __name__ == "__main__":
    main()
