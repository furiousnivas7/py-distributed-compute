"""TCP master server: registers a worker, then dispatches a task to it over RPC."""

import socket

from master import rpc_handler
from rpc import protocol
from rpc.connection import Connection
from rpc.protocol import ProtocolError, build_message

HOST = "127.0.0.1"
PORT = 5000


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


def main():
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((HOST, PORT))
    server_sock.listen(1)
    print(f"Master started on {HOST}:{PORT}")

    conn, worker_id = accept_and_register(server_sock)

    try:
        result = dispatch_task(conn, "task-1", "ADD", {"a": 10, "b": 20})
        print(f"Task result: {result['payload']}")
    finally:
        conn.close()
        server_sock.close()


if __name__ == "__main__":
    main()
