"""TCP worker client: connects to the master and exchanges RPC requests/responses."""

import socket
import uuid

from rpc import protocol
from rpc.connection import Connection
from rpc.protocol import build_message

MASTER_HOST = "127.0.0.1"
MASTER_PORT = 5000

WORKER_ID = "worker-1"
WORKER_HOST = "127.0.0.1"
WORKER_PORT = 6001


def new_request_id() -> str:
    return f"req-{uuid.uuid4().hex[:8]}"


def send_rpc(conn: Connection, msg_type: str, payload: dict | None = None) -> dict:
    request = build_message(msg_type, new_request_id(), payload)
    print(f"Sent RPC: {msg_type}")
    conn.send_bytes(protocol.encode_message(request))

    response = protocol.decode_message(conn.recv_bytes())
    print(f"Received RPC: {response['type']}")
    return response


def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((MASTER_HOST, MASTER_PORT))
    print("Connected to master")
    conn = Connection(sock)

    try:
        ping_response = send_rpc(conn, protocol.PING)
        print(f"Status: {ping_response['payload'].get('status')}")

        register_response = send_rpc(
            conn,
            protocol.REGISTER,
            {"worker_id": WORKER_ID, "host": WORKER_HOST, "port": WORKER_PORT},
        )
        print(f"Status: {register_response['payload'].get('status')}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
