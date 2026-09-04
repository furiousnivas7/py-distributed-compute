"""TCP master server: accepts a worker connection and serves RPC requests over it."""

import socket

from master import rpc_handler
from rpc import protocol
from rpc.connection import Connection
from rpc.protocol import ProtocolError, build_message

HOST = "127.0.0.1"
PORT = 5000


def serve_connection(conn: Connection) -> None:
    while True:
        try:
            raw = conn.recv_bytes()
        except ConnectionError:
            break

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


def main():
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((HOST, PORT))
    server_sock.listen(1)
    print(f"Master started on {HOST}:{PORT}")

    client_sock, addr = server_sock.accept()
    print("Worker connected")
    conn = Connection(client_sock)

    try:
        serve_connection(conn)
    finally:
        conn.close()
        server_sock.close()


if __name__ == "__main__":
    main()
