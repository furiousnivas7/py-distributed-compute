"""Minimal TCP master server for Phase 1: accepts one worker, exchanges a PING/ACK."""

import socket

from rpc.connection import Connection

HOST = "127.0.0.1"
PORT = 5000


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
        message = conn.recv_bytes()
        print(f"Received: {message.decode()}")

        response = b"ACK"
        conn.send_bytes(response)
        print(f"Sent: {response.decode()}")
    finally:
        conn.close()
        server_sock.close()


if __name__ == "__main__":
    main()
