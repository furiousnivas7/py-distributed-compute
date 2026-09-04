"""Minimal TCP worker client for Phase 1: connects to master, sends PING, prints ACK."""

import socket

from rpc.connection import Connection

MASTER_HOST = "127.0.0.1"
MASTER_PORT = 5000


def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((MASTER_HOST, MASTER_PORT))
    print("Connected to master")
    conn = Connection(sock)

    try:
        message = b"PING"
        conn.send_bytes(message)
        print(f"Sent: {message.decode()}")

        response = conn.recv_bytes()
        print(f"Received: {response.decode()}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
