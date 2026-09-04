"""TCP connection wrapper with length-prefixed message framing."""

import socket
import struct

LENGTH_PREFIX_SIZE = 4


class Connection:
    """Wraps a connected socket and frames messages with a 4-byte length prefix."""

    def __init__(self, sock: socket.socket):
        self.sock = sock

    def send_bytes(self, data: bytes) -> None:
        header = struct.pack("!I", len(data))
        self.sock.sendall(header + data)

    def recv_bytes(self) -> bytes:
        header = self._recv_exact(LENGTH_PREFIX_SIZE)
        (length,) = struct.unpack("!I", header)
        return self._recv_exact(length)

    def _recv_exact(self, num_bytes: int) -> bytes:
        chunks = []
        remaining = num_bytes
        while remaining > 0:
            chunk = self.sock.recv(remaining)
            if not chunk:
                raise ConnectionError("Socket closed before expected data was received")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def close(self) -> None:
        self.sock.close()
