"""Async counterpart to rpc.connection.Connection.

Same length-prefixed framing (4-byte big-endian length + payload), but over
asyncio streams instead of a blocking socket. Kept as a separate class so the
existing threaded Connection keeps working untouched while the async
transport is built and proven alongside it.
"""

import asyncio
import struct

LENGTH_PREFIX_SIZE = 4


class AsyncConnection:
    """Wraps an asyncio (StreamReader, StreamWriter) pair with length-prefixed framing."""

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        self.reader = reader
        self.writer = writer

    async def send_bytes(self, data: bytes) -> None:
        header = struct.pack("!I", len(data))
        self.writer.write(header + data)
        await self.writer.drain()

    async def recv_bytes(self) -> bytes:
        header = await self._recv_exact(LENGTH_PREFIX_SIZE)
        (length,) = struct.unpack("!I", header)
        return await self._recv_exact(length)

    async def _recv_exact(self, num_bytes: int) -> bytes:
        # StreamReader.readexactly already loops internally until it has
        # num_bytes or hits EOF; it just raises IncompleteReadError instead
        # of the socket-level ConnectionError the sync Connection raises.
        # Normalizing to ConnectionError keeps both classes' callers able to
        # catch the same exception when the peer disappears mid-message.
        try:
            return await self.reader.readexactly(num_bytes)
        except asyncio.IncompleteReadError as exc:
            raise ConnectionError("Socket closed before expected data was received") from exc

    async def close(self) -> None:
        self.writer.close()
        await self.writer.wait_closed()
