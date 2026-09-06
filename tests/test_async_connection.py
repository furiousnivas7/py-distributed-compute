"""Tests for AsyncConnection. No pytest-asyncio dependency: each test drives
its own coroutine with asyncio.run() to keep the project's test setup simple.
"""

import asyncio

import pytest

from rpc.async_connection import AsyncConnection


async def _echo_once(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    conn = AsyncConnection(reader, writer)
    try:
        data = await conn.recv_bytes()
        await conn.send_bytes(data)
    finally:
        await conn.close()


async def _round_trip(message: bytes) -> bytes:
    server = await asyncio.start_server(_echo_once, "127.0.0.1", 0)
    host, port = server.sockets[0].getsockname()[:2]

    async with server:
        reader, writer = await asyncio.open_connection(host, port)
        conn = AsyncConnection(reader, writer)
        try:
            await conn.send_bytes(message)
            return await conn.recv_bytes()
        finally:
            await conn.close()


def test_send_and_recv_round_trip():
    result = asyncio.run(_round_trip(b"PING"))
    assert result == b"PING"


def test_recv_empty_message():
    result = asyncio.run(_round_trip(b""))
    assert result == b""


def test_recv_large_message():
    payload = b"x" * 100_000
    result = asyncio.run(_round_trip(payload))
    assert result == payload


def test_recv_raises_connection_error_when_peer_closes_before_replying():
    async def close_immediately(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        writer.close()
        await writer.wait_closed()

    async def scenario() -> None:
        server = await asyncio.start_server(close_immediately, "127.0.0.1", 0)
        host, port = server.sockets[0].getsockname()[:2]

        async with server:
            reader, writer = await asyncio.open_connection(host, port)
            conn = AsyncConnection(reader, writer)
            try:
                with pytest.raises(ConnectionError):
                    await conn.recv_bytes()
            finally:
                await conn.close()

    asyncio.run(scenario())


def test_recv_raises_connection_error_on_partial_message():
    """Peer sends a length prefix promising more bytes than it actually sends."""

    async def send_partial_then_close(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        import struct

        writer.write(struct.pack("!I", 100) + b"short")
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    async def scenario() -> None:
        server = await asyncio.start_server(send_partial_then_close, "127.0.0.1", 0)
        host, port = server.sockets[0].getsockname()[:2]

        async with server:
            reader, writer = await asyncio.open_connection(host, port)
            conn = AsyncConnection(reader, writer)
            try:
                with pytest.raises(ConnectionError):
                    await conn.recv_bytes()
            finally:
                await conn.close()

    asyncio.run(scenario())


def test_multiple_messages_over_one_connection():
    async def echo_three(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        conn = AsyncConnection(reader, writer)
        try:
            for _ in range(3):
                data = await conn.recv_bytes()
                await conn.send_bytes(data)
        finally:
            await conn.close()

    async def scenario() -> list[bytes]:
        server = await asyncio.start_server(echo_three, "127.0.0.1", 0)
        host, port = server.sockets[0].getsockname()[:2]

        async with server:
            reader, writer = await asyncio.open_connection(host, port)
            conn = AsyncConnection(reader, writer)
            try:
                results = []
                for msg in (b"one", b"two", b"three"):
                    await conn.send_bytes(msg)
                    results.append(await conn.recv_bytes())
                return results
            finally:
                await conn.close()

    assert asyncio.run(scenario()) == [b"one", b"two", b"three"]
