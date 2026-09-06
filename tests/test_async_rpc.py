"""Tests for the async RPC layer (rpc.async_rpc) built on AsyncConnection.

Each test drives a small async request/response loop that dispatches
through the existing, unmodified master.rpc_handler.handle_request -- proof
that the RPC *logic* didn't need to change for asyncio, only the transport.
No pytest-asyncio dependency: tests drive their coroutine via asyncio.run().
"""

import asyncio

import pytest

from master import rpc_handler
from rpc import protocol
from rpc.async_connection import AsyncConnection
from rpc.async_rpc import new_request_id, receive_message, send_message, send_request
from rpc.protocol import ProtocolError


@pytest.fixture(autouse=True)
def reset_worker_registry():
    """rpc_handler.worker_manager is a process-wide singleton also used by
    the synchronous test suite (test_task_flow.py, test_worker_manager.py),
    since every test file imports the same master.rpc_handler module."""
    rpc_handler.worker_manager.clear()
    yield


async def serve_forever(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Generic async request/response loop: receive, dispatch through the
    existing synchronous rpc_handler, respond, repeat until the peer
    disconnects. Mirrors master.server.accept_and_register's loop body,
    just driven over AsyncConnection instead of the blocking Connection."""
    conn = AsyncConnection(reader, writer)
    try:
        while True:
            try:
                request = await receive_message(conn)
            except ConnectionError:
                return
            except ProtocolError as exc:
                error = protocol.build_message(
                    protocol.ERROR, "unknown", {"code": "INVALID_MESSAGE", "message": str(exc)}
                )
                await send_message(conn, error)
                continue

            response = rpc_handler.handle_request(request)
            await send_message(conn, response)
    finally:
        await conn.close()


async def start_test_server():
    server = await asyncio.start_server(serve_forever, "127.0.0.1", 0)
    host, port = server.sockets[0].getsockname()[:2]
    return server, host, port


def test_ping_pong():
    async def scenario():
        server, host, port = await start_test_server()
        async with server:
            reader, writer = await asyncio.open_connection(host, port)
            conn = AsyncConnection(reader, writer)
            try:
                return await send_request(conn, protocol.PING)
            finally:
                await conn.close()

    response = asyncio.run(scenario())
    assert response["type"] == protocol.PONG
    assert response["payload"]["status"] == "success"


def test_register_ack_and_worker_is_recorded():
    async def scenario():
        server, host, port = await start_test_server()
        async with server:
            reader, writer = await asyncio.open_connection(host, port)
            conn = AsyncConnection(reader, writer)
            try:
                return await send_request(
                    conn, protocol.REGISTER, {"worker_id": "worker-1", "host": "127.0.0.1", "port": 6001}
                )
            finally:
                await conn.close()

    response = asyncio.run(scenario())
    assert response["type"] == protocol.REGISTER_ACK
    assert response["payload"]["status"] == "success"
    assert rpc_handler.worker_manager.get_worker("worker-1") is not None


def test_heartbeat_for_registered_worker():
    async def scenario():
        server, host, port = await start_test_server()
        async with server:
            reader, writer = await asyncio.open_connection(host, port)
            conn = AsyncConnection(reader, writer)
            try:
                await send_request(
                    conn, protocol.REGISTER, {"worker_id": "worker-1", "host": "127.0.0.1", "port": 6001}
                )
                return await send_request(conn, protocol.HEARTBEAT, {"worker_id": "worker-1"})
            finally:
                await conn.close()

    response = asyncio.run(scenario())
    assert response["type"] == protocol.HEARTBEAT_ACK
    assert response["payload"]["status"] == "success"


def test_heartbeat_for_unknown_worker_is_rejected():
    async def scenario():
        server, host, port = await start_test_server()
        async with server:
            reader, writer = await asyncio.open_connection(host, port)
            conn = AsyncConnection(reader, writer)
            try:
                return await send_request(conn, protocol.HEARTBEAT, {"worker_id": "ghost"})
            finally:
                await conn.close()

    response = asyncio.run(scenario())
    assert response["type"] == protocol.ERROR
    assert response["payload"]["code"] == "UNKNOWN_WORKER"


def test_unknown_command_returns_error():
    async def scenario():
        server, host, port = await start_test_server()
        async with server:
            reader, writer = await asyncio.open_connection(host, port)
            conn = AsyncConnection(reader, writer)
            try:
                return await send_request(conn, "BOGUS", {})
            finally:
                await conn.close()

    response = asyncio.run(scenario())
    assert response["type"] == protocol.ERROR
    assert response["payload"]["code"] == "UNKNOWN_COMMAND"


def test_malformed_message_returns_error_without_closing_connection():
    async def scenario():
        server, host, port = await start_test_server()
        async with server:
            reader, writer = await asyncio.open_connection(host, port)
            conn = AsyncConnection(reader, writer)
            try:
                # Missing request_id and payload -- fails validation on decode.
                await send_message(conn, {"type": "PING"})
                error_response = await receive_message(conn)

                # The connection must still be usable after an error reply.
                ok_response = await send_request(conn, protocol.PING)
                return error_response, ok_response
            finally:
                await conn.close()

    error_response, ok_response = asyncio.run(scenario())
    assert error_response["type"] == protocol.ERROR
    assert error_response["payload"]["code"] == "INVALID_MESSAGE"
    assert ok_response["type"] == protocol.PONG


def test_multiple_requests_over_one_connection():
    async def scenario():
        server, host, port = await start_test_server()
        async with server:
            reader, writer = await asyncio.open_connection(host, port)
            conn = AsyncConnection(reader, writer)
            try:
                ping_response = await send_request(conn, protocol.PING)
                register_response = await send_request(
                    conn, protocol.REGISTER, {"worker_id": "worker-1", "host": "127.0.0.1", "port": 6001}
                )
                return ping_response, register_response
            finally:
                await conn.close()

    ping_response, register_response = asyncio.run(scenario())
    assert ping_response["type"] == protocol.PONG
    assert register_response["type"] == protocol.REGISTER_ACK


def test_new_request_id_is_unique():
    ids = {new_request_id() for _ in range(100)}
    assert len(ids) == 100
