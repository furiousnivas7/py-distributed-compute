"""Async RPC primitives built on AsyncConnection.

Reuses rpc.protocol as-is for encoding/decoding/validation -- the JSON
envelope (type/request_id/payload) is transport-independent, so nothing
about it needs to change for asyncio. This module only adds the async
send/receive plumbing around it. Dispatching a received request to a
handler (e.g. master.rpc_handler.handle_request) is intentionally left to
the caller -- that's server-role wiring that belongs to the async master
(Phase 7.3), not to this transport-level module.
"""

import uuid

from rpc import protocol
from rpc.async_connection import AsyncConnection
from rpc.protocol import build_message


def new_request_id() -> str:
    return f"req-{uuid.uuid4().hex[:8]}"


async def send_message(conn: AsyncConnection, message: dict) -> None:
    await conn.send_bytes(protocol.encode_message(message))


async def receive_message(conn: AsyncConnection) -> dict:
    """Receive and decode one RPC message. Raises ProtocolError on malformed input."""
    raw = await conn.recv_bytes()
    return protocol.decode_message(raw)


async def send_request(conn: AsyncConnection, msg_type: str, payload: dict | None = None) -> dict:
    """Build and send an RPC request, then await and decode its response (client role)."""
    request = build_message(msg_type, new_request_id(), payload)
    await send_message(conn, request)
    return await receive_message(conn)
