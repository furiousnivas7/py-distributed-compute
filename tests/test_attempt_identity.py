"""Phase 9.1: explicit (task_id, attempt) execution-attempt identity.

Before this phase, WorkerLink.send_task's wire request_id was derived from
task_id alone (f"task-{task_id}"). That was safe under the CURRENT worker
lifecycle rules -- a worker that ever fails (heartbeat timeout or
connection death) is marked FAILED permanently and excluded from future
assignment (see master.worker_manager.assign_task's IDLE-only scan, and
test_async_fault_tolerance.py's test_reconnection_with_same_worker_id_after_failure_is_currently_rejected),
so in practice a single worker_id is never mid-flight on two attempts of
the same task_id at once. But that safety was a property of the CURRENT
scheduler/worker-manager rules, not of WorkerLink itself -- nothing in
WorkerLink.resolve() would have stopped a stale attempt-N reply from
resolving attempt-(N+1)'s pending Future if a future feature (worker
recovery, speculative execution, retry-without-marking-FAILED) ever let
the same worker_id legitimately hold two attempts of one task_id.

Phase 9.1 makes (task_id, attempt) -- not task_id alone -- the explicit
wire identity of an execution attempt (see attempt_request_id in
master/async_server.py), so this invariant holds by construction rather
than depending on scheduler/worker-manager behavior elsewhere never
changing. These tests exercise WorkerLink directly (bypassing the
scheduler) specifically so they keep testing the mechanism itself even
though the full stack can't currently produce this scenario end-to-end.
"""

import asyncio

import pytest

from master import async_server, rpc_handler
from master.async_server import attempt_request_id
from rpc import protocol
from rpc.async_connection import AsyncConnection
from rpc.async_rpc import receive_message, send_message, send_request
from rpc.protocol import build_message


@pytest.fixture(autouse=True)
def reset_async_master_state():
    rpc_handler.worker_manager.clear()
    async_server.scheduler.clear()
    async_server.connections.clear()
    yield


async def start_master_server():
    server = await asyncio.start_server(async_server.handle_worker_connection, "127.0.0.1", 0)
    host, port = server.sockets[0].getsockname()[:2]
    return server, host, port


async def connect_manual_worker(host: str, port: int, worker_id: str, task_queue: asyncio.Queue) -> AsyncConnection:
    """Registers a real connection and forwards every TASK it receives onto
    `task_queue`, WITHOUT auto-replying -- the test drives replies itself
    (choosing the request_id and payload explicitly), so it can control
    reply order and content precisely instead of behaving like a normal
    worker."""
    reader, writer = await asyncio.open_connection(host, port)
    conn = AsyncConnection(reader, writer)
    await send_request(conn, protocol.PING)
    await send_request(conn, protocol.REGISTER, {"worker_id": worker_id, "host": "127.0.0.1", "port": 6000})

    async def forward_tasks():
        while True:
            try:
                message = await receive_message(conn)
            except ConnectionError:
                return
            if message["type"] == protocol.TASK:
                await task_queue.put(message)

    asyncio.create_task(forward_tasks())
    return conn


def test_attempt_request_id_is_distinct_per_attempt_of_the_same_task():
    assert attempt_request_id("t1", 1) != attempt_request_id("t1", 2)
    assert attempt_request_id("t1", 1) != attempt_request_id("t2", 1)
    assert attempt_request_id("t1", 1) == attempt_request_id("t1", 1)


def test_worker_link_correlates_replies_by_attempt_not_just_task_id():
    """Two attempts of the SAME task_id are made concurrently in-flight on
    ONE WorkerLink (bypassing the scheduler, which wouldn't currently let
    this happen -- see module docstring). Replying to the SECOND attempt
    FIRST, out of send order, must resolve only attempt 2's Future; the
    first attempt's Future must remain untouched until its own matching
    reply arrives, and must then get its OWN result, not attempt 2's."""

    async def scenario():
        server, host, port = await start_master_server()
        task_queue: asyncio.Queue = asyncio.Queue()
        conn = await connect_manual_worker(host, port, "worker-1", task_queue)
        try:
            await async_server.wait_for_workers(1)
            link = async_server.connections["worker-1"]

            future1 = asyncio.create_task(link.send_task("shared-task", "ADD", {"a": 1, "b": 1}, attempt=1))
            msg1 = await asyncio.wait_for(task_queue.get(), timeout=5)
            assert msg1["request_id"] == attempt_request_id("shared-task", 1)
            assert msg1["payload"]["attempt"] == 1

            future2 = asyncio.create_task(link.send_task("shared-task", "ADD", {"a": 1, "b": 1}, attempt=2))
            msg2 = await asyncio.wait_for(task_queue.get(), timeout=5)
            assert msg2["request_id"] == attempt_request_id("shared-task", 2)
            assert msg2["payload"]["attempt"] == 2
            assert msg1["request_id"] != msg2["request_id"]

            # Reply to attempt 2 FIRST -- out of order relative to when the
            # attempts were sent -- to prove routing is by request_id, not
            # by which send_task() call happened to go first.
            await send_message(
                conn,
                build_message(
                    protocol.TASK_RESULT,
                    msg2["request_id"],
                    {"task_id": "shared-task", "attempt": 2, "status": "success", "result": 222},
                ),
            )
            response2 = await asyncio.wait_for(future2, timeout=5)
            assert response2["payload"]["result"] == 222
            assert not future1.done(), "attempt 2's reply must not have resolved attempt 1's Future"

            await send_message(
                conn,
                build_message(
                    protocol.TASK_RESULT,
                    msg1["request_id"],
                    {"task_id": "shared-task", "attempt": 1, "status": "success", "result": 111},
                ),
            )
            response1 = await asyncio.wait_for(future1, timeout=5)
            return response1, response2
        finally:
            await conn.close()
            server.close()
            await server.wait_closed()

    response1, response2 = asyncio.run(scenario())
    assert response1["payload"]["result"] == 111
    assert response2["payload"]["result"] == 222


def test_unmatched_attempt_reply_is_ignored_without_corrupting_the_connection():
    """A TASK_RESULT whose request_id doesn't match any currently-pending
    attempt (e.g. a duplicate or very-late reply for an attempt whose
    Future was already resolved and popped) must not raise, must not
    resolve some unrelated pending Future, and must leave the connection
    usable -- handle_worker_connection's fallback to rpc_handler.handle_request
    already replies UNKNOWN_COMMAND to it rather than crashing; this proves
    that in practice, including that the SAME connection keeps working
    for a real, still-pending attempt afterward."""

    async def scenario():
        server, host, port = await start_master_server()
        task_queue: asyncio.Queue = asyncio.Queue()
        conn = await connect_manual_worker(host, port, "worker-1", task_queue)
        try:
            await async_server.wait_for_workers(1)
            link = async_server.connections["worker-1"]

            real_future = asyncio.create_task(link.send_task("real-task", "ADD", {"a": 2, "b": 3}, attempt=1))
            await asyncio.wait_for(task_queue.get(), timeout=5)

            # A reply for an attempt that was never actually dispatched
            # (task_id "ghost-task" was never submitted) -- request_id
            # can't match anything in link._pending.
            await send_message(
                conn,
                build_message(
                    protocol.TASK_RESULT,
                    attempt_request_id("ghost-task", 1),
                    {"task_id": "ghost-task", "attempt": 1, "status": "success", "result": 999},
                ),
            )

            # The connection must still be alive and the real attempt must
            # still be able to complete normally afterward.
            await send_message(
                conn,
                build_message(
                    protocol.TASK_RESULT,
                    attempt_request_id("real-task", 1),
                    {"task_id": "real-task", "attempt": 1, "status": "success", "result": 5},
                ),
            )
            response = await asyncio.wait_for(real_future, timeout=5)
            return response
        finally:
            await conn.close()
            server.close()
            await server.wait_closed()

    response = asyncio.run(scenario())
    assert response["payload"]["result"] == 5
