"""Async TCP worker client: registers with the master, then executes TASK
requests and reports liveness, all on one persistent connection.

Unlike the threaded worker (which opens a second, short-lived connection
per heartbeat specifically to avoid two OS threads reading the same socket),
this worker sends HEARTBEAT as a fire-and-forget message on the SAME
connection its TASK/TASK_RESULT traffic uses. That's safe here because
serve_tasks() is the only coroutine that ever reads this connection -- the
heartbeat loop only ever writes to it, and asyncio guarantees only one
coroutine runs Python code at a time, so concurrent writes from two
coroutines can never interleave mid-message the way two OS threads could.
"""

import asyncio

from rpc import protocol
from rpc.async_connection import AsyncConnection
from rpc.async_rpc import new_request_id, receive_message, send_message, send_request
from rpc.protocol import build_message
from worker.executor import execute_task

MASTER_HOST = "127.0.0.1"
MASTER_PORT = 5000

WORKER_ID = "worker-1"
WORKER_HOST = "127.0.0.1"
WORKER_PORT = 6001

# Must stay well below master.async_server.HEARTBEAT_TIMEOUT (5.0s) -- see
# the same note on worker.worker.HEARTBEAT_INTERVAL_SECONDS.
HEARTBEAT_INTERVAL_SECONDS = 1.5


async def register(conn: AsyncConnection, worker_id: str, worker_host: str, worker_port: int) -> dict:
    return await send_request(
        conn, protocol.REGISTER, {"worker_id": worker_id, "host": worker_host, "port": worker_port}
    )


async def send_heartbeat(conn: AsyncConnection, worker_id: str) -> None:
    """Send a HEARTBEAT without waiting for its ack.

    serve_tasks()'s read loop owns this connection's only read; it'll see
    the HEARTBEAT_ACK show up as just another incoming message and ignore
    it (nothing here needs to correlate a heartbeat with its reply).
    """
    request = build_message(protocol.HEARTBEAT, new_request_id(), {"worker_id": worker_id})
    await send_message(conn, request)


async def start_heartbeat_loop(
    conn: AsyncConnection,
    worker_id: str,
    stop_event: asyncio.Event,
    interval: float = HEARTBEAT_INTERVAL_SECONDS,
) -> None:
    """Send a HEARTBEAT every `interval` seconds until stop_event is set."""
    while True:
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
            return
        except asyncio.TimeoutError:
            pass

        try:
            await send_heartbeat(conn, worker_id)
        except (ConnectionError, OSError):
            return


async def serve_tasks(conn: AsyncConnection) -> None:
    """Read every message on this connection until it closes.

    Executes any TASK it receives and replies with TASK_RESULT; silently
    ignores anything else it doesn't recognize (e.g. the HEARTBEAT_ACK for
    a heartbeat it sent, since those are fire-and-forget here).
    """
    while True:
        try:
            message = await receive_message(conn)
        except ConnectionError:
            return

        if message["type"] != protocol.TASK:
            continue

        task_payload = message["payload"]
        task_id = task_payload["task_id"]
        task_type = task_payload["task_type"]
        task_args = task_payload["task_payload"]
        attempt = task_payload.get("attempt", 1)

        result = execute_task(task_type, task_args)
        print(f"Executed task {task_id} (attempt {attempt}): {result}")

        response = build_message(
            protocol.TASK_RESULT,
            message["request_id"],
            {"task_id": task_id, "attempt": attempt, **result},
        )
        await send_message(conn, response)


async def run_worker(
    master_host: str,
    master_port: int,
    worker_id: str = WORKER_ID,
    worker_host: str = WORKER_HOST,
    worker_port: int = WORKER_PORT,
    heartbeat_interval: float = HEARTBEAT_INTERVAL_SECONDS,
) -> None:
    reader, writer = await asyncio.open_connection(master_host, master_port)
    conn = AsyncConnection(reader, writer)
    print("Connected to master")

    stop_heartbeat = asyncio.Event()
    heartbeat_task = None

    try:
        ping_response = await send_request(conn, protocol.PING)
        print(f"Status: {ping_response['payload'].get('status')}")

        register_response = await register(conn, worker_id, worker_host, worker_port)
        print(f"Status: {register_response['payload'].get('status')}")

        heartbeat_task = asyncio.create_task(
            start_heartbeat_loop(conn, worker_id, stop_heartbeat, heartbeat_interval)
        )

        await serve_tasks(conn)
    finally:
        stop_heartbeat.set()
        if heartbeat_task is not None:
            await heartbeat_task
        await conn.close()


def main() -> None:
    asyncio.run(run_worker(MASTER_HOST, MASTER_PORT))


if __name__ == "__main__":
    main()
