"""Processes incoming RPC requests and produces RPC responses."""

from common.models import WorkerStatus
from master.worker_manager import DuplicateWorkerError, WorkerManager, WorkerNotFoundError
from rpc import protocol
from rpc.protocol import build_message

worker_manager = WorkerManager()


def handle_ping(request: dict) -> dict:
    return build_message(protocol.PONG, request["request_id"], {"status": "success"})


def handle_register(request: dict) -> dict:
    payload = request["payload"]
    worker_id = payload.get("worker_id")
    host = payload.get("host")
    port = payload.get("port")

    try:
        worker_manager.register_worker(worker_id, host, port)
    except DuplicateWorkerError:
        return build_error(request["request_id"], "DUPLICATE_WORKER", "Worker already registered")
    except ValueError as exc:
        return build_error(request["request_id"], "INVALID_PAYLOAD", str(exc))

    print(f"Worker registered: {worker_id}")

    return build_message(
        protocol.REGISTER_ACK,
        request["request_id"],
        {"status": "success", "message": "Worker registered"},
    )


def handle_heartbeat(request: dict) -> dict:
    payload = request["payload"]
    worker_id = payload.get("worker_id")

    try:
        worker_manager.record_heartbeat(worker_id)
    except WorkerNotFoundError:
        return build_error(request["request_id"], "UNKNOWN_WORKER", f"Unknown worker: {worker_id}")

    return build_message(protocol.HEARTBEAT_ACK, request["request_id"], {"status": "success"})


def handle_shutdown(request: dict) -> dict:
    """A worker asking to stop accepting new work (Phase 9.2.3). Only
    transitions a currently-live worker (REGISTERED/IDLE/BUSY) to
    DRAINING; a worker that's already DRAINING/STOPPED/FAILED just gets
    acknowledged again rather than an error, since a duplicate or
    late-arriving SHUTDOWN isn't a protocol violation. What actually
    happens once a worker is DRAINING (whether its connection can be
    closed immediately or must wait for an in-flight task to finish) is
    connection-level bookkeeping this module doesn't have access to -- see
    master/async_server.py's handle_worker_connection and
    dispatch_assigned_task.
    """
    payload = request["payload"]
    worker_id = payload.get("worker_id")

    worker = worker_manager.get_worker(worker_id)
    if worker is None:
        return build_error(request["request_id"], "UNKNOWN_WORKER", f"Unknown worker: {worker_id}")

    if worker.status in (WorkerStatus.REGISTERED, WorkerStatus.IDLE, WorkerStatus.BUSY):
        worker_manager.update_status(worker_id, WorkerStatus.DRAINING)

    return build_message(protocol.SHUTDOWN_ACK, request["request_id"], {"status": "success"})


def build_error(request_id: str, code: str, message: str) -> dict:
    return build_message(protocol.ERROR, request_id, {"code": code, "message": message})


HANDLERS = {
    protocol.PING: handle_ping,
    protocol.REGISTER: handle_register,
    protocol.HEARTBEAT: handle_heartbeat,
    protocol.SHUTDOWN: handle_shutdown,
}


def handle_request(request: dict) -> dict:
    handler = HANDLERS.get(request["type"])
    if handler is None:
        return build_error(request["request_id"], "UNKNOWN_COMMAND", f"Unsupported RPC type: {request['type']}")
    return handler(request)
