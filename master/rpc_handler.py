"""Processes incoming RPC requests and produces RPC responses."""

from common.models import Worker
from rpc import protocol
from rpc.protocol import build_message

# In-memory worker registry: worker_id -> Worker
registered_workers: dict[str, Worker] = {}


def handle_ping(request: dict) -> dict:
    return build_message(protocol.PONG, request["request_id"], {"status": "success"})


def handle_register(request: dict) -> dict:
    payload = request["payload"]
    worker_id = payload.get("worker_id")
    host = payload.get("host")
    port = payload.get("port")

    if not worker_id or not host or not port:
        return build_error(request["request_id"], "INVALID_PAYLOAD", "worker_id, host, and port are required")

    registered_workers[worker_id] = Worker(worker_id=worker_id, host=host, port=port)
    print(f"Worker registered: {worker_id}")

    return build_message(
        protocol.REGISTER_ACK,
        request["request_id"],
        {"status": "success", "message": "Worker registered"},
    )


def build_error(request_id: str, code: str, message: str) -> dict:
    return build_message(protocol.ERROR, request_id, {"code": code, "message": message})


HANDLERS = {
    protocol.PING: handle_ping,
    protocol.REGISTER: handle_register,
}


def handle_request(request: dict) -> dict:
    handler = HANDLERS.get(request["type"])
    if handler is None:
        return build_error(request["request_id"], "UNKNOWN_COMMAND", f"Unsupported RPC type: {request['type']}")
    return handler(request)
