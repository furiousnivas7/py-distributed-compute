"""RPC message format: encoding, decoding, and validation."""

import json

# RPC message types
PING = "PING"
PONG = "PONG"
REGISTER = "REGISTER"
REGISTER_ACK = "REGISTER_ACK"
TASK = "TASK"
TASK_RESULT = "TASK_RESULT"
HEARTBEAT = "HEARTBEAT"
HEARTBEAT_ACK = "HEARTBEAT_ACK"
ERROR = "ERROR"

REQUIRED_FIELDS = ("type", "request_id", "payload")


class ProtocolError(ValueError):
    """Raised when a message cannot be encoded/decoded or fails validation."""


def build_message(msg_type: str, request_id: str, payload: dict | None = None) -> dict:
    return {
        "type": msg_type,
        "request_id": request_id,
        "payload": payload or {},
    }


def encode_message(message: dict) -> bytes:
    try:
        return json.dumps(message).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"Failed to encode message: {exc}") from exc


def decode_message(data: bytes) -> dict:
    try:
        message = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Failed to decode message: {exc}") from exc
    validate_message(message)
    return message


def validate_message(message) -> None:
    if not isinstance(message, dict):
        raise ProtocolError("Message must be a JSON object")

    for field in REQUIRED_FIELDS:
        if field not in message:
            raise ProtocolError(f"Missing required field: {field}")

    if not isinstance(message["type"], str) or not message["type"]:
        raise ProtocolError("Field 'type' must be a non-empty string")

    if not isinstance(message["request_id"], str) or not message["request_id"]:
        raise ProtocolError("Field 'request_id' must be a non-empty string")

    if not isinstance(message["payload"], dict):
        raise ProtocolError("Field 'payload' must be an object")
