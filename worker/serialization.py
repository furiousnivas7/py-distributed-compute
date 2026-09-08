"""Phase 10.3: callable serialization -- one, isolated boundary between
"code as data" (an arbitrary Python callable submitted for a worker
process to execute) and the rest of this codebase. Every dependency on
cloudpickle lives here and nowhere else: scheduler, master, RPC, and even
worker/executor.py only ever call serialize_callable/deserialize_callable
(or the wire-encoding helpers below) -- never cloudpickle directly. That
isolation is what lets the serializer itself (cloudpickle -> dill, a
restricted/sandboxed alternative, a version bump) change later without
touching any of those.

cloudpickle over stdlib pickle specifically because it can serialize
closures, lambdas, and functions defined interactively (a REPL, a
notebook, a __main__ script) -- exactly the kind of callable a user would
actually want to submit ad-hoc. Plain pickle only serializes a function
BY REFERENCE (its module + qualified name, re-imported on the other end),
which fails outright for anything not already importable by that name on
the worker -- precisely the case Phase 10.2's registered-function path
already covers. Serialized-callable submission exists for what that path
can't: a function whose code has never been separately registered
anywhere the worker can import it from.

TRUST BOUNDARY, stated explicitly rather than left implicit: deserializing
and calling a submitted callable is, by design, remote code execution --
a worker that accepts one is trusting whatever produced it (the master,
and transitively anyone who could submit a task) as much as it would
trust its own source code. That's an acceptable, deliberate trade-off for
a cooperative, trusted compute cluster (this project's stated scope
throughout), but it is NOT something to add sandboxing/restriction
theater around here -- a determined sender can always craft a callable
that does whatever the worker process itself is permitted to do. If a
future deployment needs to run untrusted submitters' code, that requires
actual process/OS-level isolation of the worker, not a check in this
module.
"""

import base64

import cloudpickle


class SerializationError(Exception):
    """Raised when a callable can't be serialized, deserialized, or
    doesn't survive the wire-encoding round trip."""


def serialize_callable(fn) -> bytes:
    """Serialize a callable to bytes. Raises SerializationError for a
    non-callable input or anything cloudpickle itself can't handle (e.g.
    a closure over an unpicklable object like an open socket)."""
    if not callable(fn):
        raise SerializationError("fn must be callable")
    try:
        return cloudpickle.dumps(fn)
    except Exception as exc:
        raise SerializationError(f"Failed to serialize callable: {exc}") from exc


def deserialize_callable(data: bytes):
    """Inverse of serialize_callable. Raises SerializationError for
    malformed bytes, or if whatever they decode to isn't callable after
    all (a submitter serialized the wrong kind of object)."""
    if not isinstance(data, (bytes, bytearray)):
        raise SerializationError("data must be bytes")
    try:
        fn = cloudpickle.loads(bytes(data))
    except Exception as exc:
        raise SerializationError(f"Failed to deserialize callable: {exc}") from exc
    if not callable(fn):
        raise SerializationError("Deserialized object is not callable")
    return fn


def encode_for_wire(data: bytes) -> str:
    """base64-encode serialized bytes for JSON transport -- rpc/protocol.py's
    wire format is JSON, which has no native bytes type. Kept in this
    module (not duplicated at every call site) for the same isolation
    reason as serialize_callable/deserialize_callable themselves."""
    if not isinstance(data, (bytes, bytearray)):
        raise SerializationError("data must be bytes")
    return base64.b64encode(data).decode("ascii")


def decode_from_wire(text: str) -> bytes:
    """Inverse of encode_for_wire."""
    if not isinstance(text, str) or not text:
        raise SerializationError("text must be a non-empty string")
    try:
        return base64.b64decode(text.encode("ascii"), validate=True)
    except Exception as exc:
        raise SerializationError(f"Failed to decode wire-format callable: {exc}") from exc
