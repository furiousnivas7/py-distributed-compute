"""Phase 10.3: the worker/serialization.py abstraction, in isolation from
task execution entirely."""

import pytest

from worker import serialization


def test_serialize_then_deserialize_roundtrips_a_simple_function():
    def add(a, b):
        return a + b

    data = serialization.serialize_callable(add)
    fn = serialization.deserialize_callable(data)
    assert fn(2, 3) == 5


def test_serialize_then_deserialize_roundtrips_a_lambda():
    data = serialization.serialize_callable(lambda x: x * x)
    fn = serialization.deserialize_callable(data)
    assert fn(4) == 16


def test_serialize_captures_a_closure():
    multiplier = 10

    def scaled(x):
        return x * multiplier

    data = serialization.serialize_callable(scaled)
    fn = serialization.deserialize_callable(data)
    assert fn(3) == 30


def test_serialize_non_callable_raises():
    with pytest.raises(serialization.SerializationError):
        serialization.serialize_callable(42)


def test_deserialize_garbage_bytes_raises():
    with pytest.raises(serialization.SerializationError):
        serialization.deserialize_callable(b"not a valid pickle stream")


def test_deserialize_non_bytes_raises():
    with pytest.raises(serialization.SerializationError):
        serialization.deserialize_callable("not bytes")


def test_deserialize_a_non_callable_object_raises():
    data = serialization.serialize_callable(lambda: None)
    # Tamper isn't feasible without breaking the pickle stream outright;
    # instead prove the guard exists by serializing something that
    # ISN'T a function through a workaround: pickle a plain value using
    # the same underlying mechanism serialize_callable uses, bypassing
    # its own callable check, and confirm deserialize_callable still
    # rejects the result.
    import cloudpickle

    non_callable_data = cloudpickle.dumps(42)
    with pytest.raises(serialization.SerializationError):
        serialization.deserialize_callable(non_callable_data)


def test_encode_decode_wire_roundtrip():
    data = serialization.serialize_callable(lambda x: x + 1)
    encoded = serialization.encode_for_wire(data)
    assert isinstance(encoded, str)
    decoded = serialization.decode_from_wire(encoded)
    assert decoded == data
    fn = serialization.deserialize_callable(decoded)
    assert fn(1) == 2


def test_encode_for_wire_rejects_non_bytes():
    with pytest.raises(serialization.SerializationError):
        serialization.encode_for_wire("not bytes")


def test_decode_from_wire_rejects_invalid_base64():
    with pytest.raises(serialization.SerializationError):
        serialization.decode_from_wire("not valid base64!!! ###")


def test_decode_from_wire_rejects_empty_string():
    with pytest.raises(serialization.SerializationError):
        serialization.decode_from_wire("")


def test_serialize_callable_rejects_a_closure_over_an_oversized_object(monkeypatch):
    """Doesn't actually build a 10MB closure -- lowers the limit instead,
    so the test stays fast while still exercising the real guard."""
    monkeypatch.setattr(serialization, "MAX_SERIALIZED_CALLABLE_BYTES", 10)

    def fn():
        return "this closure's pickled form is well over 10 bytes"

    with pytest.raises(serialization.SerializationError, match="exceeding"):
        serialization.serialize_callable(fn)


def test_serialize_callable_under_the_limit_succeeds(monkeypatch):
    monkeypatch.setattr(serialization, "MAX_SERIALIZED_CALLABLE_BYTES", 10_000)
    data = serialization.serialize_callable(lambda x: x + 1)
    fn = serialization.deserialize_callable(data)
    assert fn(1) == 2
