"""Public API for the worker runtime.

Stable import surface -- deep imports (``worker.backend``,
``worker.registry``, ...) keep working exactly as before; this module only
adds re-exports on top of what already exists.

Only the async worker (Phase 9 onward) is exported here. ``worker/worker.py``
is this project's earlier synchronous/threaded worker, superseded by
``worker/async_worker.py`` but kept around for its own pre-existing tests
-- not part of the public API.

``worker.serialization`` (``serialize_callable``/``deserialize_callable``)
is deliberately NOT re-exported here either: it's an internal building
block for ``jobs.call.submit_serialized_call`` and
``jobs.map.build_map_job``/``jobs.reduce.build_reduce_job`` (via
``ExecutionSpec.serialized``), not something a caller should reach for
directly -- go through those, not raw serialization, to submit a
serialized-callable task (see this project's README "Security
Considerations" section for why that boundary matters).

Typical usage::

    from worker import DirectBackend, MultiprocessingBackend, register_function, run_worker

    register_function("double", lambda x: x * 2)
    await run_worker(master_host, master_port, backend=DirectBackend())
    # or: await run_worker(master_host, master_port,
    #                       backend=MultiprocessingBackend(max_workers=4))
"""

from worker.async_worker import run_worker
from worker.backend import BackendMetrics, DirectBackend, ExecutionBackend, MultiprocessingBackend
from worker.config import BackendConfig, build_backend, resolve_backend_config
from worker.executor import MAP_OPERATIONS, REDUCE_OPERATIONS, ExecutionError, ExecutionErrorCode
from worker.registry import (
    FunctionAlreadyRegisteredError,
    clear as clear_registry,
    get_function,
    is_registered,
    register,
    register_function,
)

__all__ = [
    "run_worker",
    "ExecutionBackend",
    "DirectBackend",
    "MultiprocessingBackend",
    "BackendMetrics",
    "BackendConfig",
    "resolve_backend_config",
    "build_backend",
    "ExecutionError",
    "ExecutionErrorCode",
    "MAP_OPERATIONS",
    "REDUCE_OPERATIONS",
    "register_function",
    "register",
    "get_function",
    "is_registered",
    "clear_registry",
    "FunctionAlreadyRegisteredError",
]
