"""Phase 10.1: the registered-function execution contract.

The engine distributes named operations (ADD, MULTIPLY, the MAP/REDUCE
vocabularies in worker/executor.py) -- fixed, worker-code vocabularies a
task_type payload selects from. This module extends that same idea to
user code: a CALL task names a function by a string key, and every worker
process looks it up in ITS OWN copy of this registry rather than the
function being shipped over the wire.

That's a deliberate, narrower first step than "submit arbitrary Python
callables" (deferred -- see the module docstring in worker/executor.py's
CALL handler for why): a registered function must already be importable
and registered identically on every worker before a job runs, so there's
no closure/dependency-capture problem to solve here at all -- the
function's code never crosses the wire, only its name and (JSON-safe)
arguments do. Serialized-callable submission (cloudpickle/dill, capturing
closures, shipping code) is real, harder territory -- reintroduces
security concerns (arbitrary code execution from an untrusted master) and
serialization-fidelity concerns (closures, module-level state, C
extensions) that a pre-registered function sidesteps entirely -- and is
intentionally out of scope until this contract is proven.
"""

from typing import Callable

_functions: dict[str, Callable] = {}


class FunctionAlreadyRegisteredError(ValueError):
    """Raised by register_function(..., replace=False) for a name that's
    already registered."""


def register_function(name: str, fn: Callable, *, replace: bool = False) -> None:
    """Register `fn` under `name` for CALL tasks to invoke by that name.

    Re-registering an existing name is rejected by default (`replace`
    must be True) -- silently replacing a name is much more likely to be
    a bug (a second module registering the same name by accident) than
    something a caller actually wants; the explicit flag exists for the
    legitimate cases (test fixtures re-registering between runs, a
    deliberate hot-reload) without making that the default.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("name must be a non-empty string")
    if not callable(fn):
        raise ValueError("fn must be callable")
    if name in _functions and not replace:
        raise FunctionAlreadyRegisteredError(f"Function already registered: {name}")
    _functions[name] = fn


def registered(name: str):
    """Decorator form of register_function: @registered("square")."""

    def decorator(fn: Callable) -> Callable:
        register_function(name, fn)
        return fn

    return decorator


def get_function(name: str) -> Callable | None:
    return _functions.get(name)


def is_registered(name: str) -> bool:
    return name in _functions


def clear() -> None:
    _functions.clear()
