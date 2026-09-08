"""Phase 10.2: the function registry.

The engine distributes named operations (ADD, MULTIPLY, the MAP/REDUCE
vocabularies in worker/executor.py) -- fixed, worker-code vocabularies a
task's operation name (task_type) selects from. This module extends that
same idea to user code, conceptually shared by master and workers: a task
whose operation name isn't one of the built-ins is looked up here by that
same name and invoked with the task's payload as keyword arguments --

    result = registry.get_function(operation)(**payload)

-- see worker/executor.py's execute_task, which does exactly this as a
fallback once HANDLERS (the built-in operations) comes up empty. Payload
must be a dict *because* the calling convention is keyword-only: this is
what keeps a registered function callable straight off the JSON-decoded
wire payload, with no argument-list/positional-vs-keyword translation
layer to design or get wrong.

This is a deliberate, narrower step than "submit arbitrary Python
callables": a registered function must already be importable and
registered identically on every worker before a job runs, so there's no
closure/dependency-capture problem to solve here at all -- the function's
CODE never crosses the wire, only its name and (JSON-safe) keyword
arguments do. That sidesteps arbitrary-code-execution-from-the-master
entirely while still letting users run their own logic distributed, which
is what makes this "avoid immediately introducing arbitrary code
execution" while still being real user-code execution, not just a bigger
fixed vocabulary. Serialized-callable submission (cloudpickle/dill,
capturing closures, shipping code) is real, harder territory -- and is
intentionally deferred until this contract is proven.
"""

from typing import Callable

_functions: dict[str, Callable] = {}


class FunctionAlreadyRegisteredError(ValueError):
    """Raised by register_function(..., replace=False) for a name that's
    already registered."""


def register_function(name: str, fn: Callable, *, replace: bool = False) -> None:
    """Register `fn` under `name` (imperative form of @register).

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


def register(name: str):
    """Decorator form: @register("add") def add(a, b): return a + b."""

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
