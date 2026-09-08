"""Phase 11.1: execution-backend abstraction.

Separates WHAT gets executed (a task_type + payload -- the existing
execution contract: built-in operations, registered functions, serialized
callables, all of Phase 10's ExecutionErrorCode taxonomy) from HOW it gets
executed (today: a direct in-process call; Phase 11.2: a multiprocessing
child process; potentially threading/remote/GPU backends later, per the
same interface). The wire protocol and task contract stay exactly as they
are regardless of which backend a worker uses -- a task still just means
"execute this operation with these arguments," and every backend must
produce the exact same result shape worker.executor.execute_task already
produces:

    success: {"status": "success", "result": ...}
    failure: {"status": "error", "code": ..., "message": ...}

so serve_tasks() (and anything testing it) never needs to know or care
which backend actually ran a given task.

`execute` is async, not sync, even though DirectBackend's implementation
is a plain blocking call underneath -- this is deliberate, not premature
abstraction: a backend that hands work to a process pool (Phase 11.2)
needs to be able to `await` that work without blocking the worker's own
asyncio event loop (and therefore its heartbeat loop, and its ability to
notice a SHUTDOWN request) for however long the task takes to run in the
child process. Defining the interface as async now means Phase 11.2 can
implement it correctly without a breaking interface change later.
"""

from abc import ABC, abstractmethod

from worker.executor import execute_task


class ExecutionBackend(ABC):
    """How a worker actually runs a task_type + payload.

    start()/stop() bracket the backend's lifetime (e.g. spin up and tear
    down a process pool) -- called once each by run_worker(), around
    serve_tasks(). Both are no-ops by default; DirectBackend needs
    neither. Phase 11.4 will integrate stop() with the existing graceful
    worker shutdown lifecycle (Phase 9.2.3) so a pool is torn down
    cleanly rather than leaving child processes behind.
    """

    @abstractmethod
    async def execute(self, task_type: str, payload: dict) -> dict:
        """Run one task and return its result dict. Must never raise --
        same contract as worker.executor.execute_task: any failure comes
        back as {"status": "error", "code": ..., "message": ...}, not an
        exception propagating out of this call, so serve_tasks() can
        always send a TASK_RESULT back rather than needing its own
        try/except around every backend."""

    async def start(self) -> None:
        """Called once before this backend serves any tasks."""

    async def stop(self) -> None:
        """Called once when this backend will serve no more tasks."""


class DirectBackend(ExecutionBackend):
    """The baseline backend (Phase 11.1): runs worker.executor.execute_task
    in the SAME process and the SAME coroutine that calls it -- exactly
    what every worker has always done before Phase 11. No process pool,
    no extra machinery, nothing to start or stop. This is what makes
    DirectBackend the safe default that changes nothing about existing
    behavior: it's not a new execution path, just the old one wearing the
    ExecutionBackend interface so callers can swap it out.
    """

    async def execute(self, task_type: str, payload: dict) -> dict:
        return execute_task(task_type, payload)
