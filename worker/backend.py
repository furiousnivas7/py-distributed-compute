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

import asyncio
import multiprocessing
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor

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
        """Run one task and return its result dict.

        Two distinct failure categories, deliberately handled differently
        (Phase 11.2 makes this distinction load-bearing; DirectBackend
        never hits the second one):

        - An EXECUTION failure -- the task ran (somewhere), but failed:
          bad arguments, the function's own exception, a non-serializable
          result. Must NEVER raise for this -- comes back as
          {"status": "error", "code": ..., "message": ...}, exactly
          worker.executor.execute_task's own contract, so serve_tasks()
          can always send a normal TASK_RESULT back. This failure is
          deterministic (see README's "Retry Semantics") and the master
          will not retry it.

        - A BACKEND failure -- the task was never safely completed
          because whatever was running it (e.g. a multiprocessing child
          process) died/became unusable, unrelated to the task's own
          logic. This MAY raise. Letting it propagate out of execute() is
          intentional, not an oversight: serve_tasks() has no special
          handling for it, so it propagates further and tears down this
          worker's connection to the master -- which the master already
          treats exactly like any other worker crash (mark FAILED,
          requeue the task, Phase 8/9's existing connection-death path).
          That reuses the existing retry mechanism instead of inventing a
          second one or teaching the wire protocol a new "please retry
          me" message -- see worker/backend.py's MultiprocessingBackend
          for the concrete case this exists for.
        """

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


class MultiprocessingBackend(ExecutionBackend):
    """Phase 11.2: runs each task in a child process from a
    concurrent.futures.ProcessPoolExecutor, so a CPU-bound task (heavy
    computation inside a registered function or a serialized callable)
    doesn't block this worker's own event loop -- and therefore doesn't
    delay its heartbeat, its ability to notice a SHUTDOWN request, or any
    other task's dispatch:

        CPU-bound task -> child process -> result/exception ->
        MultiprocessingBackend.execute() -> the SAME execution-result
        contract worker.executor.execute_task already produces

    `execute_task` itself runs UNCHANGED, just inside the child process
    instead of this one -- it already never raises for a normal execution
    failure (every failure mode is caught and turned into
    {"status": "error", "code": ..., "message": ...} before it returns),
    so that contract, including every ExecutionErrorCode, survives the
    process boundary completely unchanged. The only NEW failure mode this
    backend introduces is the child process itself dying (crash, OOM
    kill, an unpicklable value that surprises the pool machinery) -- see
    ExecutionBackend.execute's docstring for why that's allowed to raise
    rather than being folded into the normal result contract.

    THE CONTRACT (platform-independent, holds for every mp_context):
    a registered function must be importable/available in whatever
    process actually executes it. Both execution modes satisfy this the
    same way DirectBackend always required it -- registration has to
    have happened, in THAT process, before the lookup -- multiprocessing
    doesn't change the requirement, only how many processes it applies
    to. Serialized callables never depend on this at all: cloudpickle
    ships the callable itself as part of the task payload and decodes it
    explicitly inside the child, so they work identically under any
    mp_context.

    THE IMPLEMENTATION CHOICE this backend makes (Unix-specific, and
    explicit rather than incidental): mp_context defaults to "fork", not
    the platform default ("spawn" on macOS since Python 3.8, and the only
    option on Windows -- fork isn't available there at all). A fork child
    inherits the parent's memory at fork time, including whatever's
    already in worker.registry, which is the SIMPLEST way to satisfy the
    contract above for an ad-hoc runtime register_function() call: no
    extra discipline needed beyond "register before start()". A spawn
    child is instead a fresh interpreter that only re-runs registration
    code that happens to execute again during its own startup/import (a
    module-level @registry.register(...) decorator in a module that gets
    imported, not a one-off call made after the process already forked
    away) -- satisfying the SAME contract under spawn just requires that
    extra bit of import-time discipline instead of fork's "for free"
    inheritance.

    This project's scope has been Unix-oriented throughout (see the
    threaded worker's fork-adjacent assumptions and this codebase's test
    environment), so defaulting to "fork" here is a deliberate, documented
    platform decision, not a portability oversight -- pass
    mp_context="spawn" (or "forkserver") explicitly to run under it
    instead (Windows requires this, since fork doesn't exist there), and
    make sure every registered function is (re-)registered via code that
    naturally gets imported in a fresh interpreter, not a runtime-only
    register_function() call.
    """

    def __init__(self, max_workers: int | None = None, mp_context: str = "fork"):
        self._max_workers = max_workers
        self._mp_context = mp_context
        self._pool: ProcessPoolExecutor | None = None

    async def start(self) -> None:
        self._pool = ProcessPoolExecutor(
            max_workers=self._max_workers,
            mp_context=multiprocessing.get_context(self._mp_context),
        )

    async def stop(self) -> None:
        pool, self._pool = self._pool, None
        if pool is None:
            return
        # shutdown(wait=True) blocks until every child process actually
        # terminates -- run it off the event loop thread so a slow-to-die
        # child doesn't freeze the worker's own async machinery during
        # shutdown either.
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: pool.shutdown(wait=True, cancel_futures=True))

    async def execute(self, task_type: str, payload: dict) -> dict:
        if self._pool is None:
            raise RuntimeError("MultiprocessingBackend.execute() called before start()")

        # run_in_executor returns immediately with an awaitable -- this
        # does NOT block the event loop while the child process runs;
        # other coroutines (the heartbeat loop, a shutdown watcher,
        # serve_tasks() itself for a DIFFERENT connection) keep running
        # normally in the meantime.
        #
        # No try/except here is deliberate: if the child process running
        # (or that would have run) this task dies unexpectedly, awaiting
        # this raises BrokenProcessPool -- not a normal execution
        # failure, so it's allowed to propagate straight out of execute()
        # rather than becoming an error result (see ExecutionBackend.
        # execute's docstring for why). Once broken, a ProcessPoolExecutor
        # never recovers; every subsequent submission fails the same way,
        # so there's nothing to salvage for a later task either.
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._pool, execute_task, task_type, payload)
