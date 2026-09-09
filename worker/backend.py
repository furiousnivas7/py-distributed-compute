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
    neither. Phase 11.3 integrated stop() (and start()) with the existing
    graceful worker shutdown lifecycle deterministically -- see
    worker/async_worker.py's run_worker for the failure-path handling.

    max_concurrency (Phase 11.4): optional capacity metadata -- the most
    execute() calls this backend intends to actually run at once, or None
    if it doesn't impose one. Purely informational (nothing in this base
    class or serve_tasks() reads or enforces it); a backend that wants an
    actual limit enforces it itself (see MultiprocessingBackend) and
    publishes the number here so a caller/observer can introspect it
    without needing to know which concrete backend it's looking at.
    """

    max_concurrency: int | None = None

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

    def __init__(
        self,
        max_workers: int | None = None,
        mp_context: str = "fork",
        max_in_flight: int | None = None,
    ):
        """`max_workers` (Phase 11.2, validated as of Phase 11.4): the
        process pool's own size -- ProcessPoolExecutor never runs more
        than this many tasks in true parallel. None (the explicit,
        documented default) delegates to ProcessPoolExecutor's own
        default, os.cpu_count() -- not reproduced or hardcoded here, so
        it stays correct if that default's definition ever changes.
        Must be a positive integer otherwise; 0 or negative is rejected
        immediately, at construction time, rather than failing
        confusingly later inside ProcessPoolExecutor's own constructor.

        `max_in_flight` (Phase 11.4): a SEPARATE, optional cap on how
        many execute() calls this backend will have submitted to the
        pool at once -- distinct from max_workers, which only bounds
        parallel EXECUTION. Without max_in_flight, submitting more
        execute() calls than max_workers is still fully correct
        (ProcessPoolExecutor queues the excess internally and runs them
        as workers free up -- already proven in Phase 11.2's concurrent-
        tasks test); the only thing max_in_flight adds is an explicit
        backend-side point where an OVER-limit caller blocks (awaiting a
        semaphore) rather than having every excess submission accepted
        and queued inside the pool's own unbounded internal queue.
        Defaults to None -- unbounded, i.e. today's Phase 11.2 behavior,
        unchanged unless a caller explicitly opts in. If set, must also
        be a positive integer.
        """
        if max_workers is not None and max_workers <= 0:
            raise ValueError(f"max_workers must be a positive integer or None, got {max_workers}")
        if max_in_flight is not None and max_in_flight <= 0:
            raise ValueError(f"max_in_flight must be a positive integer or None, got {max_in_flight}")

        self._max_workers = max_workers
        self._mp_context = mp_context
        self._max_in_flight = max_in_flight
        self._semaphore: asyncio.Semaphore | None = None
        self._pool: ProcessPoolExecutor | None = None
        self.max_concurrency = max_in_flight if max_in_flight is not None else max_workers

    async def start(self) -> None:
        self._pool = ProcessPoolExecutor(
            max_workers=self._max_workers,
            mp_context=multiprocessing.get_context(self._mp_context),
        )
        if self._max_in_flight is not None:
            self._semaphore = asyncio.Semaphore(self._max_in_flight)

    async def stop(self) -> None:
        pool, self._pool = self._pool, None
        self._semaphore = None
        if pool is None:
            return
        # shutdown(wait=True, cancel_futures=True) cancels anything still
        # QUEUED inside the pool (not yet running) and blocks until every
        # child process actually terminates -- run it off the event loop
        # thread so a slow-to-die child doesn't freeze the worker's own
        # async machinery during shutdown either. A task still WAITING on
        # the in-flight semaphore (never got as far as submitting to the
        # pool at all) isn't the pool's concern -- its own execute() call
        # is simply still suspended on `async with self._semaphore`; once
        # this method returns, resetting self._pool/self._semaphore above
        # means that suspended call resumes into a pool-is-None
        # RuntimeError the next time it's scheduled, the same clear
        # failure execute() already raises for any other post-stop call --
        # see _submit()'s own self._pool check, which is what actually
        # makes that true rather than silently falling through to
        # run_in_executor's None-means-"use the default thread pool"
        # behavior.
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: pool.shutdown(wait=True, cancel_futures=True))

    async def execute(self, task_type: str, payload: dict) -> dict:
        if self._pool is None:
            raise RuntimeError("MultiprocessingBackend.execute() called before start()")

        if self._semaphore is None:
            return await self._submit(task_type, payload)

        # Phase 11.4: with max_in_flight configured, a caller beyond the
        # limit waits here (queues) rather than every excess submission
        # being handed straight to the pool's own unbounded internal
        # queue -- "allow tasks to queue instead of being silently
        # dropped" is satisfied by `async with` blocking, not by
        # rejecting anything. Acquired before submission and released in
        # a finally -- guaranteed even if _submit() raises (a normal
        # execution failure returns normally and releases the same way;
        # a BrokenProcessPool propagating still releases before it does,
        # so a permit can never leak just because the pool died).
        async with self._semaphore:
            return await self._submit(task_type, payload)

    async def _submit(self, task_type: str, payload: dict) -> dict:
        # Re-check here, not just in execute(): a call that was waiting on
        # self._semaphore when stop() ran only resumes once whatever
        # currently holds the permit releases it (stop()'s
        # pool.shutdown(wait=True, ...) already waits for that) -- by then
        # self._pool has been reset to None. Without this check,
        # run_in_executor(self._pool, ...) would receive None and silently
        # fall back to Python's own default thread pool executor instead
        # of failing clearly -- exactly the "silently degrades instead of
        # resolving with an explicit error" gap this check closes.
        if self._pool is None:
            raise RuntimeError("MultiprocessingBackend.execute() called before start() (or after stop())")

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
