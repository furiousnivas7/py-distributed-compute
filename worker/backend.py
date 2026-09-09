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
import logging
import multiprocessing
import time
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace

from worker.executor import execute_task

# Phase 11.6: structured lifecycle/diagnostic logging -- deliberately the
# stdlib logging module, not print(), so a caller can route, filter, or
# silence this independently of the worker's own stdout (which
# async_worker.py still uses print() for a couple of legacy status lines
# outside this phase's scope). Every message here is a single-line
# "event_name key=value ..." record and NEVER includes a task's payload
# contents or a serialized callable's bytes -- only identifiers (task_type,
# execution_mode, backend name, counts, exception TYPE names).
logger = logging.getLogger(__name__)


@dataclass
class BackendMetrics:
    """Phase 11.5: a point-in-time snapshot of a backend's execution
    activity -- purely observational, never read by anything that affects
    scheduling, retries, or the wire protocol. `submitted` counts every
    execute() call accepted; `completed`/`failed` split on the SAME
    success/error distinction execute_task's own result dict already
    makes (never on exceptions); `backend_errors` counts the OTHER
    category, an execute() call that raised (see ExecutionBackend.execute's
    docstring for that split). `total_execution_time`/`total_wait_time`
    are running sums, not the count-derived average -- divide by
    completed+failed (or submitted) yourself if you want a mean.

    Concurrency safety: every counter here is mutated only from coroutine
    code running on the worker's own asyncio event loop -- never from a
    worker thread, and never from the child PROCESS a task actually runs
    in (that process only ever returns a plain result/exception back
    through run_in_executor's own future, which resolves back on the
    event loop). A single OS thread runs Python bytecode at a time under
    asyncio, and no `await` sits between a counter's read and its write
    anywhere in this file, so plain `+=`/`-=` needs no lock. This would
    change if a future backend updated these counters from a real OS
    thread (e.g. a ThreadPoolExecutor-based backend) instead.
    """

    submitted: int = 0
    running: int = 0
    completed: int = 0
    failed: int = 0
    backend_errors: int = 0
    total_execution_time: float = 0.0
    total_wait_time: float = 0.0


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

    def get_metrics(self) -> BackendMetrics:
        """Phase 11.5: a snapshot of this backend's execution activity.

        Non-breaking observability contract -- the default here returns an
        empty/default snapshot, so any existing ExecutionBackend subclass
        (including ones defined outside this file, e.g. in tests) keeps
        working unchanged without needing to implement this at all. A
        backend that wants real numbers overrides it (see DirectBackend,
        MultiprocessingBackend). Never read by anything in this codebase
        that affects scheduling, retries, or the wire protocol.
        """
        return BackendMetrics()

    def describe(self) -> dict[str, object]:
        """Phase 11.6: diagnostic identity/capability metadata about this
        backend -- for logging and troubleshooting only. NOT part of the
        task execution protocol; nothing in serve_tasks(), the wire
        protocol, or the scheduler reads this. Deterministic and JSON-
        serializable (plain str/int/bool/None values only), so it's safe
        to log or ship as-is. The base implementation covers every
        backend that just wraps worker.executor.execute_task (true of
        every backend in this codebase so far, hence
        supports_serialized_callables defaulting to True here rather than
        False) -- override to add backend-specific fields (see
        MultiprocessingBackend).
        """
        return {
            "backend": type(self).__name__,
            "max_concurrency": self.max_concurrency,
            "supports_serialized_callables": True,
        }


class DirectBackend(ExecutionBackend):
    """The baseline backend (Phase 11.1): runs worker.executor.execute_task
    in the SAME process and the SAME coroutine that calls it -- exactly
    what every worker has always done before Phase 11. No process pool,
    no extra machinery, nothing to start or stop. This is what makes
    DirectBackend the safe default that changes nothing about existing
    behavior: it's not a new execution path, just the old one wearing the
    ExecutionBackend interface so callers can swap it out.
    """

    def __init__(self):
        self._metrics = BackendMetrics()

    async def execute(self, task_type: str, payload: dict) -> dict:
        # execute_task never raises (see ExecutionBackend.execute's
        # docstring) -- there is no backend-error path here to guard
        # with try/except; `running` still needs a finally so a caller
        # that cancels this coroutine before it resumes doesn't leave the
        # counter stuck (a cancellation can only actually be delivered at
        # an await point, and there isn't one inside this synchronous
        # call, but the finally keeps the invariant correct regardless of
        # how a future change might add one).
        self._metrics.submitted += 1
        self._metrics.running += 1
        logger.debug("task_submitted backend=DirectBackend task_type=%s", task_type)
        start = time.perf_counter()
        try:
            result = execute_task(task_type, payload)
        finally:
            self._metrics.running -= 1
        elapsed = time.perf_counter() - start
        self._metrics.total_execution_time += elapsed
        if result["status"] == "success":
            self._metrics.completed += 1
            logger.debug(
                "task_execution_completed backend=DirectBackend task_type=%s duration=%.4f",
                task_type,
                elapsed,
            )
        else:
            self._metrics.failed += 1
            # WARNING, not ERROR -- a structured execution failure is an
            # expected, deterministic outcome (see ExecutionBackend.
            # execute's docstring), not a system fault. Never logs
            # result["message"] -- that can echo back caller-supplied
            # argument values; the error `code` alone is enough to
            # diagnose from logs without risking payload exposure.
            logger.warning(
                "task_execution_failed backend=DirectBackend task_type=%s code=%s",
                task_type,
                result.get("code"),
            )
        return result

    def get_metrics(self) -> BackendMetrics:
        return replace(self._metrics)


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
        # Phase 11.6: every message here states what was actually
        # received, not just which field was wrong -- "invalid
        # max_workers" forces a caller back to the source to find out
        # what value they actually passed; naming it directly doesn't.
        if max_workers is not None and max_workers <= 0:
            raise ValueError(f"max_workers must be a positive integer or None; received {max_workers!r}")
        if max_in_flight is not None and max_in_flight <= 0:
            raise ValueError(f"max_in_flight must be a positive integer or None; received {max_in_flight!r}")
        valid_contexts = ("fork", "spawn", "forkserver")
        if mp_context not in valid_contexts:
            raise ValueError(
                f"mp_context must be one of {valid_contexts}; received {mp_context!r}"
            )

        self._max_workers = max_workers
        self._mp_context = mp_context
        self._max_in_flight = max_in_flight
        self._semaphore: asyncio.Semaphore | None = None
        self._pool: ProcessPoolExecutor | None = None
        self.max_concurrency = max_in_flight if max_in_flight is not None else max_workers
        self._metrics = BackendMetrics()
        logger.info(
            "backend_created backend=MultiprocessingBackend max_workers=%s mp_context=%s max_in_flight=%s",
            max_workers,
            mp_context,
            max_in_flight,
        )

    async def start(self) -> None:
        try:
            self._pool = ProcessPoolExecutor(
                max_workers=self._max_workers,
                mp_context=multiprocessing.get_context(self._mp_context),
            )
        except Exception:
            logger.error(
                "backend_start_failed backend=MultiprocessingBackend max_workers=%s mp_context=%s",
                self._max_workers,
                self._mp_context,
                exc_info=True,
            )
            raise
        if self._max_in_flight is not None:
            self._semaphore = asyncio.Semaphore(self._max_in_flight)
        # Phase 11.5: metrics reset on every start(), matching the
        # semaphore/pool -- each start()/stop() lifetime gets its own
        # clean metrics, not numbers carried over from whatever this
        # backend did in a previous lifetime. Documented contract, not an
        # accident: call get_metrics() before stop() if you need the
        # final numbers from a lifetime that's ending.
        self._metrics = BackendMetrics()
        logger.info(
            "backend_started backend=MultiprocessingBackend max_workers=%s max_in_flight=%s",
            self._max_workers,
            self._max_in_flight,
        )

    async def stop(self) -> None:
        pool, self._pool = self._pool, None
        self._semaphore = None
        if pool is None:
            return
        logger.info("backend_stopping backend=MultiprocessingBackend")
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
        try:
            await loop.run_in_executor(None, lambda: pool.shutdown(wait=True, cancel_futures=True))
        except Exception:
            logger.error("backend_stop_failed backend=MultiprocessingBackend", exc_info=True)
            raise
        logger.info("backend_stopped backend=MultiprocessingBackend")

    async def execute(self, task_type: str, payload: dict) -> dict:
        if self._pool is None:
            raise RuntimeError(
                "MultiprocessingBackend.execute() called before start() (or after stop())"
            )

        self._metrics.submitted += 1
        logger.debug("task_submitted backend=MultiprocessingBackend task_type=%s", task_type)
        try:
            if self._semaphore is None:
                return await self._run_and_record(task_type, payload)

            # Phase 11.4: with max_in_flight configured, a caller beyond
            # the limit waits here (queues) rather than every excess
            # submission being handed straight to the pool's own
            # unbounded internal queue -- "allow tasks to queue instead
            # of being silently dropped" is satisfied by `async with`
            # blocking, not by rejecting anything. Acquired before
            # submission and released in a finally -- guaranteed even if
            # _run_and_record() raises (a normal execution failure
            # returns normally and releases the same way; a
            # BrokenProcessPool propagating still releases before it
            # does, so a permit can never leak just because the pool
            # died).
            wait_start = time.perf_counter()
            async with self._semaphore:
                self._metrics.total_wait_time += time.perf_counter() - wait_start
                return await self._run_and_record(task_type, payload)
        except Exception as exc:
            # Phase 11.5: everything that reaches here is a raise, not a
            # normal {"status": "error", ...} result (those are counted
            # as `failed` inside _run_and_record instead) -- i.e. exactly
            # the BACKEND-failure category from this class's own
            # docstring (a dead child process, or execute() called after
            # stop()). asyncio.CancelledError is a BaseException, not an
            # Exception, since Python 3.8 -- this deliberately does NOT
            # catch it, so a caller cancelling this coroutine (while
            # waiting for the semaphore or for the pool result) is never
            # miscounted as a backend error.
            self._metrics.backend_errors += 1
            # Phase 11.6: process-pool failure diagnostics -- deliberately
            # logs only identifiers (task_type, execution_mode, the
            # exception's TYPE name) and counts, never payload
            # contents/args/results and never a serialized callable's
            # bytes. execution_mode is only present on EXECUTE-task
            # payloads (Phase 10); "n/a" for everything else (built-ins,
            # plain registered functions, MAP/REDUCE).
            logger.error(
                "process_pool_failed backend=MultiprocessingBackend task_type=%s "
                "execution_mode=%s error=%s running_tasks=%d max_workers=%s max_in_flight=%s",
                task_type,
                payload.get("execution_mode", "n/a"),
                type(exc).__name__,
                self._metrics.running,
                self._max_workers,
                self._max_in_flight,
            )
            raise

    async def _run_and_record(self, task_type: str, payload: dict) -> dict:
        self._metrics.running += 1
        logger.debug("task_execution_started backend=MultiprocessingBackend task_type=%s", task_type)
        exec_start = time.perf_counter()
        try:
            result = await self._submit(task_type, payload)
        finally:
            # Runs on success, on a normal execution-failure result, AND
            # on a raise (BrokenProcessPool, or cancellation) -- `running`
            # must never stay stuck incremented just because the task
            # didn't finish cleanly.
            self._metrics.running -= 1
        elapsed = time.perf_counter() - exec_start
        self._metrics.total_execution_time += elapsed
        if result["status"] == "success":
            self._metrics.completed += 1
            logger.debug(
                "task_execution_completed backend=MultiprocessingBackend task_type=%s duration=%.4f",
                task_type,
                elapsed,
            )
        else:
            self._metrics.failed += 1
            # WARNING, not ERROR -- same reasoning as DirectBackend's
            # identical log: a structured execution failure is expected
            # and deterministic, not a system fault. Never logs
            # result["message"].
            logger.warning(
                "task_execution_failed backend=MultiprocessingBackend task_type=%s code=%s",
                task_type,
                result.get("code"),
            )
        return result

    def get_metrics(self) -> BackendMetrics:
        return replace(self._metrics)

    def describe(self) -> dict[str, object]:
        info = super().describe()
        info["max_workers"] = self._max_workers
        info["max_in_flight"] = self._max_in_flight
        return info

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
