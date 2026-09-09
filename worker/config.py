"""Phase 11.7: selects and configures a worker's ExecutionBackend without
touching application code -- "execution strategy is a worker runtime
concern, not a scheduler concern" (the same principle Phase 11.3 applied to
backend lifecycle) now extends to backend SELECTION too. Three sources, in
precedence order:

    CLI arguments > environment variables > programmatic defaults

Nothing here validates max_workers/max_in_flight/mp_context VALUES itself
beyond parsing them as integers/strings -- that's already
MultiprocessingBackend's own job (Phase 11.4/11.6's actionable "; received
..." ValueErrors), and build_backend() just forwards to its constructor so
that validation runs exactly once, in exactly one place.
"""

import argparse
import logging
import os
import uuid
from dataclasses import dataclass

from common.env import ENV_MASTER_HOST, ENV_MASTER_PORT
from worker.backend import DirectBackend, ExecutionBackend, MultiprocessingBackend

logger = logging.getLogger(__name__)

ENV_BACKEND = "PY_DISTRIBUTED_BACKEND"
ENV_MAX_WORKERS = "PY_DISTRIBUTED_MAX_WORKERS"
ENV_MAX_IN_FLIGHT = "PY_DISTRIBUTED_MAX_IN_FLIGHT"
ENV_MP_CONTEXT = "PY_DISTRIBUTED_MP_CONTEXT"
ENV_WORKER_ID = "PY_DISTRIBUTED_WORKER_ID"
ENV_WORKER_HOST = "PY_DISTRIBUTED_WORKER_HOST"
ENV_WORKER_PORT = "PY_DISTRIBUTED_WORKER_PORT"

VALID_BACKENDS = ("direct", "multiprocessing")

DEFAULT_MASTER_HOST = "127.0.0.1"
DEFAULT_MASTER_PORT = 5000
DEFAULT_WORKER_HOST = "127.0.0.1"
DEFAULT_WORKER_PORT = 6001


@dataclass(frozen=True)
class BackendConfig:
    """The resolved configuration -- NOT yet a backend instance (see
    build_backend). Immutable: a config, once resolved, is a fact about
    what was requested, not something later code should mutate."""

    backend: str = "direct"
    max_workers: int | None = None
    max_in_flight: int | None = None
    mp_context: str = "fork"


@dataclass(frozen=True)
class WorkerRuntimeConfig:
    """Phase 12.3: WHERE and WHO a worker process is -- deliberately
    separate from BackendConfig (WHAT it executes with), a distinct
    concern resolved from the same CLI-args/env-vars/defaults precedence.
    Every field here was already a plain parameter to run_worker() since
    Phase 9/11.1 -- this only adds a CLI/env-driven way to supply them
    instead of the async_worker module's own hardcoded constants, so two
    worker processes on the same machine (or a worker pointed at a
    non-default master) don't require editing source to run.
    """

    master_host: str
    master_port: int
    worker_id: str
    worker_host: str
    worker_port: int


def _parse_int(field_name: str, raw: str) -> int:
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{field_name} must be an integer; received {raw!r}") from None


def _build_arg_parser() -> argparse.ArgumentParser:
    # No `choices=`/`type=int` here deliberately -- argparse would raise
    # SystemExit(2) for a bad value, which is awkward to test and
    # inconsistent with this project's ValueError-with-actionable-message
    # style everywhere else (Phase 11.4/11.6). Values are taken as plain
    # strings and validated uniformly, once, in resolve_backend_config /
    # build_backend, regardless of which source (CLI, env, default) they
    # came from.
    #
    # add_help=True (Phase 11.8): worker.config's parser is the only CLI
    # argument surface a worker process has (see worker/async_worker.py's
    # main()), so there's nothing else here for --help to conflict with --
    # `python -m worker.async_worker --help` now prints usage for these
    # four flags and exits, instead of --help silently falling through
    # parse_known_args as an unrecognized argument.
    parser = argparse.ArgumentParser(
        prog="worker.async_worker",
        description=(
            "Run a worker process. Backend selection precedence: "
            "CLI arguments > environment variables > defaults."
        ),
    )
    parser.add_argument(
        "--backend",
        default=None,
        help=(
            f"Execution backend to use, one of {VALID_BACKENDS!r}. "
            f"Falls back to the {ENV_BACKEND} environment variable, then 'direct'."
        ),
    )
    parser.add_argument(
        "--max-workers",
        default=None,
        help=(
            "Process pool size for the multiprocessing backend (positive integer). "
            f"Falls back to {ENV_MAX_WORKERS}, then the pool's own default "
            "(os.cpu_count()). Ignored by the direct backend."
        ),
    )
    parser.add_argument(
        "--max-in-flight",
        default=None,
        help=(
            "Cap on concurrently in-flight executions for the multiprocessing "
            f"backend (positive integer). Falls back to {ENV_MAX_IN_FLIGHT}, then "
            "unbounded. Ignored by the direct backend."
        ),
    )
    parser.add_argument(
        "--mp-context",
        default=None,
        help=(
            "multiprocessing start method for the multiprocessing backend, one of "
            f"('fork', 'spawn', 'forkserver'). Falls back to {ENV_MP_CONTEXT}, "
            "then 'fork'. Ignored by the direct backend."
        ),
    )
    parser.add_argument(
        "--master-host",
        default=None,
        help=f"Master host to connect to. Falls back to {ENV_MASTER_HOST}, then {DEFAULT_MASTER_HOST!r}.",
    )
    parser.add_argument(
        "--master-port",
        default=None,
        help=f"Master port to connect to. Falls back to {ENV_MASTER_PORT}, then {DEFAULT_MASTER_PORT}.",
    )
    parser.add_argument(
        "--worker-id",
        default=None,
        help=(
            f"This worker's id, reported to the master on REGISTER. Falls back to "
            f"{ENV_WORKER_ID}, then a randomly generated id (so multiple workers "
            "started without this flag never collide)."
        ),
    )
    parser.add_argument(
        "--worker-host",
        default=None,
        help=(
            f"This worker's own host, reported to the master as metadata (Phase 8+ "
            f"never dials back to it). Falls back to {ENV_WORKER_HOST}, then "
            f"{DEFAULT_WORKER_HOST!r}."
        ),
    )
    parser.add_argument(
        "--worker-port",
        default=None,
        help=(
            f"This worker's own port, reported to the master as metadata. Falls "
            f"back to {ENV_WORKER_PORT}, then {DEFAULT_WORKER_PORT}."
        ),
    )
    return parser


def resolve_backend_config(argv: list[str] | None = None, env: dict | None = None) -> BackendConfig:
    """Resolve which backend a worker should use, and how it should be
    configured, from CLI args / env vars / defaults (in that precedence
    order). `argv`/`env` are injectable purely for testability -- a
    production caller (worker.async_worker.main()) passes neither and
    gets the real process argv (parse_known_args ignores anything this
    parser doesn't recognize, so other worker CLI flags aren't disturbed)
    and os.environ.
    """
    if env is None:
        env = os.environ

    args, _ = _build_arg_parser().parse_known_args(argv)

    backend = args.backend or env.get(ENV_BACKEND) or "direct"
    if backend not in VALID_BACKENDS:
        raise ValueError(f"backend must be one of {VALID_BACKENDS}; received {backend!r}")

    max_workers_raw = args.max_workers if args.max_workers is not None else env.get(ENV_MAX_WORKERS)
    max_workers = _parse_int("max_workers", max_workers_raw) if max_workers_raw is not None else None

    max_in_flight_raw = (
        args.max_in_flight if args.max_in_flight is not None else env.get(ENV_MAX_IN_FLIGHT)
    )
    max_in_flight = (
        _parse_int("max_in_flight", max_in_flight_raw) if max_in_flight_raw is not None else None
    )

    mp_context = args.mp_context or env.get(ENV_MP_CONTEXT) or "fork"

    config = BackendConfig(
        backend=backend, max_workers=max_workers, max_in_flight=max_in_flight, mp_context=mp_context
    )
    logger.info(
        "backend_config_resolved backend=%s max_workers=%s max_in_flight=%s mp_context=%s",
        config.backend,
        config.max_workers,
        config.max_in_flight,
        config.mp_context,
    )
    return config


def resolve_worker_runtime_config(argv: list[str] | None = None, env: dict | None = None) -> WorkerRuntimeConfig:
    """Resolve WHERE/WHO a worker is -- same CLI/env/defaults precedence
    and injectable argv/env as resolve_backend_config, from the SAME
    shared parser (both functions parse their own subset of its flags out
    of the same argv; the flags each doesn't care about are simply
    unused, not an error -- see _build_arg_parser)."""
    if env is None:
        env = os.environ

    args, _ = _build_arg_parser().parse_known_args(argv)

    master_host = args.master_host or env.get(ENV_MASTER_HOST) or DEFAULT_MASTER_HOST
    master_port_raw = args.master_port if args.master_port is not None else env.get(ENV_MASTER_PORT)
    master_port = (
        _parse_int("master_port", master_port_raw) if master_port_raw is not None else DEFAULT_MASTER_PORT
    )

    # No CLI/env value means a fresh random id per process, not a shared
    # default -- unlike every other field here, "worker-1" for every
    # worker that didn't set one would guarantee a collision the moment a
    # second worker starts.
    worker_id = args.worker_id or env.get(ENV_WORKER_ID) or f"worker-{uuid.uuid4().hex[:8]}"

    worker_host = args.worker_host or env.get(ENV_WORKER_HOST) or DEFAULT_WORKER_HOST
    worker_port_raw = args.worker_port if args.worker_port is not None else env.get(ENV_WORKER_PORT)
    worker_port = (
        _parse_int("worker_port", worker_port_raw) if worker_port_raw is not None else DEFAULT_WORKER_PORT
    )

    config = WorkerRuntimeConfig(
        master_host=master_host,
        master_port=master_port,
        worker_id=worker_id,
        worker_host=worker_host,
        worker_port=worker_port,
    )
    logger.info(
        "worker_runtime_config_resolved master_host=%s master_port=%s worker_id=%s worker_host=%s worker_port=%s",
        config.master_host,
        config.master_port,
        config.worker_id,
        config.worker_host,
        config.worker_port,
    )
    return config


def build_backend(config: BackendConfig) -> ExecutionBackend:
    """Construct the ExecutionBackend a resolved BackendConfig describes.

    mp_context/max_workers/max_in_flight are forwarded to
    MultiprocessingBackend UNCONDITIONALLY -- even the "direct" branch's
    unused fields on `config` are simply ignored rather than validated,
    since they have no meaning for DirectBackend at all.
    """
    if config.backend == "direct":
        return DirectBackend()
    if config.backend == "multiprocessing":
        return MultiprocessingBackend(
            max_workers=config.max_workers,
            mp_context=config.mp_context,
            max_in_flight=config.max_in_flight,
        )
    # Unreachable via resolve_backend_config (already validated above),
    # but build_backend is public -- a caller that constructs a
    # BackendConfig directly (bypassing resolve_backend_config) still
    # gets the same actionable error rather than a silent None/KeyError.
    raise ValueError(f"backend must be one of {VALID_BACKENDS}; received {config.backend!r}")
