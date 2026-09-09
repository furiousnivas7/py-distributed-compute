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
from dataclasses import dataclass

from worker.backend import DirectBackend, ExecutionBackend, MultiprocessingBackend

logger = logging.getLogger(__name__)

ENV_BACKEND = "PY_DISTRIBUTED_BACKEND"
ENV_MAX_WORKERS = "PY_DISTRIBUTED_MAX_WORKERS"
ENV_MAX_IN_FLIGHT = "PY_DISTRIBUTED_MAX_IN_FLIGHT"
ENV_MP_CONTEXT = "PY_DISTRIBUTED_MP_CONTEXT"

VALID_BACKENDS = ("direct", "multiprocessing")


@dataclass(frozen=True)
class BackendConfig:
    """The resolved configuration -- NOT yet a backend instance (see
    build_backend). Immutable: a config, once resolved, is a fact about
    what was requested, not something later code should mutate."""

    backend: str = "direct"
    max_workers: int | None = None
    max_in_flight: int | None = None
    mp_context: str = "fork"


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
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--max-workers", default=None)
    parser.add_argument("--max-in-flight", default=None)
    parser.add_argument("--mp-context", default=None)
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
