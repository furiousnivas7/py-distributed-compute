"""Phase 12.3: resolves which host/port a master process binds to, from
CLI arguments / environment variables / defaults (in that precedence
order) -- the same pattern worker/config.py established for backend
selection and worker identity. This is a CLI/runtime concern only: the
scheduler and dispatch machinery (master/scheduler.py,
master/async_server.py's dispatcher_loop) are completely unaware of how
`host`/`port` got chosen.
"""

import argparse
import logging
import os
from dataclasses import dataclass

from common.env import ENV_MASTER_HOST, ENV_MASTER_PORT

logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5000


@dataclass(frozen=True)
class MasterConfig:
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT


def _parse_int(field_name: str, raw: str) -> int:
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{field_name} must be an integer; received {raw!r}") from None


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="master.async_server",
        description=(
            "Start the master and serve indefinitely. "
            "Configuration precedence: CLI arguments > environment variables > defaults."
        ),
    )
    parser.add_argument(
        "--host",
        default=None,
        help=f"Host to bind to. Falls back to {ENV_MASTER_HOST}, then {DEFAULT_HOST!r}.",
    )
    parser.add_argument(
        "--port",
        default=None,
        help=f"Port to bind to. Falls back to {ENV_MASTER_PORT}, then {DEFAULT_PORT}.",
    )
    return parser


def resolve_master_config(argv: list[str] | None = None, env: dict | None = None) -> MasterConfig:
    """`argv`/`env` are injectable purely for testability -- a production
    caller (master.async_server.main()) passes neither and gets the real
    process argv/environment, matching worker/config.py's
    resolve_backend_config."""
    if env is None:
        env = os.environ

    args, _ = _build_arg_parser().parse_known_args(argv)

    host = args.host or env.get(ENV_MASTER_HOST) or DEFAULT_HOST
    port_raw = args.port if args.port is not None else env.get(ENV_MASTER_PORT)
    port = _parse_int("port", port_raw) if port_raw is not None else DEFAULT_PORT

    config = MasterConfig(host=host, port=port)
    logger.info("master_config_resolved host=%s port=%s", config.host, config.port)
    return config
