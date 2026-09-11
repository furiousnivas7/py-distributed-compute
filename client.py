"""pydc: the command-line entry point for this project (see
pyproject.toml's [project.scripts]).

Three subcommands:

    pydc master   -- start a master and serve indefinitely
    pydc worker   -- start a worker, connecting to a master
    pydc run      -- start an in-process master + one local worker,
                     run ONE job, print its result as JSON, and exit

`pydc master`/`pydc worker` forward directly to master.async_server.main()/
worker.async_worker.main() -- both already parse real process argv via
argparse's parse_known_args, which simply ignores the leading "master"/
"worker" token this dispatcher does NOT strip (there's no positional
argument in either parser to collide with it). So every flag documented
by `pydc master --help` / `pydc worker --help` is identical to running
`python -m master.async_server --help` / `python -m worker.async_worker
--help` directly -- this file adds no new flags or behavior to either,
only a shorter way to reach them.

Why `pydc run` exists instead of a `pydc submit` that talks to an
already-running remote master: master.scheduler is an in-process
singleton, and the wire protocol (rpc/protocol.py) has no client-facing
job-submission message type at all -- only worker<->master messages
(REGISTER/HEARTBEAT/TASK/TASK_RESULT/PING/SHUTDOWN). Submitting to a
REMOTE master from a separate `pydc` process isn't implementable without
adding one, which is a wire-protocol change out of this phase's scope
(Phase 12.3 audit). `pydc run` is the closest available equivalent that
needs no protocol change at all: everything happens in one process,
exactly the pattern this project's own README "Getting Started" examples
and test suite already use.
"""

import argparse
import asyncio
import json
import sys

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:  # pragma: no cover - importlib.metadata is stdlib on 3.10+
    from importlib_metadata import PackageNotFoundError, version

_DISTRIBUTION_NAME = "py-distributed-compute"
_FALLBACK_VERSION = "0.2.0"  # kept in sync with pyproject.toml's version


def _version_string() -> str:
    try:
        return version(_DISTRIBUTION_NAME)
    except PackageNotFoundError:
        # Not installed (e.g. running client.py directly from a checkout
        # without `pip install -e .`) -- still give a sensible answer
        # rather than crashing on --version.
        return _FALLBACK_VERSION


def _print_top_level_help() -> None:
    print(
        "usage: pydc [-h] [--version] {master,worker,run} ...\n"
        "\n"
        "py-distributed-compute command-line interface.\n"
        "\n"
        "commands:\n"
        "  master    Start a master and serve indefinitely.\n"
        "            See `pydc master --help` for its own flags.\n"
        "  worker    Start a worker, connecting to a master.\n"
        "            See `pydc worker --help` for its own flags.\n"
        "  run       Start an in-process master + one local worker, run ONE\n"
        "            job, print its result as JSON, and exit.\n"
        "            See `pydc run --help` for its own flags.\n"
        "\n"
        "options:\n"
        "  -h, --help     show this help message and exit\n"
        "  -V, --version  show program's version number and exit"
    )


def _parse_kv_args(pairs: list[str]) -> dict:
    result = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"pydc run: invalid --arg {pair!r}, expected key=value")
        key, raw_value = pair.split("=", 1)
        try:
            # A CLI value is text by default -- JSON parsing is a
            # convenience so `--arg x=5` means the int 5 (what almost
            # every built-in operation actually expects), not the string
            # "5", without requiring a caller to type `--arg x=5.0` or
            # quote every non-string value awkwardly. Falls back to the
            # raw string for anything that isn't valid JSON (e.g. a plain
            # word), so `--arg name=alice` still works as a string.
            value = json.loads(raw_value)
        except json.JSONDecodeError:
            value = raw_value
        result[key] = value
    return result


def _register_imports(specs: list[str]) -> None:
    """--import module:function:name, repeatable -- the CLI-level version
    of "the operator imports and registers a function" (worker.registry
    already requires exactly this before a registered-function task can
    run); not a new trust boundary, just a CLI-level way to do the same
    thing an operator's own startup code already does."""
    import importlib

    from worker.registry import register_function

    for spec in specs:
        parts = spec.split(":", 2)
        if len(parts) != 3:
            raise SystemExit(f"pydc run: invalid --import {spec!r}, expected module:function:name")
        module_name, func_name, register_as = parts
        module = importlib.import_module(module_name)
        try:
            fn = getattr(module, func_name)
        except AttributeError:
            raise SystemExit(f"pydc run: {module_name!r} has no attribute {func_name!r}") from None
        register_function(register_as, fn)


def _build_run_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pydc run",
        description=(
            "Start an in-process master + one local worker, run ONE job, "
            "print its result as JSON, and exit."
        ),
    )
    subparsers = parser.add_subparsers(dest="job_kind", required=True)

    call_parser = subparsers.add_parser("call", help="Invoke one operation once.")
    call_parser.add_argument(
        "operation", help="A built-in operation (e.g. ADD, MULTIPLY) or a name registered via --import."
    )
    call_parser.add_argument(
        "--arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Keyword argument, repeatable. Value is parsed as JSON if possible, else kept as a string.",
    )
    call_parser.add_argument(
        "--import",
        dest="imports",
        action="append",
        default=[],
        metavar="MODULE:FUNCTION:NAME",
        help="Import MODULE, register its FUNCTION under NAME. Repeatable.",
    )

    mr_parser = subparsers.add_parser("mapreduce", help="Run one Map -> Shuffle -> Reduce job.")
    mr_parser.add_argument("--data", required=True, help="Path to a JSON file containing a JSON array.")
    mr_parser.add_argument(
        "--map",
        dest="map_operation",
        required=True,
        help="Map operation: a built-in (e.g. WORD_COUNT) or a name registered via --import.",
    )
    mr_parser.add_argument(
        "--reduce",
        dest="reduce_operation",
        required=True,
        help="Reduce operation: a built-in (e.g. SUM) or a name registered via --import.",
    )
    mr_parser.add_argument("--partitions", type=int, default=4, help="Number of Map partitions (default: 4).")
    mr_parser.add_argument(
        "--import",
        dest="imports",
        action="append",
        default=[],
        metavar="MODULE:FUNCTION:NAME",
        help="Import MODULE, register its FUNCTION under NAME. Repeatable.",
    )

    return parser


def _cmd_run(argv: list[str]) -> None:
    from worker.config import build_backend, resolve_backend_config

    # parse_known_args here (not parse_args): resolve_backend_config
    # below reads the SAME argv for --backend/--max-workers/... via its
    # own parser -- this run-specific parser only needs to recognize its
    # OWN flags (job_kind, operation, --arg, --data, ...) and silently
    # ignore backend flags it doesn't define, exactly the same
    # ignore-what-you-don't-recognize pattern master/worker's CLIs use.
    args, _ = _build_run_parser().parse_known_args(argv)

    try:
        backend_config = resolve_backend_config(argv=argv)
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None

    _register_imports(args.imports)

    async def scenario():
        from jobs.call import collect_call_result, submit_call
        from jobs.map_reduce import run_map_reduce
        from master import async_server
        from worker.async_worker import run_worker

        backend = build_backend(backend_config)
        server = await async_server.start_master("127.0.0.1", 0)
        host, port = server.sockets[0].getsockname()[:2]
        worker_task = asyncio.create_task(
            run_worker(host, port, worker_id="pydc-run-worker", backend=backend)
        )
        try:
            await async_server.wait_for_workers(1)

            if args.job_kind == "call":
                kwargs = _parse_kv_args(args.arg)
                task = submit_call(async_server.scheduler, "pydc-run-call", args.operation, **kwargs)
                [response] = await async_server.wait_for_tasks({task.task_id})
                return collect_call_result(response)

            with open(args.data) as f:
                data = json.load(f)
            return await run_map_reduce(
                async_server.scheduler,
                async_server.drain_tasks_for,
                "pydc-run-job",
                data,
                args.map_operation,
                args.reduce_operation,
                args.partitions,
            )
        finally:
            worker_task.cancel()
            try:
                await worker_task
            except (asyncio.CancelledError, Exception):
                pass
            await async_server.stop_master()

    try:
        result = asyncio.run(scenario())
    except Exception as exc:
        print(f"Job failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from None

    print(json.dumps(result))


def main() -> None:
    argv = sys.argv[1:]

    if not argv or argv[0] in ("-h", "--help"):
        _print_top_level_help()
        return
    if argv[0] in ("-V", "--version"):
        print(_version_string())
        return

    command = argv[0]
    if command == "master":
        from master.async_server import main as master_main

        master_main()
    elif command == "worker":
        from worker.async_worker import main as worker_main

        worker_main()
    elif command == "run":
        _cmd_run(argv[1:])
    else:
        print(f"pydc: unknown command {command!r} (expected: master, worker, run)", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
