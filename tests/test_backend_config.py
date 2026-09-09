"""Phase 11.7: worker.config's runtime backend selection -- resolving a
BackendConfig from CLI args / environment variables / defaults (in that
precedence order), and building the actual ExecutionBackend it describes.
Value validation itself (max_workers <= 0, unsupported mp_context, etc.) is
NOT reimplemented here -- it's already MultiprocessingBackend's job (Phase
11.4/11.6); these tests confirm build_backend() reaches that same
validation rather than silently swallowing a bad value.
"""

import asyncio
import os
import subprocess
import sys

import pytest

from worker import registry
from worker.backend import DirectBackend, MultiprocessingBackend
from worker.config import BackendConfig, ENV_BACKEND, ENV_MAX_IN_FLIGHT, ENV_MAX_WORKERS, ENV_MP_CONTEXT, build_backend, resolve_backend_config


@pytest.fixture(autouse=True)
def reset_registry():
    registry.clear()
    yield
    registry.clear()


# -- defaults --------------------------------------------------------------


def test_default_backend_is_direct_when_nothing_is_configured():
    config = resolve_backend_config(argv=[], env={})
    assert config == BackendConfig(backend="direct", max_workers=None, max_in_flight=None, mp_context="fork")


def test_build_backend_default_produces_direct_backend():
    backend = build_backend(resolve_backend_config(argv=[], env={}))
    assert isinstance(backend, DirectBackend)


# -- environment variables ---------------------------------------------


def test_env_selects_multiprocessing_backend():
    env = {ENV_BACKEND: "multiprocessing", ENV_MAX_WORKERS: "3", ENV_MAX_IN_FLIGHT: "2", ENV_MP_CONTEXT: "spawn"}
    config = resolve_backend_config(argv=[], env=env)
    assert config == BackendConfig(backend="multiprocessing", max_workers=3, max_in_flight=2, mp_context="spawn")


def test_build_backend_from_env_produces_correctly_configured_multiprocessing_backend():
    env = {ENV_BACKEND: "multiprocessing", ENV_MAX_WORKERS: "3", ENV_MAX_IN_FLIGHT: "2", ENV_MP_CONTEXT: "spawn"}
    backend = build_backend(resolve_backend_config(argv=[], env=env))
    assert isinstance(backend, MultiprocessingBackend)
    info = backend.describe()
    assert info["max_workers"] == 3
    assert info["max_in_flight"] == 2


# -- CLI arguments -----------------------------------------------------


def test_cli_selects_multiprocessing_backend():
    argv = ["--backend", "multiprocessing", "--max-workers", "5", "--max-in-flight", "1", "--mp-context", "forkserver"]
    config = resolve_backend_config(argv=argv, env={})
    assert config == BackendConfig(backend="multiprocessing", max_workers=5, max_in_flight=1, mp_context="forkserver")


# -- precedence: CLI > env > defaults -----------------------------------


def test_cli_overrides_env():
    env = {ENV_BACKEND: "direct", ENV_MAX_WORKERS: "9"}
    argv = ["--backend", "multiprocessing", "--max-workers", "2"]
    config = resolve_backend_config(argv=argv, env=env)
    assert config.backend == "multiprocessing"
    assert config.max_workers == 2


def test_env_overrides_default_when_cli_omits_the_field():
    env = {ENV_BACKEND: "multiprocessing", ENV_MAX_WORKERS: "7"}
    config = resolve_backend_config(argv=[], env=env)
    assert config.backend == "multiprocessing"
    assert config.max_workers == 7


def test_cli_can_override_only_one_field_leaving_others_from_env():
    env = {ENV_BACKEND: "multiprocessing", ENV_MAX_WORKERS: "7", ENV_MAX_IN_FLIGHT: "4"}
    argv = ["--max-workers", "2"]  # backend/max_in_flight still come from env
    config = resolve_backend_config(argv=argv, env=env)
    assert config.backend == "multiprocessing"
    assert config.max_workers == 2  # CLI wins
    assert config.max_in_flight == 4  # env, since CLI omitted it


# -- validation: unknown backend names -----------------------------------


def test_unknown_backend_name_from_cli_is_rejected_with_the_received_value():
    with pytest.raises(ValueError, match="received 'not-a-real-backend'"):
        resolve_backend_config(argv=["--backend", "not-a-real-backend"], env={})


def test_unknown_backend_name_from_env_is_rejected_with_the_received_value():
    with pytest.raises(ValueError, match="received 'not-a-real-backend'"):
        resolve_backend_config(argv=[], env={ENV_BACKEND: "not-a-real-backend"})


def test_build_backend_rejects_an_unknown_backend_name_constructed_directly():
    with pytest.raises(ValueError, match="backend"):
        build_backend(BackendConfig(backend="not-a-real-backend"))


# -- validation: invalid worker counts / in-flight limits ----------------


def test_non_integer_max_workers_is_rejected_with_an_actionable_message():
    with pytest.raises(ValueError, match="max_workers must be an integer; received 'abc'"):
        resolve_backend_config(argv=["--backend", "multiprocessing", "--max-workers", "abc"], env={})


def test_non_integer_max_in_flight_is_rejected_with_an_actionable_message():
    with pytest.raises(ValueError, match="max_in_flight must be an integer; received 'xyz'"):
        resolve_backend_config(
            argv=["--backend", "multiprocessing", "--max-in-flight", "xyz"], env={}
        )


def test_zero_max_workers_is_rejected_when_the_backend_is_actually_built():
    """resolve_backend_config only parses -- the positive-integer check
    itself is MultiprocessingBackend's own validation (Phase 11.4),
    reached via build_backend()."""
    config = resolve_backend_config(argv=["--backend", "multiprocessing", "--max-workers", "0"], env={})
    with pytest.raises(ValueError, match="max_workers"):
        build_backend(config)


def test_negative_max_in_flight_is_rejected_when_the_backend_is_actually_built():
    config = resolve_backend_config(
        argv=["--backend", "multiprocessing", "--max-in-flight", "-3"], env={}
    )
    with pytest.raises(ValueError, match="max_in_flight"):
        build_backend(config)


# -- validation: unsupported multiprocessing contexts ---------------------


def test_unsupported_mp_context_is_rejected_when_the_backend_is_actually_built():
    config = resolve_backend_config(
        argv=["--backend", "multiprocessing", "--mp-context", "not-a-real-context"], env={}
    )
    with pytest.raises(ValueError, match="mp_context"):
        build_backend(config)


# -- startup diagnostics: selected backend is visible ---------------------


def test_selected_backend_is_visible_through_describe():
    config = resolve_backend_config(
        argv=["--backend", "multiprocessing", "--max-workers", "4", "--max-in-flight", "8"], env={}
    )
    backend = build_backend(config)
    info = backend.describe()
    assert info["backend"] == "MultiprocessingBackend"
    assert info["max_workers"] == 4
    assert info["max_in_flight"] == 8


def test_backend_config_resolution_is_logged(caplog):
    import logging

    with caplog.at_level(logging.INFO, logger="worker.config"):
        resolve_backend_config(argv=["--backend", "multiprocessing", "--max-workers", "4"], env={})

    messages = [r.getMessage() for r in caplog.records]
    assert any("backend_config_resolved" in m and "multiprocessing" in m for m in messages)


# -- integration: config -> backend -> real worker pipeline ---------------


def test_full_config_to_backend_to_worker_pipeline_preserves_execution_behavior():
    """The whole point of Phase 11.7: an operator selecting the backend
    via env vars must produce a worker that behaves EXACTLY like one
    built with an explicitly constructed MultiprocessingBackend --
    execution, results, and retry-relevant behavior are all unaffected by
    how the backend was selected."""
    from master import async_server, rpc_handler
    from worker import async_worker

    registry.register_function("square", lambda x: x * x)
    rpc_handler.worker_manager.clear()
    async_server.scheduler.clear()
    async_server.connections.clear()

    env = {ENV_BACKEND: "multiprocessing", ENV_MAX_WORKERS: "2", ENV_MAX_IN_FLIGHT: "2"}
    config = resolve_backend_config(argv=[], env=env)
    backend = build_backend(config)

    async def scenario():
        server = await asyncio.start_server(async_server.handle_worker_connection, "127.0.0.1", 0)
        host, port = server.sockets[0].getsockname()[:2]
        worker_task = asyncio.create_task(
            async_worker.run_worker(host, port, worker_id="worker-1", backend=backend)
        )
        await async_server.wait_for_workers(1)
        try:
            task = async_server.scheduler.submit_task("t1", "square", {"x": 6})
            [response] = await async_server.wait_for_tasks({task.task_id})
            return response
        finally:
            worker_task.cancel()
            try:
                await worker_task
            except (asyncio.CancelledError, Exception):
                pass
            server.close()
            await server.wait_closed()

    response = asyncio.run(scenario())
    assert response["payload"]["result"] == 36


# -- Phase 11.8: subprocess-level CLI/env resolution and CLI polish ------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_PRINT_RESOLVED_BACKEND = (
    "from worker.config import build_backend, resolve_backend_config; "
    "print(build_backend(resolve_backend_config()).describe())"
)


def _run(args, env_extra=None):
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-c", _PRINT_RESOLVED_BACKEND, *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )


def test_subprocess_cli_resolution_selects_multiprocessing_backend():
    result = _run(["--backend", "multiprocessing", "--max-workers", "3"])
    assert result.returncode == 0
    assert "'backend': 'MultiprocessingBackend'" in result.stdout
    assert "'max_workers': 3" in result.stdout


def test_subprocess_env_resolution_selects_multiprocessing_backend():
    result = _run(
        [],
        env_extra={ENV_BACKEND: "multiprocessing", ENV_MAX_WORKERS: "4", ENV_MAX_IN_FLIGHT: "2"},
    )
    assert result.returncode == 0
    assert "'backend': 'MultiprocessingBackend'" in result.stdout
    assert "'max_workers': 4" in result.stdout
    assert "'max_in_flight': 2" in result.stdout


def test_subprocess_cli_overrides_env_end_to_end():
    result = _run(
        ["--max-workers", "9"],
        env_extra={ENV_BACKEND: "multiprocessing", ENV_MAX_WORKERS: "1"},
    )
    assert result.returncode == 0
    assert "'max_workers': 9" in result.stdout


def test_subprocess_default_with_no_configuration_is_direct_backend():
    env = {k: v for k, v in os.environ.items() if not k.startswith("PY_DISTRIBUTED_")}
    result = subprocess.run(
        [sys.executable, "-c", _PRINT_RESOLVED_BACKEND],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0
    assert "'backend': 'DirectBackend'" in result.stdout


def test_worker_cli_help_exits_cleanly_with_usage_text():
    result = subprocess.run(
        [sys.executable, "-m", "worker.async_worker", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0
    assert "--backend" in result.stdout
    assert "--max-workers" in result.stdout
    assert "--max-in-flight" in result.stdout
    assert "--mp-context" in result.stdout


def test_worker_cli_invalid_backend_reports_a_clean_error_not_a_traceback():
    result = subprocess.run(
        [sys.executable, "-m", "worker.async_worker", "--backend", "not-a-real-backend"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 1
    assert "Configuration error:" in result.stderr
    assert "not-a-real-backend" in result.stderr
    assert "Traceback" not in result.stderr


def test_worker_cli_invalid_max_workers_reports_a_clean_error_not_a_traceback():
    result = subprocess.run(
        [sys.executable, "-m", "worker.async_worker", "--backend", "multiprocessing", "--max-workers", "0"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 1
    assert "Configuration error:" in result.stderr
    assert "received 0" in result.stderr
    assert "Traceback" not in result.stderr
