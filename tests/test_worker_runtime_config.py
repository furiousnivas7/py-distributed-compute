"""Phase 12.3: worker.config's WorkerRuntimeConfig -- WHERE/WHO a worker
is (master host/port, its own id/host/port), resolved CLI > env > defaults
just like BackendConfig (tests/test_backend_config.py). Value validation
itself is minimal (just integer parsing for ports) -- there's no
"positive integer" style constraint a port could meaningfully fail beyond
that, unlike max_workers/max_in_flight.
"""

import pytest

from worker.config import (
    ENV_MASTER_HOST,
    ENV_MASTER_PORT,
    ENV_WORKER_HOST,
    ENV_WORKER_ID,
    ENV_WORKER_PORT,
    WorkerRuntimeConfig,
    resolve_worker_runtime_config,
)


# -- defaults ----------------------------------------------------------


def test_defaults_when_nothing_is_configured():
    config = resolve_worker_runtime_config(argv=[], env={})
    assert config.master_host == "127.0.0.1"
    assert config.master_port == 5000
    assert config.worker_host == "127.0.0.1"
    assert config.worker_port == 6001
    assert config.worker_id.startswith("worker-")


def test_worker_id_is_randomly_generated_and_unique_per_resolution():
    """No CLI/env value means a FRESH id each time -- never a shared
    hardcoded default -- so two workers started without --worker-id never
    collide."""
    first = resolve_worker_runtime_config(argv=[], env={})
    second = resolve_worker_runtime_config(argv=[], env={})
    assert first.worker_id != second.worker_id


# -- environment variables ----------------------------------------------


def test_env_configures_every_field():
    env = {
        ENV_MASTER_HOST: "10.0.0.5",
        ENV_MASTER_PORT: "9999",
        ENV_WORKER_ID: "my-worker",
        ENV_WORKER_HOST: "10.0.0.6",
        ENV_WORKER_PORT: "7000",
    }
    config = resolve_worker_runtime_config(argv=[], env=env)
    assert config == WorkerRuntimeConfig(
        master_host="10.0.0.5", master_port=9999, worker_id="my-worker", worker_host="10.0.0.6", worker_port=7000
    )


# -- CLI arguments --------------------------------------------------------


def test_cli_configures_every_field():
    argv = [
        "--master-host", "192.168.1.1",
        "--master-port", "6000",
        "--worker-id", "cli-worker",
        "--worker-host", "192.168.1.2",
        "--worker-port", "8000",
    ]
    config = resolve_worker_runtime_config(argv=argv, env={})
    assert config == WorkerRuntimeConfig(
        master_host="192.168.1.1", master_port=6000, worker_id="cli-worker", worker_host="192.168.1.2", worker_port=8000
    )


# -- precedence: CLI > env > defaults -------------------------------------


def test_cli_overrides_env():
    env = {ENV_MASTER_HOST: "from-env", ENV_WORKER_ID: "from-env-id"}
    argv = ["--master-host", "from-cli"]
    config = resolve_worker_runtime_config(argv=argv, env=env)
    assert config.master_host == "from-cli"
    assert config.worker_id == "from-env-id"  # CLI omitted it, env still applies


def test_env_overrides_default_when_cli_omits_the_field():
    config = resolve_worker_runtime_config(argv=[], env={ENV_MASTER_PORT: "4242"})
    assert config.master_port == 4242


# -- validation: non-integer ports ------------------------------------


def test_non_integer_master_port_is_rejected_with_an_actionable_message():
    with pytest.raises(ValueError, match="master_port must be an integer; received 'abc'"):
        resolve_worker_runtime_config(argv=["--master-port", "abc"], env={})


def test_non_integer_worker_port_is_rejected_with_an_actionable_message():
    with pytest.raises(ValueError, match="worker_port must be an integer; received 'xyz'"):
        resolve_worker_runtime_config(argv=["--worker-port", "xyz"], env={})
