"""Phase 12.3: master.config's MasterConfig -- host/port a master process
binds to, resolved CLI > env > defaults, same pattern as
worker.config.resolve_backend_config/resolve_worker_runtime_config."""

import pytest

from master.config import ENV_MASTER_HOST, ENV_MASTER_PORT, MasterConfig, resolve_master_config


def test_defaults_when_nothing_is_configured():
    config = resolve_master_config(argv=[], env={})
    assert config == MasterConfig(host="127.0.0.1", port=5000)


def test_env_configures_both_fields():
    env = {ENV_MASTER_HOST: "0.0.0.0", ENV_MASTER_PORT: "9000"}
    config = resolve_master_config(argv=[], env=env)
    assert config == MasterConfig(host="0.0.0.0", port=9000)


def test_cli_configures_both_fields():
    argv = ["--host", "0.0.0.0", "--port", "9001"]
    config = resolve_master_config(argv=argv, env={})
    assert config == MasterConfig(host="0.0.0.0", port=9001)


def test_cli_overrides_env():
    env = {ENV_MASTER_HOST: "from-env", ENV_MASTER_PORT: "1111"}
    config = resolve_master_config(argv=["--host", "from-cli"], env=env)
    assert config.host == "from-cli"
    assert config.port == 1111  # CLI omitted it, env still applies


def test_non_integer_port_is_rejected_with_an_actionable_message():
    with pytest.raises(ValueError, match="port must be an integer; received 'abc'"):
        resolve_master_config(argv=["--port", "abc"], env={})


def test_master_and_worker_share_the_same_env_var_names():
    """common/env.py's whole point -- both sides agree on
    PY_DISTRIBUTED_MASTER_HOST/PORT without master importing worker or
    vice versa."""
    from worker.config import ENV_MASTER_HOST as worker_env_host
    from worker.config import ENV_MASTER_PORT as worker_env_port

    assert ENV_MASTER_HOST == worker_env_host == "PY_DISTRIBUTED_MASTER_HOST"
    assert ENV_MASTER_PORT == worker_env_port == "PY_DISTRIBUTED_MASTER_PORT"
