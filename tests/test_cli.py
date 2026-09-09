"""Phase 12.3: client.py (`pydc`) end-to-end, via real subprocesses --
proves the actual CLI a user runs works, not just the argparse plumbing
underneath it. Covers: top-level dispatch (--help/--version/unknown
command), `pydc run call`/`pydc run mapreduce` (including --import and
--backend forwarding), error paths (clean message + exit 1, no
traceback), and that `pydc master`/`pydc worker` genuinely delegate to
master.async_server.main()/worker.async_worker.main() with no behavior
of their own layered on top.
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CLIENT_PY = REPO_ROOT / "client.py"


def _run(args, timeout=30, env=None):
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    return subprocess.run(
        [sys.executable, str(CLIENT_PY), *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
        env=full_env,
    )


# -- top-level dispatch -----------------------------------------------


def test_no_args_prints_help_and_exits_zero():
    result = _run([])
    assert result.returncode == 0
    assert "pydc" in result.stdout
    assert "master" in result.stdout
    assert "worker" in result.stdout
    assert "run" in result.stdout


def test_help_flag_prints_help_and_exits_zero():
    result = _run(["--help"])
    assert result.returncode == 0
    assert "master" in result.stdout


def test_version_flag_prints_a_version_and_exits_zero():
    result = _run(["--version"])
    assert result.returncode == 0
    assert result.stdout.strip()  # non-empty


def test_unknown_command_is_a_clean_error_not_a_traceback():
    result = _run(["bogus"])
    assert result.returncode == 2
    assert "unknown command" in result.stderr
    assert "Traceback" not in result.stderr


# -- pydc master / pydc worker delegate to the existing CLIs --------------


def test_master_help_delegates_to_master_async_server_parser():
    result = _run(["master", "--help"])
    assert result.returncode == 0
    assert "--host" in result.stdout
    assert "--port" in result.stdout


def test_worker_help_delegates_to_worker_async_worker_parser():
    result = _run(["worker", "--help"])
    assert result.returncode == 0
    assert "--backend" in result.stdout
    assert "--master-host" in result.stdout
    assert "--worker-id" in result.stdout


def test_worker_invalid_backend_reports_a_clean_error_not_a_traceback():
    result = _run(["worker", "--backend", "not-a-real-backend"])
    assert result.returncode == 1
    assert "Configuration error:" in result.stderr
    assert "Traceback" not in result.stderr


def test_master_and_worker_start_and_stop_cleanly_together():
    """The real end-to-end path: `pydc master` binds and serves, `pydc
    worker` connects to it -- both via the actual pydc dispatcher, not a
    direct module invocation."""
    port = "15199"
    # -u (unbuffered): stdout/stderr are piped here, not a tty, so
    # Python's default block-buffering can hold prints back until the
    # process exits normally -- which a killed process never does. Without
    # -u, output this test actually needs to see can be silently lost.
    master_proc = subprocess.Popen(
        [sys.executable, "-u", str(CLIENT_PY), "master", "--port", port],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        time.sleep(1.0)
        assert master_proc.poll() is None, "master exited early"

        worker_proc = subprocess.Popen(
            [sys.executable, "-u", str(CLIENT_PY), "worker", "--master-port", port, "--worker-id", "cli-e2e-worker"],
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            worker_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass  # expected -- a worker with no shutdown mechanism runs until killed
        finally:
            worker_proc.kill()
            worker_output = worker_proc.communicate()[0]

        assert "Worker registered" in worker_output or "Status: success" in worker_output
    finally:
        master_proc.send_signal(signal.SIGINT)
        try:
            master_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            master_proc.kill()
            master_proc.wait(timeout=5)
        assert master_proc.returncode == 0 or master_proc.returncode == -signal.SIGINT


# -- pydc run call ----------------------------------------------------


def test_run_call_builtin_operation():
    result = _run(["run", "call", "ADD", "--arg", "a=10", "--arg", "b=32"])
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == 42


def test_run_call_string_argument_stays_a_string():
    result = _run(["run", "call", "ADD", "--arg", "a=1", "--arg", 'b={"not": "a number"}'])
    # Deliberately a type mismatch (ADD expects numbers) -- proves --arg's
    # JSON parsing is real, not just passing strings through: this fails
    # with a normal execution error, not a CLI parsing error.
    assert result.returncode == 1
    assert "Job failed:" in result.stderr


def test_run_call_unknown_operation_is_a_clean_error():
    result = _run(["run", "call", "NOT_A_REAL_OPERATION"])
    assert result.returncode == 1
    assert "Job failed:" in result.stderr
    assert "Traceback" not in result.stderr


def test_run_call_with_import_registers_and_invokes_a_custom_function(tmp_path):
    module_path = tmp_path / "my_pydc_funcs.py"
    module_path.write_text("def cube(x):\n    return x ** 3\n")

    env = {"PYTHONPATH": str(tmp_path)}
    result = _run(["run", "call", "cube", "--arg", "x=3", "--import", "my_pydc_funcs:cube:cube"], env=env)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == 27


def test_run_call_with_multiprocessing_backend():
    result = _run(["run", "call", "ADD", "--arg", "a=1", "--arg", "b=2", "--backend", "multiprocessing", "--max-workers", "2"])
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == 3


# -- pydc run mapreduce -------------------------------------------------


def test_run_mapreduce_builtin_operations(tmp_path):
    data_file = tmp_path / "words.json"
    data_file.write_text(json.dumps(["a", "b", "a", "c", "b", "c"]))

    result = _run(
        ["run", "mapreduce", "--data", str(data_file), "--map", "WORD_COUNT", "--reduce", "SUM", "--partitions", "2"]
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {"a": 2, "b": 2, "c": 2}


def test_run_mapreduce_missing_data_file_is_a_clean_error():
    result = _run(["run", "mapreduce", "--data", "/no/such/file.json", "--map", "WORD_COUNT", "--reduce", "SUM"])
    assert result.returncode == 1
    assert "Job failed:" in result.stderr
    assert "Traceback" not in result.stderr
