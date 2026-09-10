"""Phase 12.4: runs every example in examples/ as a real subprocess and
checks its actual output -- proves they still work against the current
code, not just that they parse. examples/07_cli_usage.sh is deliberately
NOT run here (see its own header); the same commands it demonstrates are
already exercised as real subprocess tests in tests/test_cli.py.
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"


def _run_example(name: str, timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(EXAMPLES_DIR / name)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_01_registered_function():
    result = _run_example("01_registered_function.py")
    assert result.returncode == 0, result.stderr
    assert "double(21) = 42" in result.stdout


def test_02_serialized_callable():
    result = _run_example("02_serialized_callable.py")
    assert result.returncode == 0, result.stderr
    assert "scale(14) = 42" in result.stdout


def test_03_multiple_workers():
    result = _run_example("03_multiple_workers.py")
    assert result.returncode == 0, result.stderr
    lines = [line for line in result.stdout.splitlines() if line.startswith("add-")]
    assert len(lines) == 6
    workers_used = {line.split("ran on ")[-1] for line in lines}
    # All three workers should have run at least one task between them --
    # not a hard guarantee of the scheduler (round-robin isn't a
    # documented contract), but with 6 tasks and 3 idle workers this
    # should reliably happen and demonstrates the actual point of the
    # example (work IS spread across workers, not stuck on one).
    assert len(workers_used) >= 2


def test_04_map_reduce():
    result = _run_example("04_map_reduce.py")
    assert result.returncode == 0, result.stderr
    assert "Word counts: {'apple': 3, 'banana': 2, 'orange': 1}" in result.stdout


def test_05_multiprocessing_backend():
    result = _run_example("05_multiprocessing_backend.py")
    assert result.returncode == 0, result.stderr
    assert "slow_square(7) = 49" in result.stdout


def test_06_fault_tolerance():
    result = _run_example("06_fault_tolerance.py")
    assert result.returncode == 0, result.stderr
    assert "Result: 101 (attempts: 2, completed by: worker-" in result.stdout
    assert "status: FAILED" in result.stdout


def test_all_python_examples_are_covered_by_a_test():
    """Guards against a new example being added without a matching test
    above -- an example that's never run is exactly the kind of thing
    that silently rots."""
    covered = {"01_registered_function.py", "02_serialized_callable.py", "03_multiple_workers.py",
               "04_map_reduce.py", "05_multiprocessing_backend.py", "06_fault_tolerance.py"}
    actual = {p.name for p in EXAMPLES_DIR.glob("*.py")}
    assert actual == covered, f"examples/ changed but this test wasn't updated: {actual}"
