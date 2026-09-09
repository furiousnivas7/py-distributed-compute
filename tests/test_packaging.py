"""Phase 12.2: packaging and installability -- proves the project is a
real, installable Python package, not something that only works because
the repository happens to be the current working directory.

These tests actually build a wheel and create real, isolated virtualenvs
(never reusing this repo's own .venv, and never running from inside the
repo directory) -- session-scoped fixtures pay that cost once per test
run, not once per test. Expect this file alone to take significantly
longer than the rest of the suite; that cost is the whole point (an
import-only check could pass for the wrong reason -- e.g. the repo still
being on sys.path -- a real subprocess from a real temp directory with a
real isolated venv cannot).
"""

import shutil
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run(args, cwd=None, timeout=120):
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=timeout)


def _create_venv(path: Path) -> Path:
    venv.EnvBuilder(with_pip=True, clear=True).create(str(path))
    python = path / ("Scripts" if sys.platform == "win32" else "bin") / "python"
    # ensurepip's bundled pip (here: 21.2.3) predates PEP 660 editable-install
    # support entirely -- "pip install -e ." on a pyproject.toml-only project
    # fails on it with a stale "setup.py not found" error, unrelated to
    # anything about THIS project's packaging. Upgrading pip first is a
    # real prerequisite for installing this package, not a test-only
    # workaround -- worth documenting in README's Installation section too.
    upgrade = _run([str(python), "-m", "pip", "install", "--quiet", "--upgrade", "pip"], timeout=60)
    assert upgrade.returncode == 0, f"pip upgrade failed:\n{upgrade.stdout}\n{upgrade.stderr}"
    return python


@pytest.fixture(scope="session")
def built_wheel(tmp_path_factory):
    """Build the wheel ONCE for the whole test session -- python -m build
    into an isolated temp dist dir (never REPO_ROOT/dist, so this never
    interferes with a real release build the user might run separately).

    --no-isolation: by default `python -m build` creates a SEPARATE,
    throwaway virtualenv and pip-installs build-system.requires (here:
    setuptools>=61) into it for every single invocation -- expensive and
    (worse) network-dependent. sys.executable (this repo's own .venv)
    already has a new enough setuptools installed (see the dev-setup
    note in README's Installation section), so --no-isolation reuses it
    directly instead, cutting real build time and one whole source of
    environment churn a repeated build/test loop would otherwise pay for
    every iteration.
    """
    dist_dir = tmp_path_factory.mktemp("dist")
    result = _run(
        [sys.executable, "-m", "build", "--no-isolation", "--outdir", str(dist_dir)],
        cwd=str(REPO_ROOT),
        timeout=180,
    )
    assert result.returncode == 0, f"python -m build failed:\n{result.stdout}\n{result.stderr}"
    wheels = list(dist_dir.glob("*.whl"))
    assert len(wheels) == 1, f"expected exactly one wheel, found {wheels}"
    return wheels[0]


@pytest.fixture(scope="session")
def clean_installed_venv(tmp_path_factory, built_wheel):
    """A genuinely fresh venv (no editable install, no dev extra) with
    ONLY the built wheel installed -- proves the wheel's own declared
    runtime dependencies are sufficient on their own, and that nothing
    test-only (pytest) leaked into a plain runtime install."""
    venv_dir = tmp_path_factory.mktemp("clean_venv")
    python = _create_venv(venv_dir)
    result = _run([str(python), "-m", "pip", "install", "--quiet", str(built_wheel)], timeout=120)
    assert result.returncode == 0, f"wheel install failed:\n{result.stdout}\n{result.stderr}"
    return python


@pytest.fixture(scope="session")
def outside_repo_dir(tmp_path_factory):
    """A real directory outside the repository -- proves an import isn't
    silently succeeding just because the repo happens to be on sys.path
    (cwd, or a stray PYTHONPATH entry)."""
    return tmp_path_factory.mktemp("outside_repo")


# -- 12.2.2: build metadata -----------------------------------------------


def test_build_metadata():
    pyproject = (REPO_ROOT / "pyproject.toml").read_text()
    assert 'name = "py-distributed-compute"' in pyproject
    assert 'version = "0.1.0"' in pyproject
    assert 'requires-python = ">=3.10"' in pyproject
    assert 'build-backend = "setuptools.build_meta"' in pyproject


# -- 12.2.4: editable installation -----------------------------------------


def test_editable_install(tmp_path):
    """pip install -e . into a fresh venv succeeds and makes the public
    API importable -- run from the repo root (an editable install's
    whole point is developing IN the repo), but the resulting install
    must not depend on any prior state in an existing venv."""
    venv_dir = tmp_path / "venv"
    python = _create_venv(venv_dir)
    result = _run([str(python), "-m", "pip", "install", "--quiet", "-e", "."], cwd=str(REPO_ROOT), timeout=120)
    assert result.returncode == 0, f"editable install failed:\n{result.stdout}\n{result.stderr}"

    check = _run([str(python), "-c", "import master, worker, jobs, common; print('ok')"])
    assert check.returncode == 0, check.stderr
    assert "ok" in check.stdout


# -- 12.2.4 (continued): import outside the repository, PYTHONPATH-clean --


def test_import_from_outside_repo(clean_installed_venv, outside_repo_dir):
    """The critical negative control: PYTHONPATH explicitly cleared, cwd
    is a directory that has never heard of this repo. If this import only
    worked because of an accidental sys.path entry, clearing PYTHONPATH
    and moving cwd away from the repo is exactly what would break it."""
    import os

    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    result = subprocess.run(
        [str(clean_installed_venv), "-c", "import master, worker, jobs, common; print('ok')"],
        cwd=str(outside_repo_dir),
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert "ok" in result.stdout


# -- 12.2.5: wheel build/install -------------------------------------------


def test_wheel_build(built_wheel):
    assert built_wheel.exists()
    assert built_wheel.name.startswith("py_distributed_compute-0.1.0-")
    assert built_wheel.suffix == ".whl"


def test_wheel_install(clean_installed_venv):
    result = _run([str(clean_installed_venv), "-m", "pip", "show", "py-distributed-compute"])
    assert result.returncode == 0
    assert "Version: 0.1.0" in result.stdout


# -- 12.2.6: public API survives installation ------------------------------


def test_public_api_after_install(clean_installed_venv, outside_repo_dir):
    script = outside_repo_dir / "check_public_api.py"
    script.write_text(
        "from worker import DirectBackend, MultiprocessingBackend, ExecutionBackend, run_worker, register_function\n"
        "from master import scheduler, start_master, stop_master, wait_for_tasks, Scheduler, WorkerManager\n"
        "from jobs import submit_call, collect_call_result, run_map_reduce, ExecutionSpec\n"
        "from common import Task, TaskStatus, Worker, WorkerStatus\n"
        "assert issubclass(DirectBackend, ExecutionBackend)\n"
        "assert issubclass(MultiprocessingBackend, ExecutionBackend)\n"
        "print('public-api-ok')\n"
    )
    result = _run([str(clean_installed_venv), str(script)], cwd=str(outside_repo_dir))
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert "public-api-ok" in result.stdout


def test_functional_smoke_after_wheel_install(clean_installed_venv, outside_repo_dir):
    """Not just importable -- actually runs a task through master+worker
    using nothing but the wheel-installed package, from outside the repo."""
    script = outside_repo_dir / "run_smoke.py"
    script.write_text(
        "import asyncio\n"
        "from worker import DirectBackend, register_function, run_worker\n"
        "from master import scheduler, start_master, stop_master, wait_for_tasks\n"
        "from master import async_server\n"
        "from jobs import submit_call, collect_call_result\n"
        "\n"
        "register_function('double', lambda x: x * 2)\n"
        "\n"
        "async def main():\n"
        "    server = await start_master('127.0.0.1', 0)\n"
        "    host, port = server.sockets[0].getsockname()[:2]\n"
        "    worker_task = asyncio.create_task(run_worker(host, port, worker_id='w1', backend=DirectBackend()))\n"
        "    await async_server.wait_for_workers(1)\n"
        "    task = submit_call(scheduler, 't1', 'double', x=21)\n"
        "    [response] = await wait_for_tasks({task.task_id})\n"
        "    result = collect_call_result(response)\n"
        "    worker_task.cancel()\n"
        "    try:\n"
        "        await worker_task\n"
        "    except (asyncio.CancelledError, Exception):\n"
        "        pass\n"
        "    await stop_master()\n"
        "    return result\n"
        "\n"
        "result = asyncio.run(main())\n"
        "assert result == 42, result\n"
        "print('smoke-ok')\n"
    )
    result = _run([str(clean_installed_venv), str(script)], cwd=str(outside_repo_dir))
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert "smoke-ok" in result.stdout


# -- 12.2.2 (continued): dependency declaration -----------------------------


def test_runtime_dependency_declaration():
    pyproject = (REPO_ROOT / "pyproject.toml").read_text()
    assert 'dependencies = ["cloudpickle>=3.0"]' in pyproject


def test_dev_dependencies_not_runtime(clean_installed_venv):
    """pytest must NOT be present in a plain wheel (runtime-only) install
    -- if it were, that would mean it leaked in as a runtime dependency
    instead of staying in the dev extra."""
    result = _run([str(clean_installed_venv), "-m", "pip", "show", "pytest"])
    assert result.returncode != 0, "pytest must not be installed by the plain wheel"


def test_console_script_installed_and_runs(clean_installed_venv):
    """Phase 12.3: `pydc` (pyproject.toml's [project.scripts]) must be a
    real, runnable command after a plain wheel install -- not just
    importable as `python -c "import client"`."""
    pydc = clean_installed_venv.parent / "pydc"
    assert pydc.exists(), f"pydc console script not found next to {clean_installed_venv}"
    result = _run([str(pydc), "--version"])
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert result.stdout.strip()


def test_dev_extra_installs_pytest(tmp_path):
    """The other half of the same contract: `pip install .[dev]` (the
    documented dev-setup path) DOES pull in pytest."""
    venv_dir = tmp_path / "venv"
    python = _create_venv(venv_dir)
    result = _run(
        [str(python), "-m", "pip", "install", "--quiet", "-e", ".[dev]"], cwd=str(REPO_ROOT), timeout=120
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    show = _run([str(python), "-m", "pip", "show", "pytest"])
    assert show.returncode == 0, "pytest must be installed via the dev extra"
