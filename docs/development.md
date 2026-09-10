# Development Guide

Setting up, running, and testing this project locally. For what the
public API actually offers, see [`api.md`](api.md); for how the pieces
fit together, see [`architecture.md`](architecture.md).

## Setup

```bash
git clone <this-repository-url>
cd py-distributed-compute
python -m venv .venv
source .venv/bin/activate      # .venv\Scripts\activate on Windows
pip install --upgrade pip      # ensurepip's bundled pip is often too old
                                # for this project's pyproject.toml-only
                                # editable installs (PEP 660)
pip install -e .[dev]
```

`cloudpickle` is the only runtime dependency; `pytest` is the only dev
dependency (see `pyproject.toml` — audited against actual imports, not
copied blindly from any prior `requirements.txt`).

If you'll also run `python -m build` locally (`tests/test_packaging.py`
does this), install a new enough `setuptools` in the same venv:

```bash
pip install --upgrade "setuptools>=61.0" wheel build
```

## Running the test suite

```bash
pytest -q
```

As of Phase 12.3 this is **560+ tests** across ~60 files. A few things
worth knowing:

- No `pytest-asyncio` dependency — every async test drives its own
  `asyncio.run(...)` inside a plain `def test_...()` (see
  `tests/conftest.py`'s own docstring for why).
- `tests/test_packaging.py` and `tests/test_cli.py` build real wheels
  and spin up real subprocesses/venvs — noticeably slower than the rest
  of the suite (tens of seconds), and the only files where you might
  actually see filesystem contention from your OS's own file indexer
  (Spotlight on macOS, Search Indexer on Windows) if you run them in a
  tight loop. If you hit spurious `TimeoutError`s reading files that
  otherwise work fine, that's almost certainly what's happening — not a
  bug in the code being tested. Space out repeated full-suite runs, or
  target a narrower set of files, rather than looping the whole suite
  back-to-back.
- Run a single file/test the normal pytest way: `pytest -q
  tests/test_backend.py`, `pytest -q -k test_execute_registered_function`.

## Running things locally

Three terminals, using the `pydc` CLI (installed with the package):

```bash
# terminal 1
pydc master

# terminal 2
pydc worker

# terminal 3, once both are up
pydc run call ADD --arg a=10 --arg b=32
```

Or skip the separate processes entirely for a quick check:

```bash
pydc run call ADD --arg a=10 --arg b=32
pydc run mapreduce --data words.json --map WORD_COUNT --reduce SUM --partitions 4
```

See [`api.md`'s "pydc CLI" section](api.md#pydc-cli) for every command,
and the README's "Execution Backend Configuration" section for every
`--backend`/`--max-workers`/etc. flag and its `PY_DISTRIBUTED_*`
environment-variable equivalent.

## Running the examples

See [`../examples/README.md`](../examples/README.md) — each example is a
standalone, runnable `.py` file with no separate master/worker process
needed (they start their own in-process master + worker, exactly like
`pydc run` does). `tests/test_examples.py` runs every one of them as
part of the normal test suite, so they can't silently rot.

## Configuration reference

Every `PY_DISTRIBUTED_*` environment variable and its CLI equivalent is
documented in the README's "Execution Backend Configuration" section.
The short version: **CLI arguments > environment variables > defaults**,
resolved independently per field.

## Troubleshooting

**`pip install -e .` fails with "editable mode currently requires a
setuptools-based build"** — your venv's `pip` predates PEP 660. Run `pip
install --upgrade pip` first.

**A test hangs or a worker never connects** — check you're not pointing
two different processes at mismatched ports; `pydc worker`'s default
`--master-port` is `5000`, matching `pydc master`'s default `--port`.

**Spurious `TimeoutError`/filesystem errors during repeated test runs**
— see the "Running the test suite" note above; this is an OS file-indexer
interaction with the packaging tests' real venv/wheel creation, not a
product bug. Wait a bit and retry, or run a narrower set of test files.

**A registered function "isn't found" under `MultiprocessingBackend`**
— registration must happen in the process that actually executes the
task. With the default `mp_context="fork"`, register before
`backend.start()` and it's inherited "for free"; under `spawn`/
`forkserver`, register via code that naturally re-runs on import in the
fresh child interpreter (a module-level `@worker.register(...)`
decorator), not a one-off runtime call. See
`worker/backend.py`'s `MultiprocessingBackend` docstring for the full
explanation.

**A worker exits immediately with a `Configuration error:` message** —
that's a deliberately clean, actionable message (not a bug) for an
invalid `--backend`/`--max-workers`/etc. value; the message itself says
what was actually received.
