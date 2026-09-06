"""Project-wide pytest fixtures.

Phase 8.9 adds module-level state to master.async_server (the centralized
dispatcher's task_id-keyed response registry, and the dispatcher task
itself) alongside the pre-existing rpc_handler.worker_manager and
master.async_server.scheduler singletons. Every existing async test file
already resets those two locally; this fixture complements them globally
so none of those ~14 files need to be touched individually.

Without this, a stale, already-resolved response left over from an
earlier test reusing the same task_id (e.g. "job-1-map-0") could resolve a
later, unrelated test's wait_for_tasks() call immediately with the wrong
data instead of ever dispatching its actual task.

This is a plain (sync) fixture, not `async def`: the project deliberately
has no pytest-asyncio dependency (every test drives its own asyncio.run()
inside a plain `def test_...()`), and pytest can't await an async fixture
without that plugin. No async teardown is needed anyway -- asyncio.run()
already cancels every leftover task (including a dispatcher started
during the test) when it returns, and is_dispatcher_running()'s check
against the *current* running loop already treats a dispatcher task left
over from a previous (now-closed) loop as not running, so the next test's
ensure_dispatcher_running() naturally starts a fresh one.
"""

import pytest

from master import async_server


@pytest.fixture(autouse=True)
def _reset_dispatch_registry():
    async_server.clear_dispatch_registry()
    yield
