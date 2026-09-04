import pytest

from common.models import WorkerStatus
from master.worker_manager import DuplicateWorkerError, WorkerManager, WorkerNotFoundError


@pytest.fixture
def manager():
    return WorkerManager()


def test_register_new_worker(manager):
    worker = manager.register_worker("worker-1", "127.0.0.1", 6001)
    assert worker.worker_id == "worker-1"
    assert worker.host == "127.0.0.1"
    assert worker.port == 6001
    assert worker.status == WorkerStatus.IDLE


def test_get_worker(manager):
    manager.register_worker("worker-1", "127.0.0.1", 6001)
    worker = manager.get_worker("worker-1")
    assert worker is not None
    assert worker.worker_id == "worker-1"


def test_get_worker_missing_returns_none(manager):
    assert manager.get_worker("missing") is None


def test_register_duplicate_id_raises(manager):
    manager.register_worker("worker-1", "127.0.0.1", 6001)
    with pytest.raises(DuplicateWorkerError):
        manager.register_worker("worker-1", "127.0.0.1", 6002)


def test_list_workers(manager):
    manager.register_worker("worker-1", "127.0.0.1", 6001)
    manager.register_worker("worker-2", "127.0.0.1", 6002)
    workers = manager.get_all_workers()
    assert {w.worker_id for w in workers} == {"worker-1", "worker-2"}


def test_update_status(manager):
    manager.register_worker("worker-1", "127.0.0.1", 6001)
    manager.update_status("worker-1", WorkerStatus.BUSY)
    assert manager.get_worker("worker-1").status == WorkerStatus.BUSY


def test_update_status_missing_worker_raises(manager):
    with pytest.raises(WorkerNotFoundError):
        manager.update_status("missing", WorkerStatus.BUSY)


def test_update_status_invalid_status_raises(manager):
    manager.register_worker("worker-1", "127.0.0.1", 6001)
    with pytest.raises(ValueError):
        manager.update_status("worker-1", "NOT_A_REAL_STATUS")


def test_remove_worker(manager):
    manager.register_worker("worker-1", "127.0.0.1", 6001)
    manager.remove_worker("worker-1")
    assert manager.get_worker("worker-1") is None


def test_remove_missing_worker_raises(manager):
    with pytest.raises(WorkerNotFoundError):
        manager.remove_worker("missing")


def test_has_worker(manager):
    assert manager.has_worker("worker-1") is False
    manager.register_worker("worker-1", "127.0.0.1", 6001)
    assert manager.has_worker("worker-1") is True


@pytest.mark.parametrize(
    "worker_id, host, port",
    [
        ("", "127.0.0.1", 6001),
        ("worker-1", "", 6001),
        ("worker-1", "127.0.0.1", 0),
        ("worker-1", "127.0.0.1", 70000),
        ("worker-1", "127.0.0.1", "6001"),
        (None, "127.0.0.1", 6001),
    ],
)
def test_register_invalid_worker_info_raises(manager, worker_id, host, port):
    with pytest.raises(ValueError):
        manager.register_worker(worker_id, host, port)


def test_full_lifecycle_scenario(manager):
    manager.register_worker("worker-1", "127.0.0.1", 6001)
    manager.register_worker("worker-2", "127.0.0.1", 6002)
    assert {w.worker_id for w in manager.get_all_workers()} == {"worker-1", "worker-2"}

    manager.update_status("worker-1", WorkerStatus.BUSY)
    assert manager.get_worker("worker-1").status == WorkerStatus.BUSY

    manager.update_status("worker-1", WorkerStatus.IDLE)
    assert manager.get_worker("worker-1").status == WorkerStatus.IDLE

    manager.remove_worker("worker-2")
    assert [w.worker_id for w in manager.get_all_workers()] == ["worker-1"]
