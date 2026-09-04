import socket
import threading

import pytest

from master import rpc_handler, server as master_server
from master.worker_manager import WorkerManager
from rpc import protocol
from worker import worker


@pytest.fixture(autouse=True)
def reset_worker_registry():
    """Each test registers 'worker-1' independently; the registry is a
    module-level singleton in rpc_handler, so it must be reset per test."""
    rpc_handler.worker_manager = WorkerManager()
    yield


def start_server_socket() -> socket.socket:
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(("127.0.0.1", 0))
    server_sock.listen(1)
    return server_sock


def test_task_dispatched_and_executed_over_real_tcp():
    server_sock = start_server_socket()
    port = server_sock.getsockname()[1]

    worker_thread = threading.Thread(target=worker.run_worker, args=("127.0.0.1", port), daemon=True)
    worker_thread.start()

    try:
        conn, worker_id = master_server.accept_and_register(server_sock)
        assert worker_id == worker.WORKER_ID

        response = master_server.dispatch_task(conn, "task-1", "ADD", {"a": 10, "b": 20})

        assert response["type"] == protocol.TASK_RESULT
        assert response["payload"] == {"task_id": "task-1", "status": "success", "result": 30}
    finally:
        conn.close()
        server_sock.close()
        worker_thread.join(timeout=2)

    assert not worker_thread.is_alive()


def test_multiply_task_dispatched_and_executed():
    server_sock = start_server_socket()
    port = server_sock.getsockname()[1]

    worker_thread = threading.Thread(target=worker.run_worker, args=("127.0.0.1", port), daemon=True)
    worker_thread.start()

    try:
        conn, _ = master_server.accept_and_register(server_sock)
        response = master_server.dispatch_task(conn, "task-2", "MULTIPLY", {"a": 4, "b": 5})

        assert response["payload"] == {"task_id": "task-2", "status": "success", "result": 20}
    finally:
        conn.close()
        server_sock.close()
        worker_thread.join(timeout=2)


def test_task_execution_failure_is_reported():
    server_sock = start_server_socket()
    port = server_sock.getsockname()[1]

    worker_thread = threading.Thread(target=worker.run_worker, args=("127.0.0.1", port), daemon=True)
    worker_thread.start()

    try:
        conn, _ = master_server.accept_and_register(server_sock)
        response = master_server.dispatch_task(conn, "task-3", "ADD", {"a": "x", "b": 1})

        assert response["payload"]["task_id"] == "task-3"
        assert response["payload"]["status"] == "error"
        assert "message" in response["payload"]
    finally:
        conn.close()
        server_sock.close()
        worker_thread.join(timeout=2)
