import socket
import threading

import pytest

from master import rpc_handler, server as master_server
from rpc import protocol
from worker import worker


@pytest.fixture(autouse=True)
def reset_master_state():
    """rpc_handler.worker_manager and master_server.scheduler are module-level
    singletons, and server.py binds scheduler = Scheduler(rpc_handler.worker_manager)
    once at import time. Clear their state in place rather than replacing the
    objects — replacing rpc_handler.worker_manager would desync it from
    master_server.scheduler.worker_manager, which still points at the old one."""
    rpc_handler.worker_manager.clear()
    master_server.scheduler.clear()
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


def test_scheduler_queues_tasks_across_two_workers():
    """2 workers + 4 tasks: task-1 -> worker-1, task-2 -> worker-2, task-3/4
    stay PENDING until a worker frees up, then get dispatched too."""
    server_sock = start_server_socket()
    port = server_sock.getsockname()[1]

    worker_threads = [
        threading.Thread(
            target=worker.run_worker,
            args=("127.0.0.1", port),
            kwargs={"worker_id": "worker-1", "worker_port": 6001},
            daemon=True,
        ),
        threading.Thread(
            target=worker.run_worker,
            args=("127.0.0.1", port),
            kwargs={"worker_id": "worker-2", "worker_port": 6002},
            daemon=True,
        ),
    ]
    for thread in worker_threads:
        thread.start()

    connections = {}

    try:
        for _ in range(2):
            conn, worker_id = master_server.accept_and_register(server_sock)
            connections[worker_id] = conn

        assert set(connections) == {"worker-1", "worker-2"}

        for task_id, task_type, payload in [
            ("task-1", "ADD", {"a": 1, "b": 1}),
            ("task-2", "ADD", {"a": 2, "b": 2}),
            ("task-3", "ADD", {"a": 3, "b": 3}),
            ("task-4", "ADD", {"a": 4, "b": 4}),
        ]:
            master_server.scheduler.submit_task(task_id, task_type, payload)

        # Pure in-memory assignment: exactly one round of IDLE workers gets used.
        first = master_server.scheduler.assign_next_pending_task()
        second = master_server.scheduler.assign_next_pending_task()
        third = master_server.scheduler.assign_next_pending_task()

        assert {first.task_id, second.task_id} == {"task-1", "task-2"}
        assert {first.assigned_worker_id, second.assigned_worker_id} == {"worker-1", "worker-2"}
        assert third is None
        assert master_server.scheduler.get_task("task-3").status == "PENDING"
        assert master_server.scheduler.get_task("task-4").status == "PENDING"

        # Dispatch the two assigned tasks over the wire; each worker frees up after.
        master_server.dispatch_assigned_task(connections, first)
        master_server.dispatch_assigned_task(connections, second)

        assert master_server.scheduler.get_task(first.task_id).status == "COMPLETED"
        assert master_server.scheduler.get_task(second.task_id).status == "COMPLETED"

        # Now that workers are free again, drain the rest of the queue.
        master_server.drain_pending_tasks(connections)

        final_statuses = {t.task_id: t.status for t in master_server.scheduler.get_all_tasks()}
        assert final_statuses == {
            "task-1": "COMPLETED",
            "task-2": "COMPLETED",
            "task-3": "COMPLETED",
            "task-4": "COMPLETED",
        }
        for w in master_server.worker_manager.get_all_workers():
            assert w.status == "IDLE"
    finally:
        for conn in connections.values():
            conn.close()
        server_sock.close()
        for thread in worker_threads:
            thread.join(timeout=2)
