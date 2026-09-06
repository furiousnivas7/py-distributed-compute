import socket
import threading
import time

import pytest

from master import rpc_handler, server as master_server
from rpc import protocol
from rpc.connection import Connection
from worker import worker
from worker.executor import execute_task


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


def run_gated_worker(host: str, port: int, worker_id: str, ready_event: threading.Event, release_event: threading.Event) -> None:
    """Like worker.run_worker, but pauses right before executing a TASK until
    `release_event` is set, and signals `ready_event` once it's paused there.
    Lets a test observe a task mid-flight (status RUNNING) deterministically,
    without relying on sleep-based timing."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))
    conn = Connection(sock)

    try:
        worker.send_rpc(conn, protocol.PING)
        worker.register(conn, worker_id, "127.0.0.1", 6000)

        while True:
            try:
                raw = conn.recv_bytes()
            except ConnectionError:
                return

            request = protocol.decode_message(raw)
            if request["type"] != protocol.TASK:
                continue

            task_payload = request["payload"]
            ready_event.set()
            release_event.wait(timeout=5)

            result = execute_task(task_payload["task_type"], task_payload["task_payload"])
            response = protocol.build_message(
                protocol.TASK_RESULT,
                request["request_id"],
                {"task_id": task_payload["task_id"], **result},
            )
            conn.send_bytes(protocol.encode_message(response))
    finally:
        conn.close()


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


def test_dispatch_concurrently_runs_both_tasks_at_once():
    """Deterministic proof of concurrency: both tasks reach RUNNING and sit
    there simultaneously, gated on events rather than sleeps/timing."""
    server_sock = start_server_socket()
    port = server_sock.getsockname()[1]

    ready = {"worker-1": threading.Event(), "worker-2": threading.Event()}
    release = {"worker-1": threading.Event(), "worker-2": threading.Event()}

    worker_threads = [
        threading.Thread(
            target=run_gated_worker,
            args=("127.0.0.1", port, worker_id, ready[worker_id], release[worker_id]),
            daemon=True,
        )
        for worker_id in ("worker-1", "worker-2")
    ]
    for thread in worker_threads:
        thread.start()

    connections = {}

    try:
        for _ in range(2):
            conn, worker_id = master_server.accept_and_register(server_sock)
            connections[worker_id] = conn

        master_server.scheduler.submit_task("task-1", "ADD", {"a": 1, "b": 1})
        master_server.scheduler.submit_task("task-2", "ADD", {"a": 2, "b": 2})

        task1 = master_server.scheduler.assign_next_pending_task()
        task2 = master_server.scheduler.assign_next_pending_task()

        dispatch_thread = threading.Thread(
            target=master_server.dispatch_concurrently, args=(connections, [task1, task2])
        )
        dispatch_thread.start()

        assert ready["worker-1"].wait(timeout=5)
        assert ready["worker-2"].wait(timeout=5)

        # Both workers are now paused mid-task -> both tasks must be RUNNING
        # at the same time, which is only possible if dispatch didn't block
        # on worker-1 before ever contacting worker-2.
        assert master_server.scheduler.get_task("task-1").status == "RUNNING"
        assert master_server.scheduler.get_task("task-2").status == "RUNNING"

        release["worker-1"].set()
        release["worker-2"].set()
        dispatch_thread.join(timeout=5)

        assert master_server.scheduler.get_task("task-1").status == "COMPLETED"
        assert master_server.scheduler.get_task("task-2").status == "COMPLETED"
        for w in master_server.worker_manager.get_all_workers():
            assert w.status == "IDLE"
    finally:
        for conn in connections.values():
            conn.close()
        server_sock.close()
        for thread in worker_threads:
            thread.join(timeout=2)


def test_dispatch_concurrently_is_faster_than_sequential():
    """Sanity check: two workers that each take DELAY seconds finish in
    roughly DELAY seconds total when dispatched concurrently, not 2x DELAY."""
    server_sock = start_server_socket()
    port = server_sock.getsockname()[1]
    DELAY = 0.3

    worker_threads = [
        threading.Thread(target=worker.run_worker, args=("127.0.0.1", port), kwargs={"worker_id": wid}, daemon=True)
        for wid in ("worker-1", "worker-2")
    ]
    for thread in worker_threads:
        thread.start()

    connections = {}

    try:
        for _ in range(2):
            conn, worker_id = master_server.accept_and_register(server_sock)
            connections[worker_id] = conn

        # SLEEP isn't a real task type; ADD/MULTIPLY execute effectively
        # instantly, so the delay has to come from the worker side instead —
        # monkeypatch execute_task briefly to simulate slow work.
        import worker.worker as worker_module

        original_execute = worker_module.execute_task

        def slow_execute(task_type, payload):
            time.sleep(DELAY)
            return original_execute(task_type, payload)

        worker_module.execute_task = slow_execute
        try:
            master_server.scheduler.submit_task("task-1", "ADD", {"a": 1, "b": 1})
            master_server.scheduler.submit_task("task-2", "ADD", {"a": 2, "b": 2})

            task1 = master_server.scheduler.assign_next_pending_task()
            task2 = master_server.scheduler.assign_next_pending_task()

            start = time.monotonic()
            responses = master_server.dispatch_concurrently(connections, [task1, task2])
            elapsed = time.monotonic() - start
        finally:
            worker_module.execute_task = original_execute

        assert all(r["payload"]["status"] == "success" for r in responses)
        # Concurrent: ~DELAY total. Sequential would be ~2*DELAY. Threshold
        # sits well between the two with generous margin for scheduling jitter.
        assert elapsed < DELAY * 1.7
    finally:
        for conn in connections.values():
            conn.close()
        server_sock.close()
        for thread in worker_threads:
            thread.join(timeout=2)


def test_heartbeat_round_trip_over_real_tcp():
    """A HEARTBEAT is sent on its own short-lived connection (not the
    long-lived task-dispatch one) and updates the worker's last_heartbeat."""
    server_sock = start_server_socket()
    port = server_sock.getsockname()[1]

    worker_thread = threading.Thread(target=worker.run_worker, args=("127.0.0.1", port), daemon=True)
    worker_thread.start()

    conn, worker_id = master_server.accept_and_register(server_sock)

    try:
        before = master_server.worker_manager.get_worker(worker_id).last_heartbeat

        heartbeat_thread = threading.Thread(
            target=worker.send_heartbeat, args=("127.0.0.1", port, worker_id), daemon=True
        )
        heartbeat_thread.start()

        request, response = master_server.accept_and_handle_one(server_sock)
        heartbeat_thread.join(timeout=2)

        assert request["type"] == protocol.HEARTBEAT
        assert request["payload"] == {"worker_id": worker_id}
        assert response["type"] == protocol.HEARTBEAT_ACK
        assert response["payload"]["status"] == "success"

        after = master_server.worker_manager.get_worker(worker_id).last_heartbeat
        assert after >= before
    finally:
        conn.close()
        server_sock.close()
        worker_thread.join(timeout=2)


def test_heartbeat_for_unknown_worker_is_rejected():
    server_sock = start_server_socket()
    port = server_sock.getsockname()[1]

    heartbeat_thread = threading.Thread(
        target=worker.send_heartbeat, args=("127.0.0.1", port, "ghost-worker"), daemon=True
    )
    heartbeat_thread.start()

    try:
        request, response = master_server.accept_and_handle_one(server_sock)
        heartbeat_thread.join(timeout=2)

        assert request["type"] == protocol.HEARTBEAT
        assert response["type"] == protocol.ERROR
        assert response["payload"]["code"] == "UNKNOWN_WORKER"
    finally:
        server_sock.close()


def test_heartbeat_loop_sends_periodically():
    server_sock = start_server_socket()
    port = server_sock.getsockname()[1]

    worker_thread = threading.Thread(target=worker.run_worker, args=("127.0.0.1", port), daemon=True)
    worker_thread.start()
    conn, worker_id = master_server.accept_and_register(server_sock)

    stop_event = threading.Event()
    heartbeat_thread = threading.Thread(
        target=worker.start_heartbeat_loop,
        args=("127.0.0.1", port, worker_id, stop_event),
        kwargs={"interval": 0.05},
        daemon=True,
    )
    heartbeat_thread.start()

    try:
        seen = 0
        deadline = time.monotonic() + 2
        while seen < 3 and time.monotonic() < deadline:
            master_server.accept_and_handle_one(server_sock)
            seen += 1

        assert seen == 3
        assert master_server.worker_manager.get_worker(worker_id).last_heartbeat is not None
    finally:
        stop_event.set()
        conn.close()
        server_sock.close()
        worker_thread.join(timeout=2)


def test_heartbeat_listener_accepts_connections_on_dedicated_port():
    """heartbeat_listener runs on its own socket, separate from the one
    accept_and_register uses, so registration and heartbeats can't race."""
    server_sock = start_server_socket()
    port = server_sock.getsockname()[1]
    heartbeat_sock = start_server_socket()
    heartbeat_port = heartbeat_sock.getsockname()[1]

    worker_thread = threading.Thread(target=worker.run_worker, args=("127.0.0.1", port), daemon=True)
    worker_thread.start()
    conn, worker_id = master_server.accept_and_register(server_sock)

    stop_event = threading.Event()
    listener_thread = threading.Thread(
        target=master_server.heartbeat_listener, args=(heartbeat_sock, stop_event), daemon=True
    )
    listener_thread.start()

    try:
        for _ in range(3):
            response = worker.send_heartbeat("127.0.0.1", heartbeat_port, worker_id)
            assert response["type"] == protocol.HEARTBEAT_ACK

        assert master_server.worker_manager.get_worker(worker_id).last_heartbeat is not None
    finally:
        stop_event.set()
        listener_thread.join(timeout=2)
        conn.close()
        server_sock.close()
        heartbeat_sock.close()
        worker_thread.join(timeout=2)


def test_failure_monitor_detects_stale_worker(monkeypatch):
    """Test 4: start the background monitor, force a worker stale, confirm
    it flips to FAILED on its own, then stop the monitor."""
    monkeypatch.setattr(master_server, "FAILURE_CHECK_INTERVAL", 0.05)
    monkeypatch.setattr(master_server, "HEARTBEAT_TIMEOUT", 1.0)

    server_sock = start_server_socket()
    port = server_sock.getsockname()[1]

    worker_thread = threading.Thread(target=worker.run_worker, args=("127.0.0.1", port), daemon=True)
    worker_thread.start()
    conn, worker_id = master_server.accept_and_register(server_sock)

    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=master_server.failure_monitor, args=(stop_event,), daemon=True)

    try:
        assert master_server.worker_manager.get_worker(worker_id).status == "IDLE"

        # Force staleness directly instead of waiting out a real timeout.
        master_server.worker_manager.get_worker(worker_id).last_heartbeat = time.time() - 10

        monitor_thread.start()

        deadline = time.monotonic() + 2
        while (
            master_server.worker_manager.get_worker(worker_id).status != "FAILED"
            and time.monotonic() < deadline
        ):
            time.sleep(0.02)

        assert master_server.worker_manager.get_worker(worker_id).status == "FAILED"
    finally:
        stop_event.set()
        monitor_thread.join(timeout=2)
        conn.close()
        server_sock.close()
        worker_thread.join(timeout=2)
