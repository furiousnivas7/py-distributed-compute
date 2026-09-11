# Performance Baseline

Phase 12.5.5. Not a maximum-performance claim — the goal is a
reproducible number future changes can be compared against. Reproduce
with:

```bash
python scripts/benchmark.py
```

## Environment this baseline was measured on

- macOS 15.7.9, Apple M2, 8 cores
- Python 3.10.0
- localhost TCP (127.0.0.1) — no real network latency
- `DirectBackend` unless noted

## Results

### RPC round-trip overhead (PING, n=200)

| mean | median | p95 | min | max |
|---|---|---|---|---|
| 0.164ms | 0.173ms | 0.221ms | 0.101ms | 0.260ms |

Sub-millisecond — this is the actual wire-protocol/TCP-on-loopback cost,
unaffected by any polling loop.

### Worker registration time (n=20)

| mean | median | p95 | min | max |
|---|---|---|---|---|
| 11.056ms | 11.071ms | 11.168ms | 10.697ms | 11.168ms |

### Task submission → result latency, DirectBackend, 1 worker (n=100)

| mean | median | p95 | min | max |
|---|---|---|---|---|
| 11.180ms | 11.240ms | 11.520ms | 10.203ms | 11.794ms |

**Both of these cluster tightly around ~11ms, well above the ~0.16ms PING
number — this is not RPC overhead.** `wait_for_workers`/`wait_for_tasks`
poll on a default `poll_interval=0.01` (10ms) rather than being purely
event-driven end to end, so both numbers are dominated by that polling
granularity, not actual network/scheduling cost. A caller with tighter
latency requirements can pass a smaller `poll_interval` to
`wait_for_workers`; `wait_for_tasks` itself resolves via a per-task
`asyncio.Future` (event-driven, no polling) once the dispatcher is
already running, but the FIRST call that starts the dispatcher still
picks up a task via `dispatcher_loop`'s own poll cycle. This is a real,
identified inefficiency worth tracking, not a measurement artifact — see
`master/async_server.py`'s `dispatcher_loop`.

### Throughput: `DirectBackend`, workers scaling (200 tasks/run)

| workers | tasks/sec |
|---|---|
| 1 | 88.8 |
| 2 | 176.8 |
| 4 | 349.8 |
| 8 | 691.0 |

Scales close to linearly with worker count on this workload (trivial
`x + 1` tasks, so scheduling/dispatch overhead dominates over actual
execution time — see the stress-test baseline below for a similar
result: ~880 tasks/sec with `DirectBackend` and 10 workers on 100 tasks,
`tests/test_stress_concurrency.py`).

### `DirectBackend` vs `MultiprocessingBackend` (4 workers, 200 tasks)

| backend | tasks/sec |
|---|---|
| `DirectBackend` | 350.7 |
| `MultiprocessingBackend` (1 pool worker each) | 343.1 |

Comparable for this trivial workload — expected: `MultiprocessingBackend`'s
advantage is not blocking the event loop during a **genuinely CPU-bound**
task (see `examples/05_multiprocessing_backend.py`), not raw throughput
on cheap operations, where process-pool dispatch overhead is pure
overhead with nothing to hide behind.

## What this baseline does NOT cover

Heartbeat overhead in isolation (hard to measure without also measuring
whatever else is happening on the same connection — heartbeats are
fire-and-forget and don't block anything they share a connection with,
so their cost is effectively "one extra small message every
`heartbeat_interval` seconds," not something separately benchmarked
here), and multi-machine network latency (everything above is
loopback-only).
