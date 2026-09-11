# Security Boundaries

Phase 12.5.8. This is not a full security system, and Phase 12.5
deliberately does not add one (no authentication, no TLS, no
authorization — see the project README's "Future Improvements"). This
document exists to state the CURRENT boundaries explicitly, so a reader
can make an informed decision about where this engine is appropriate to
run, rather than assuming protections that don't exist.

## What this engine currently has none of

```text
❌ No authentication       -- any TCP client that can reach the master's
                               port can REGISTER as a worker, or connect
                               and speak the protocol
❌ No encryption            -- the wire protocol is plain JSON over TCP,
                               no TLS
❌ No authorization          -- no concept of "which caller is allowed to
                               submit which job" -- anything running in
                               the same process as the master can submit
                               any task type to any registered worker
❌ Workers execute submitted Python -- see "Two trust levels" below
❌ Serialized callables are trusted input -- see below
```

**Therefore: treat this system as suitable for a trusted network or a
development environment, not an exposed production cluster.** Every
master and worker in a deployment must be operated by the same trusted
party. Do not expose the master's port to an untrusted network, and do
not accept task submissions (which, in this engine's current
architecture, means running code in the same process as the master —
see `docs/architecture.md`'s note on there being no separate
client-submission protocol) from a party you don't fully trust.

## Two trust levels, by design

This engine has always drawn one real security line, established in
Phase 10 and unchanged since: **registered functions vs. serialized
callables.**

| | Registered function | Serialized callable |
|---|---|---|
| How it runs | `worker.register_function` — an operator explicitly imports and registers it | `cloudpickle`-serialized, shipped as part of the task payload |
| What a task controls | *which* registered function runs, *what arguments* it gets | the actual code that runs |
| Exposure | none beyond what the operator already chose to expose | remote code execution, by design |

A task submitted against the registered-function path can never make a
worker run code its operator didn't already choose to import — it can
only select a name and arguments. A serialized callable is different in
kind: it **is** the code. Submitting one is exactly as trusting as
handing that worker your own source code to run. This is intentional
and documented (see `worker/serialization.py`'s module docstring, and
the README's own "Security Considerations" section) — appropriate for a
cooperative cluster where every submitter is already trusted, not
something to accept from a submitter you don't trust.

## What's already verified in place

- **No secrets in logs.** `worker/backend.py`'s structured logging
  (Phase 11.6) deliberately logs only `task_type`, `execution_mode`,
  exception *type* names, and counts — never a task's payload, arguments,
  or result, and never `result["message"]` (which can echo back
  caller-supplied values). Verified directly:
  `tests/test_backend_diagnostics.py::test_logs_never_include_serialized_callable_bytes_or_payload_contents`,
  `test_execution_failure_log_never_includes_the_error_message_body`.
- **No callable-payload leakage.** The same tests above submit a
  serialized callable carrying a sentinel argument value and assert it
  never appears in any log record.
- **A reasonable serialized-payload size limit.**
  `worker/serialization.py`'s `MAX_SERIALIZED_CALLABLE_BYTES` (10 MiB) —
  added specifically so an accidentally-huge closure (e.g. one that
  captured a large object by mistake) fails fast with a clear error
  instead of silently ballooning memory/network usage.
- **No unsafe debug output.** CLI configuration errors
  (`worker/async_worker.py::main`, `master/async_server.py::main`) print
  a clean, actionable one-line message and exit — never a raw traceback
  that could incidentally include more context than intended (verified:
  `tests/test_backend_config.py`'s subprocess tests).

## What a future security phase would need to add

Not built, and explicitly out of scope for 12.5 (see the phase's own
"Do NOT add" list): authentication between workers and master, TLS for
the wire protocol, authorization/access control for task submission, and
if remote job submission is ever added (see `docs/architecture.md`'s
note on why `pydc run` is local-only today), that new client-facing
protocol would need its own trust boundary designed from the start, not
retrofitted.
