# Release Decision: Phase 12.5

**Decision: `v0.2.0` — hardened engineering prototype, not `v1.0.0`.**

Phase 12.5's own instructions were explicit: "I recommend not
automatically calling it v1.0 merely because the phases are complete.
Use the test/stress/reliability results to decide." This is that
decision, made against the actual results, not the phase checklist.

## What earns a version bump

Everything in Phase 12.5 passed clean, and it's real signal, not just
"tests exist":

- 609 tests, all passing, including (new in 12.5) end-to-end system
  tests, concurrency/stress tests (100 tasks / 10 workers, no
  duplicates, no lost results, no leaked asyncio tasks), a systematic
  failure-injection matrix, and a resource-leak audit (repeated
  start/submit/shutdown cycles, checked against process/thread/file-
  descriptor baselines).
- A frozen public-API snapshot (`tests/test_api_stability.py`) — the
  surface from Phase 12.1 hasn't drifted.
- Packaging verified from a genuinely clean environment: wheel build,
  install, import, and a real task execution, outside the repo,
  `PYTHONPATH` cleared (Phase 12.2/12.3, reconfirmed here).
- A measured performance baseline exists (`docs/performance_baseline.md`)
  to compare future changes against.

This is a materially more solid engine than the `v0.1.0` this project
started Phase 12 at — worth marking.

## What blocks `v1.0.0` specifically

`v1.0.0` conventionally signals "stable, safe to depend on, we commit to
not breaking you." Three things are true today that make that
commitment premature:

1. **No security boundary beyond "trust everyone on the network"** — no
   authentication, no encryption, no authorization (`docs/security.md`).
   A `v1.0.0` engine that can't be safely exposed to anything but a
   fully trusted network is a significant, load-bearing caveat to put on
   a "1.0" release.
2. **No remote job submission.** `master.scheduler` is an in-process
   singleton; there is no client-facing wire-protocol message for
   submitting work to an already-running, separate master process.
   `pydc run` (local/in-process only) is the closest equivalent. For a
   project whose whole premise is "distributed computing," not being
   able to submit work from a separate client process is a real gap in
   the core promise, not a peripheral feature.
3. **State is entirely in-memory.** A master restart loses every task
   and worker record — no persistence. Combined with the two points
   above, "distributed compute ENGINE, v1.0" is a stronger claim than
   the current architecture supports.

None of these are defects — they're documented, deliberate scope
boundaries from earlier phases (see `docs/architecture.md`'s notes and
the README's "Future Improvements"). But a version number is a promise
to users, and `v1.0.0` would overstate what's actually safe to build on
top of today.

## What this project actually is, right now

A well-tested, well-documented distributed-computing **engine and
research/portfolio project**: master-worker architecture, a real custom
RPC protocol, MapReduce, pluggable execution backends, retry and fault
tolerance, a CLI, and packaging — all genuinely built and verified, not
sketched. Appropriate for a trusted development environment, a learning
project, or a foundation for further work. Not yet appropriate to expose
as a production service, or to depend on for API/wire-protocol stability
guarantees.

## What would justify `v1.0.0` later

At minimum: a documented security story (even a simple shared-secret
auth would meaningfully change the calculus), a real remote-submission
path, and a decision on state persistence. None of those are in scope
for this phase — see Phase 12.5's own "Do NOT add" list — but they're
the concrete bar for the next version decision, not vague "more
testing."
