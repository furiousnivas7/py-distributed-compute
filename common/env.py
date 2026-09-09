"""Phase 12.3: environment variable NAMES shared between the master and
worker CLIs -- lives in `common` (the one package both already sit above
conceptually, via common.models) specifically so `master` and `worker`
don't need a new direct import of each other just to agree on a string.
A worker uses PY_DISTRIBUTED_MASTER_HOST/PORT to know where to CONNECT;
a master started via its own CLI uses the same two variables to know
where to BIND -- one pair of env vars for "where the master is," useful
either way in a typical deployment that sets them once.
"""

ENV_MASTER_HOST = "PY_DISTRIBUTED_MASTER_HOST"
ENV_MASTER_PORT = "PY_DISTRIBUTED_MASTER_PORT"
