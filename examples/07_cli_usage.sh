#!/usr/bin/env bash
# Example: using the pydc CLI directly, no Python code needed.
#
# This is illustrative, not auto-run by tests/test_examples.py (starting
# a real master process and leaving it running isn't something a test
# suite should do to your shell) -- the same commands are already
# exercised as real subprocess tests in tests/test_cli.py.
#
# Requires the project installed: pip install -e .[dev]

set -euo pipefail

echo "--- one-shot: run a single call, no separate processes needed ---"
pydc run call ADD --arg a=10 --arg b=32

echo
echo "--- one-shot: a full MapReduce job ---"
echo '["apple", "banana", "apple", "orange", "banana", "apple"]' > /tmp/pydc_example_words.json
pydc run mapreduce --data /tmp/pydc_example_words.json --map WORD_COUNT --reduce SUM --partitions 2
rm -f /tmp/pydc_example_words.json

echo
echo "--- one-shot: a custom function via --import ---"
cat > /tmp/pydc_example_funcs.py <<'PYEOF'
def cube(x):
    return x ** 3
PYEOF
PYTHONPATH=/tmp pydc run call cube --arg x=3 --import pydc_example_funcs:cube:cube
rm -f /tmp/pydc_example_funcs.py

echo
echo "--- real multi-process cluster (run these in separate terminals) ---"
echo "  terminal 1:  pydc master"
echo "  terminal 2:  pydc worker"
echo "  terminal 3:  pydc worker --worker-id worker-2   # a second worker"
echo
echo "  There is currently no CLI command that submits a job to that"
echo "  already-running master from a separate process -- 'pydc run'"
echo "  (above) always starts its OWN in-process master and worker."
echo "  See docs/api.md's 'pydc CLI' section for why."
