#!/usr/bin/env bash
#
# Every test, in sequence.
#
#   ./run_tests.sh
#
# SEQUENTIAL ON PURPOSE. These suites share one checkout: mbpp_score writes into
# it, mutation runs rewrite traverse.py in place, and validate_dag reads
# dag/llm.json. Running two at once produces failures that do not reproduce --
# it has already happened twice in this project, once corrupting a seven-config
# ladder and once reporting a phantom validator failure.
#
# No GPU. Roughly three minutes, most of it mbpp_plus actually executing
# generated code, which is the point of that check.

set -uo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}
rc=0

# NO BYTECODE CACHE. Python validates a .pyc by source mtime AND SIZE, both at
# one-second granularity. A mutation test that swaps `delta > budget` for
# `delta < budget` changes neither -- same length, written in the same second --
# so the mutant's .pyc is reused for the restored original and the next run
# reports failures the source cannot explain. That cost a full debugging cycle
# here: git said the file was clean, grep said the comparison was correct, and
# the tests still failed with the mutant's exact signature.
export PYTHONDONTWRITEBYTECODE=1
find . -name __pycache__ -type d -not -path "./.evalplus-pkgs/*" \
    -not -path "./.quant-pkgs/*" -exec rm -rf {} + 2>/dev/null || true

run() {
    printf '  %-26s ' "$1"
    shift
    if out=$("$@" 2>&1); then
        # Each suite reports differently; take the last line that actually says
        # something rather than assuming a fixed offset from the end.
        echo "$out" | grep -iE "checks passed|all checks|well-formed|best measured" \
            | tail -1 | sed 's/^ *//' || echo "ok"
    else
        echo "FAILED"
        echo "$out" | tail -20 | sed 's/^/      /'
        rc=1
    fi
}

echo
run "unit (DAG)"        $PY test_dag_unit.py
run "integration (DAG)" $PY test_dag_integration.py
run "selftest"          $PY selftest.py
run "validate_dag"      $PY validate_dag.py dag/llm.json
run "dryrun"            $PY dryrun.py
echo
[ $rc -eq 0 ] && echo "  all suites passed" || echo "  SOME SUITES FAILED"
exit $rc
