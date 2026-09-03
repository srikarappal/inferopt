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
