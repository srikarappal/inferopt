#!/usr/bin/env bash
#
# Run all three search methods on one model and put them side by side.
#
#   ./run_opt_ladder.sh                        # Qwen3-1.7B, the default
#   ./run_opt_ladder.sh Qwen/Qwen3-14B         # any model
#   MODEL=Qwen/Qwen3-14B TAG=14b ./run_opt_ladder.sh
#
# RESUMABLE. A method whose result.json already exists is skipped, so a run
# killed in the third method costs nothing to pick up. Delete that method's
# directory to force it again.
#
# The three methods search the SAME space -- same model, trace, SLO and DAG --
# because compare.py refuses to tabulate runs whose provenance stamps disagree,
# and it is right to: goodput counts only SLO-satisfying requests, so two runs
# with different SLOs produce numbers that look comparable and are not.
#
# WHAT IS BEING COMPARED
#
#   seqDAG   the greedy walk. Cheapest real search: one launch per factor,
#            decided against the incumbent, never revisited.
#   yolo     everything on versus nothing on. Cheapest of all, and the right
#            answer whenever every factor helps -- which is the question.
#   pbDAG    Plackett-Burman screen, then a factorial over what survives.
#            Costs the most and is the only one that measures each factor
#            against more than one background.
#
# Cost is part of the result. A method that wins by spending triple has not won,
# so compare.py prints launches beside goodput.

set -uo pipefail
cd "$(dirname "$0")"

MODEL=${MODEL:-${1:-Qwen/Qwen3-1.7B}}
TAG=${TAG:-$(echo "$MODEL" | sed 's|.*/||; s|Qwen3[.-]*||; s|-Instruct||' | tr 'A-Z' 'a-z')}
TRACE=${TRACE:-data/trace_shared.jsonl}
TTFT=${TTFT:-500}
ITL=${ITL:-250}
QPS=${QPS:-16}
GPU=${GPU:-0}
PY=${PYTHON:-python}

SEQ=runs/${TAG}-seqdag
YOLO=runs/${TAG}-yolo
PB=runs/${TAG}-pb
BASE=${BASE:-runs/${TAG}-baseline}

export CUDA_VISIBLE_DEVICES=$GPU

echo
echo "  model     $MODEL"
echo "  trace     $TRACE"
echo "  slo       ttft_p99 ${TTFT}ms  itl_p99 ${ITL}ms   qps $QPS"
echo "  gpu       $CUDA_VISIBLE_DEVICES"
echo "  dirs      $SEQ  $YOLO  $PB"
echo "  baseline  $BASE"
[ -f "$BASE/eval.json" ] || cat <<MSG

  NOTE: no $BASE/eval.json. The comparison still runs, but with no stock
  reference to read the three methods against. To produce one:

    $PY eval_repro.py --model $MODEL --benchmark math_500 --n 500 --repeats 3 \\
        --trace $TRACE --serving-concurrency 30 --run-dir $BASE
MSG
echo

# One method. A failure here is reported and does NOT stop the others: the
# comparison of two methods is still worth having, and re-running the two that
# already succeeded to get at the third would cost hours.
step() {
    local name=$1 dir=$2; shift 2
    if [ -f "$dir/result.json" ]; then
        echo "  == $name: already done ($dir), skipping"
        return 0
    fi
    echo
    echo "  ================================================================"
    echo "  == $name  ->  $dir"
    echo "  ================================================================"
    mkdir -p "$dir"
    if "$@" 2>&1 | tee "$dir/stdout.log"; then
        echo "  == $name finished"
    else
        echo "  == $name FAILED (see $dir/stdout.log); continuing with the rest"
    fi
}

# Order is cheapest first, so a short run still produces something comparable.
step "yolo   (4 launches)" "$YOLO" \
    $PY yolo_run.py --model "$MODEL" --trace "$TRACE" \
        --ttft-p99 "$TTFT" --itl-p99 "$ITL" --qps "$QPS" \
        --repeats 2 --run-dir "$YOLO"

# --quality-every-node scores the benchmark on every config rather than
# inheriting the baseline's score across lossless nodes. Slower, and required
# here: the comparison's claim is about the configs each method shipped, and an
# inherited score is an assumption, not a measurement.
step "seqDAG (~7 launches)" "$SEQ" \
    $PY run.py optimize --model "$MODEL" --trace "$TRACE" \
        --ttft-p99 "$TTFT" --itl-p99 "$ITL" --qps "$QPS" \
        --lossless-only --quality-every-node --run-dir "$SEQ"

# 12 screening rows plus 2^3 for the factorial. --repeats 1 leaves the full
# across-launch spread on each row mean, so the survivors are chosen by Lenth's
# margin rather than by group spread; raise it to 2 if nothing resolves.
step "pbDAG  (20 launches)" "$PB" \
    $PY pb_screen.py --model "$MODEL" --trace "$TRACE" \
        --ttft-p99 "$TTFT" --itl-p99 "$ITL" --qps "$QPS" \
        --survivors 3 --run-dir "$PB"

echo
echo "  ================================================================"
echo "  == comparison"
echo "  ================================================================"
DIRS=""
for d in "$SEQ" "$YOLO" "$PB"; do
    [ -f "$d/result.json" ] && DIRS="$DIRS $d"
done
if [ -z "$DIRS" ]; then
    echo "  no method produced a result.json; nothing to compare"
    exit 1
fi
ARGS=""
[ -f "$BASE/eval.json" ] && ARGS="--baseline $BASE"
$PY compare.py $DIRS $ARGS --plot "runs/${TAG}-frontier.png" 2>&1 \
    | tee "runs/${TAG}-comparison.txt"
echo
echo "  wrote runs/${TAG}-comparison.txt and runs/${TAG}-frontier.png"
echo
