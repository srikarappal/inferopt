#!/usr/bin/env bash
#
# Score every variant that ALREADY EXISTS on a given benchmark.
#
#   ./eval_ladder.sh                      # mbpp_plus, all 378, 3 repeats
#   ./eval_ladder.sh mbpp_plus 100 3      # faster, ranking-grade
#   ./eval_ladder.sh math_500 500 3       # re-run the reasoning ladder at n=500
#
# Separate from quantize_and_eval_interleaved.sh, which CONVERTS and evaluates.
# Everything is already converted, so this skips straight to measurement and a
# six-variant ladder costs hours rather than most of a day.
#
# It also discovers variants by looking at artifacts/ instead of hardcoding a
# list of kinds. The hardcoded list has already drifted: it names autoquant@5.0,
# but that budget is below this model's 5.1395 floor, so what was actually built
# is autoquant_5.15. A name-based script does not find it and silently rebuilds
# a 10.6GB artifact that is already on disk.
#
# Results go to runs/ladder-$BENCH/ so nothing overwrites the MATH-500 ladder in
# runs/quantize/.
#
# One variant failing does not abort the rest -- a dead eval still leaves the
# others comparable, and status.tsv says which are missing.

set -uo pipefail

BENCH="${1:-mbpp_plus}"
N="${2:-378}"
REPEATS="${3:-3}"

MODEL="Qwen/Qwen3-14B"
TRACE="data/trace_shared.jsonl"
OUT="runs/ladder-$BENCH"
LOGS="$OUT/logs"
SAFE=${MODEL//\//__}

mkdir -p "$LOGS"
STATUS="$OUT/status.tsv"
[ -f "$STATUS" ] || : > "$STATUS"

note()   { printf '\n=== %s ===\n' "$*"; }
record() { printf '%s\t%s\t%s\n' "$(date +%H:%M)" "$1" "$2" >> "$STATUS"; }

# Refuse to run beside anything else that wants the whole GPU.
for other in quantize_and_eval.sh quantize_and_eval_interleaved.sh; do
    if pgrep -f "$other" > /dev/null 2>&1; then
        echo "  $other is still running -- both need the whole GPU."; exit 1
    fi
done

# mbpp_plus executes generated code. Say so once, plainly, before it starts.
if [ "$BENCH" = "mbpp_plus" ]; then
    echo "  NOTE: mbpp_plus EXECUTES model-written Python in a subprocess"
    echo "        (mbpp_score.py, under evalplus rlimits and timeouts)."
    echo "        That guards against runaway code, not against an adversary."
fi

echo "  benchmark $BENCH   n=$N   repeats=$REPEATS   -> $OUT"

run_one() {   # name, then the eval_repro args
    local name="$1"; shift
    if [ -f "$OUT/$name/eval.json" ]; then
        # A cached row is only reusable if it is a RESULT. A score of exactly
        # 0.0000 is not one -- no working probe produces it, and this ladder has
        # already been poisoned once by rows cached from a run whose scorer was
        # broken. Re-running is cheap; reading a stale zero as a measurement is
        # what cost the last four hours.
        if python - "$OUT/$name/eval.json" <<'PY'
import json, sys
r = json.loads(open(sys.argv[1]).read())
res = next(iter(r["results"].values()))
commit = (r.get("environment", {}).get("inferopt_commit") or "")[:8]
if res["mean"] == 0.0:
    print(f"      cached score is 0.0000 (from {commit}) -- re-running, "
          f"that is a broken probe, not a result")
    sys.exit(1)
print(f"      reusing {res['mean']:.4f} from {commit}")
PY
        then
            note "$name already evaluated, skipping"; record "$name" cached; return
        fi
        rm -rf "$OUT/$name"
    fi
    note "evaluating $name"
    if python eval_repro.py --benchmark "$BENCH" --n "$N" --repeats "$REPEATS" \
            --trace "$TRACE" --run-dir "$OUT/$name" "$@" \
            2>&1 | tee "$LOGS/$name.log"; then
        record "$name" ok
    else
        record "$name" FAILED
    fi
    note "results so far"
    python summarize.py "$OUT" || true
}

# The three with no artifact, in the order that makes the ladder readable:
#
#   stock     bf16, vLLM defaults -- the number a user starts from
#   lossless  launch flags only, weights untouched. Quality CANNOT move here by
#             construction, so this row doubles as a check on the instrument: if
#             it moves, the eval drifted and every lossy delta below it is
#             suspect. On MATH-500 it was 4.0x goodput at zero accuracy cost.
#   fp8       the first row that rewrites weights, applied at model load
#
# lossless is not optional. Without it every quantized row gets compared against
# stock and is silently credited with the speedup that flags already delivered
# for free -- the whole lossless-before-lossy point of the project, inverted.
run_one q_stock    --model "$MODEL" --config configs/stock.json
run_one q_lossless --model "$MODEL" --config configs/lossless.json
run_one q_fp8      --model "$MODEL" --config configs/fp8.json

# Then whatever was actually built, in whatever precision it landed at.
for DIR in artifacts/${SAFE}--*; do
    [ -d "$DIR" ] || continue
    # hf_quant_config.json is written last, so a directory without it is a
    # conversion that died mid-write, not a variant to measure.
    [ -f "$DIR/hf_quant_config.json" ] || {
        echo "  skipping $(basename "$DIR") -- incomplete conversion"; continue; }
    run_one "q_$(basename "$DIR" | sed "s/^${SAFE}--//")" --model "$DIR"
done

note "final"
python summarize.py "$OUT"
echo
echo "  per-step status: $STATUS"
echo "  logs:            $LOGS/"
