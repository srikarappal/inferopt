#!/usr/bin/env bash
#
# Quantize Qwen3-14B every way we can produce, benchmarking each one IMMEDIATELY
# after it is built rather than converting everything first.
#
#   ./quantize_and_eval_interleaved.sh
#
# The interleaving is the point. quantize_and_eval.sh converts all four variants
# and then evaluates all four, so the first accuracy number arrives about four
# hours in, and a systematic problem -- every quantized variant scoring 0.2, say
# -- costs four conversions before it becomes visible. Convert-then-evaluate
# surfaces that after the first variant, and the table is printed as it grows so
# the comparison can be read while the rest is still running.
#
# Safe to run against a directory the other script has already partly filled:
# anything finished is skipped, so this picks up where that left off.
#
# Nothing is gated on effective_bits feasibility. A budget below the model's
# floor is produced AT the floor and labelled, because the point is to measure
# what comes out, not to argue in advance about what should.
#
# One step failing does not abort the rest -- a dead conversion still leaves the
# others comparable, and status.tsv says which are missing.

set -uo pipefail

MODEL="Qwen/Qwen3-14B"
TRACE="data/trace_shared.jsonl"
BENCH="math_500"
REPEATS=3
OUT="runs/quantize"
LOGS="$OUT/logs"
SAFE=${MODEL//\//__}

mkdir -p "$LOGS"
STATUS="$OUT/status.tsv"
[ -f "$STATUS" ] || : > "$STATUS"

note()   { printf '\n=== %s ===\n' "$*"; }
record() { printf '%s\t%s\t%s\t%s\n' "$(date +%H:%M)" "$1" "$2" "$3" >> "$STATUS"; }

# Refuse to run alongside the other script. Two processes converting into the
# same artifacts/ and serving on the same GPU would corrupt each other's work and
# make every measurement contended.
if pgrep -f "quantize_and_eval\.sh" > /dev/null 2>&1; then
    echo "  quantize_and_eval.sh is still running."
    echo "  Kill it first -- both write to $OUT and both need the whole GPU."
    exit 1
fi

# fp8 is absent from this list on purpose: vLLM quantizes bf16 weights during
# model load, so it is a launch flag with no checkpoint. It is handled with the
# configs at the bottom.
KINDS=(nvfp4 w4a16 autoquant@6.0 autoquant@5.0)

for K in "${KINDS[@]}"; do
    TAG="${K/@/_}"
    DIR="artifacts/${SAFE}--${TAG}"

    # ---- convert ----
    # hf_quant_config.json is written last, so its presence means the export
    # finished. A directory alone can be a conversion that died mid-write, and
    # that is exactly what a half-done run leaves behind.
    if [ -f "$DIR/hf_quant_config.json" ]; then
        note "$K already built, skipping conversion"
        record produce "$K" cached
    else
        note "converting $K"
        rm -rf "$DIR"
        if python quantize.py --model "$MODEL" --trace "$TRACE" --produce "$K" \
                2>&1 | tee "$LOGS/produce-$TAG.log"; then
            record produce "$K" ok
        else
            record produce "$K" FAILED
            echo "  $K conversion failed -- see $LOGS/produce-$TAG.log"
            continue
        fi
    fi

    # ---- evaluate immediately ----
    if [ -f "$OUT/q_$TAG/eval.json" ]; then
        note "$K already evaluated, skipping"
        record eval "$K" cached
    else
        note "evaluating $K"
        if python eval_repro.py --model "$DIR" --trace "$TRACE" \
                --benchmark "$BENCH" --repeats "$REPEATS" \
                --run-dir "$OUT/q_$TAG" 2>&1 | tee "$LOGS/eval-$TAG.log"; then
            record eval "$K" ok
        else
            record eval "$K" FAILED
        fi
    fi

    note "results so far"
    python summarize.py || true
done

# The two variants with no artifact.
#   stock  the bf16 reference, measured by the SAME code as everything else
#          rather than quoted from an earlier run
#   fp8    quantization applied at model load
for C in stock fp8; do
    if [ -f "$OUT/q_$C/eval.json" ]; then
        note "$C already evaluated, skipping"
        record eval "$C" cached
        continue
    fi
    note "evaluating $C (config, no artifact)"
    if python eval_repro.py --model "$MODEL" --trace "$TRACE" \
            --config "configs/$C.json" --benchmark "$BENCH" --repeats "$REPEATS" \
            --run-dir "$OUT/q_$C" 2>&1 | tee "$LOGS/eval-$C.log"; then
        record eval "$C" ok
    else
        record eval "$C" FAILED
    fi
    note "results so far"
    python summarize.py || true
done

note "final"
python summarize.py
echo
echo "  per-step status: $STATUS"
echo "  logs:            $LOGS/"
