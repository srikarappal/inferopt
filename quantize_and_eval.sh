#!/usr/bin/env bash
#
# Quantize Qwen3-14B every way we can produce, then benchmark each on MATH-500
# and on the workload trace.
#
#   ./quantize_and_eval.sh
#
# Nothing is gated on effective_bits feasibility. A budget below the model's
# floor is produced AT the floor and labelled, because the point is to measure
# what comes out, not to argue in advance about what should.
#
# One step failing does not abort the rest -- a conversion that dies still leaves
# the other five comparable, and the summary at the end says which are missing.
# Each step logs to runs/quantize/logs/ so a failure is diagnosable afterwards.

set -uo pipefail

MODEL="Qwen/Qwen3-14B"
TRACE="data/trace_shared.jsonl"
BENCH="math_500"
REPEATS=3
OUT="runs/quantize"
LOGS="$OUT/logs"
SAFE=${MODEL//\//__}          # Qwen__Qwen3-14B, matching artifacts/ naming

mkdir -p "$LOGS"
STATUS="$OUT/status.tsv"
: > "$STATUS"

note() { printf '\n=== %s ===\n' "$*"; }
record() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" >> "$STATUS"; }

# --------------------------------------------------------------------------
# 1. produce the artifacts
#
# fp8 is deliberately absent: vLLM quantizes bf16 weights during model load, so
# it is a launch flag with no checkpoint. It is evaluated below via configs/.
# --------------------------------------------------------------------------
KINDS=(nvfp4 w4a16 autoquant@6.0 autoquant@5.0)

for K in "${KINDS[@]}"; do
    DIR="artifacts/${SAFE}--${K/@/_}"
    if [ -d "$DIR" ]; then
        note "$K already produced, skipping"     # ensure_variant caches too
        record produce "$K" cached
        continue
    fi
    note "producing $K"
    if python quantize.py --model "$MODEL" --trace "$TRACE" --produce "$K" \
            2>&1 | tee "$LOGS/produce-${K/@/_}.log"; then
        record produce "$K" ok
    else
        record produce "$K" FAILED
        echo "  $K failed -- see $LOGS/produce-${K/@/_}.log"
    fi
done

# --------------------------------------------------------------------------
# 2. evaluate every artifact
# --------------------------------------------------------------------------
for K in "${KINDS[@]}"; do
    DIR="artifacts/${SAFE}--${K/@/_}"
    TAG="${K/@/_}"
    [ -d "$DIR" ] || { record eval "$K" "no artifact"; continue; }
    note "evaluating $K"
    if python eval_repro.py --model "$DIR" --trace "$TRACE" \
            --benchmark "$BENCH" --repeats "$REPEATS" \
            --run-dir "$OUT/q_$TAG" 2>&1 | tee "$LOGS/eval-$TAG.log"; then
        record eval "$K" ok
    else
        record eval "$K" FAILED
    fi
done

# --------------------------------------------------------------------------
# 3. the two configs that have no artifact
#
#   stock  the bf16 reference, measured by the SAME code as everything else
#          rather than quoted from an earlier run
#   fp8    quantization applied at load time
# --------------------------------------------------------------------------
for C in stock fp8; do
    note "evaluating $C (config, no artifact)"
    if python eval_repro.py --model "$MODEL" --trace "$TRACE" \
            --config "configs/$C.json" --benchmark "$BENCH" --repeats "$REPEATS" \
            --run-dir "$OUT/q_$C" 2>&1 | tee "$LOGS/eval-$C.log"; then
        record eval "$C" ok
    else
        record eval "$C" FAILED
    fi
done

# --------------------------------------------------------------------------
# 4. one table
# --------------------------------------------------------------------------
note "summary"
python - <<'PY'
import json, math
from pathlib import Path

OUT = Path("runs/quantize")
ORDER = ["stock", "fp8", "autoquant_6.0", "autoquant_5.0", "w4a16", "nvfp4"]
DEMAND = 15.36 * 259          # tok/s of on-time output this workload needs

rows = []
for tag in ORDER:
    f = OUT / f"q_{tag}" / "eval.json"
    if not f.exists():
        rows.append((tag, None)); continue
    r = json.loads(f.read_text())
    res = next(iter(r["results"].values()))
    rows.append((tag, res))

hdr = (f"  {'variant':16s} {'accuracy':>9s} {'spread':>7s} {'goodput':>9s} "
       f"{'thru':>8s} {'ttft p99':>9s} {'slo':>5s} {'GB':>6s} {'replicas':>9s}")
print(hdr); print("  " + "-" * (len(hdr) - 2))
base = None
for tag, res in rows:
    if res is None:
        print(f"  {tag:16s}   (not produced or evaluation failed)"); continue
    sv = res.get("serving") or {}
    gp = sv.get("goodput")
    size = ""
    d = OUT.parent.parent / "artifacts"
    for p in Path("artifacts").glob(f"*--{tag}"):
        size = f"{sum(x.stat().st_size for x in p.rglob('*') if x.is_file())/1e9:.1f}"
    rep = f"{math.ceil(DEMAND / gp)}" if gp else "-"
    print(f"  {tag:16s} {res['mean']:9.4f} {res['spread']:7.4f} "
          f"{(gp or 0):9.1f} {sv.get('throughput', 0):8.1f} "
          f"{sv.get('ttft_p99_ms', 0):8.0f}m {sv.get('slo_attainment', 0):5.0%} "
          f"{size:>6s} {rep:>9s}")
    if tag == "stock":
        base = res

if base:
    print(f"\n  accuracy is scored on MATH-500. Its measured resolution limit on this")
    print(f"  rig is 0.02-0.03, so a gap smaller than ~3 points is not resolved by")
    print(f"  this measurement -- it is not evidence of damage, nor of safety.")
    print(f"  Baseline accuracy {base['mean']:.4f}, spread {base['spread']:.4f}.")

# Say plainly where a bit budget was not achievable.
for tag, _ in rows:
    j = Path("artifacts") / f"Qwen__Qwen3-14B--{tag}" / "autoquantize_search.json"
    if j.exists():
        a = json.loads(j.read_text())
        if a.get("clamped_to_model_floor"):
            print(f"\n  {tag}: requested {a['effective_bits_requested']} bits, produced at "
                  f"{a['effective_bits_achieved']} -- the model's floor. Embeddings, "
                  f"lm_head and routers are never quantized and set it.")
PY

echo
echo "  per-step status: $STATUS"
echo "  logs:            $LOGS/"
