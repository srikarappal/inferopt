#!/usr/bin/env bash
#
# The whole thing on ONE H100: lossless search, then the lossy ladder.
#
#   ./run_h100_ladder.sh                                  # Qwen3-14B, math_500
#   ./run_h100_ladder.sh Qwen/Qwen3-14B math_500 500 3
#   GPU=3 ./run_h100_ladder.sh                            # pin a different card
#
# Four stages, each skipped if its output already exists, so an interrupted run
# resumes by re-running this script:
#
#   0  preflight     what this GPU is, and what it can and cannot produce
#   1  lossless      run.py optimize --lossless-only, operating point SWEPT
#   2  artifacts     build the quantized checkpoints this GPU can use
#   3  lossy ladder  eval_ladder.sh over everything that got built
#
# WHY BOTH 1 AND 3, when they overlap. They are different instruments and the
# rows do not mix. The traversal sweeps concurrency and scores each config at
# its own PEAK; the ladder pins one operating point so variants are compared
# like for like. A ladder row measured above the hardware's capacity reports the
# concurrency cliff rather than the configuration -- which is exactly what
# happened on GB10, where L=30 was pinned for a model that sustained 8.
#
# H100 IS NOT GB10, and two things follow.
#
#   gpu_memory_utilization  0.90 here, 0.75 there. On a unified-memory part the
#                           fraction is of SYSTEM memory that the CPU also
#                           competes for; 0.90 of 122GB left ~1.6GB of headroom
#                           and ran into the OOM killer. On a dedicated 80GB
#                           H100 that same 0.75 strands ~20GB. This is derived
#                           from the fingerprint by evaluator.hardware_defaults,
#                           NOT hardcoded -- preflight prints what it resolved so
#                           a wrong value is visible before hours are spent.
#
#   nvfp4                   needs FP4 tensor cores (sm100+). H100 is sm90, so
#                           check_support.py excludes it and eval_ladder skips
#                           the row. w4a16 and autoquant are ATTEMPTED rather
#                           than assumed: their weights carry NVFP4 block scales
#                           but activations stay 16-bit, and vLLM's loader
#                           declares min capability 75 for both, so they may
#                           dequantize fine on sm90. If they do not, the launch
#                           fails fast and the ladder records it and moves on.

set -uo pipefail
cd "$(dirname "$0")"

MODEL=${1:-Qwen/Qwen3-14B}
BENCH=${2:-math_500}
N=${3:-500}
REPEATS=${4:-3}
GPU=${GPU:-0}
PY=${PYTHON:-python}
TRACE=data/trace_shared.jsonl
SAFE=${MODEL//\//__}

export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONDONTWRITEBYTECODE=1

LOSSLESS_DIR=runs/h100-lossless
LADDER_DIR=runs/ladder-$BENCH
mkdir -p logs

note() { printf '\n\033[1m=== %s ===\033[0m\n' "$*"; }
have() { [ -e "$1" ]; }

# --------------------------------------------------------------------------
note "stage 0  preflight"
# --------------------------------------------------------------------------
$PY - "$MODEL" "$TRACE" <<'PY'
import sys, torch
model, trace = sys.argv[1], sys.argv[2]
n = torch.cuda.device_count()
cap = torch.cuda.get_device_capability(0)
p = torch.cuda.get_device_properties(0)
print(f"  visible GPU   {p.name}  sm{cap[0]}{cap[1]}  {p.total_memory/1e9:.0f} GB")
print(f"  CUDA_VISIBLE_DEVICES pins {n} device(s) -- the ladders are single-GPU")
if n != 1:
    print(f"  WARNING: {n} devices visible. vLLM may claim memory on all of them.")
    print(f"           Re-run with GPU=<n> to pin exactly one.")
try:
    from request import InferOptRequest, build_fingerprint
    from evaluator import hardware_defaults
    fp, _ = build_fingerprint(InferOptRequest(model=model, trace=trace))
    d = hardware_defaults(fp)
    print(f"\n  model         {fp.model.id}  {fp.model.n_params_b:.1f}B "
          f"{fp.model.attention_type}  {fp.model.weight_gb:.1f} GB weights")
    print(f"  unified mem   {fp.hw.unified_memory}   -> gpu_memory_utilization "
          f"{d['gpu_memory_utilization']}")
    if fp.hw.unified_memory:
        print(f"  NOTE: this box reports UNIFIED memory. If it is really an H100 the")
        print(f"        fingerprint is wrong and 0.75 will strand ~20 GB.")
    for k, v in d.items():
        if k != "gpu_memory_utilization":
            print(f"  also derived  {k} = {v}")
    print(f"\n  decode roofline  {fp.model.active_weight_gb:.1f} GB / "
          f"{fp.hw.memory_bandwidth_gb_s:.0f} GB/s = "
          f"{fp.model.active_weight_gb/fp.hw.memory_bandwidth_gb_s*1000:.0f} ms "
          f"per token, before any overhead")
except Exception as e:
    print(f"  fingerprint failed: {type(e).__name__}: {e}")
PY

SKIP=$($PY check_support.py 2>/dev/null)
if [ -n "$SKIP" ]; then
    echo
    echo "  cannot run here: $SKIP  (needs FP4 tensor cores, sm100+)"
    echo "  everything else is attempted; a launch that fails is recorded, not fatal"
fi

if ! have "$TRACE"; then
    echo
    echo "  MISSING $TRACE -- the workload trace defines the whole experiment."
    echo "  Copy it from the machine that built it so the numbers stay comparable:"
    echo "      rsync <host>:<repo>/$TRACE data/"
    echo "  Rebuilding it here samples ShareGPT differently and the runs will not"
    echo "  be comparable to any previous result."
    exit 1
fi
have data/${BENCH}.jsonl || { echo "  MISSING data/${BENCH}.jsonl -- run ./setup.sh"; exit 1; }

# --------------------------------------------------------------------------
note "stage 1  lossless search  (swept operating point)"
# --------------------------------------------------------------------------
if have "$LOSSLESS_DIR/result.json"; then
    echo "  already done -> $LOSSLESS_DIR/result.json"
else
    echo "  ~2-4h, ~12 launches. Progress is journaled to trials.jsonl as it goes,"
    echo "  so an interrupted run is not lost."
    $PY run.py optimize \
        --model "$MODEL" --trace "$TRACE" \
        --ttft-p99 500 --itl-p99 250 \
        --lossless-only \
        --run-dir "$LOSSLESS_DIR" 2>&1 | tee "logs/h100-lossless.log"
fi

# --------------------------------------------------------------------------
note "stage 2  build the artifacts this GPU can use"
# --------------------------------------------------------------------------
# fp8 is absent on purpose: vLLM quantizes bf16 weights during model load, so it
# is a launch flag with no checkpoint. It is covered by configs/fp8.json in
# stage 3.
for KIND in w4a16 autoquant@6.0 autoquant@5.0 nvfp4; do
    TAG=${KIND/@/_}
    DIR="artifacts/${SAFE}--${TAG}"
    if echo " $SKIP " | grep -q " ${KIND} "; then
        echo "  skip     $KIND -- unsupported on this GPU"; continue
    fi
    # hf_quant_config.json is written last, so its presence means the export
    # finished. A bare directory is a conversion that died mid-write.
    if have "$DIR/hf_quant_config.json"; then
        echo "  cached   $KIND"; continue
    fi
    echo "  building $KIND  (hours; a failure here is recorded and the rest continue)"
    rm -rf "$DIR"
    if $PY quantize.py --model "$MODEL" --trace "$TRACE" --produce "$KIND" \
            2>&1 | tee "logs/h100-produce-$TAG.log" | tail -3; then
        echo "  built    $KIND"
    else
        echo "  FAILED   $KIND -- see logs/h100-produce-$TAG.log"
    fi
done

# --------------------------------------------------------------------------
note "stage 3  lossy ladder  (pinned operating point)"
# --------------------------------------------------------------------------
# eval_ladder discovers variants from artifacts/ rather than a hardcoded list,
# so whatever stage 2 managed to build is what gets measured.
PYTHON=$PY ./eval_ladder.sh "$BENCH" "$N" "$REPEATS" 2>&1 | tee "logs/h100-ladder.log"

# --------------------------------------------------------------------------
note "results"
# --------------------------------------------------------------------------
have "$LOSSLESS_DIR/result.json" && {
    echo "  lossless search:"
    $PY - "$LOSSLESS_DIR" <<'PY'
import json, sys
r = json.load(open(f"{sys.argv[1]}/result.json"))
b = r.get("baseline") or {}
kept = [t for t in r["trials"] if t.get("kept")]
best = max(r["trials"], key=lambda t: t.get("goodput") or 0)
print(f"    seed      {b.get('goodput', 0):7.1f} tok/s  L={b.get('concurrency')}")
for t in kept:
    print(f"    KEEP      {t['goodput']:7.1f} tok/s  L={t.get('concurrency')}  {t['node_id']}")
print(f"    best      {best['goodput']:7.1f} tok/s  {best['node_id']}"
      + (f"   {best['goodput']/b['goodput']:.2f}x the seed" if b.get("goodput") else ""))
PY
}
echo
echo "  lossy ladder:"
$PY summarize.py "$LADDER_DIR" 2>/dev/null | head -14

cat <<EOF

  full outputs
    $LOSSLESS_DIR/result.json     swept search, frontier, capacity curves
    $LADDER_DIR/                  one directory per variant, generations included
    logs/                         stdout of every stage

  The two are NOT comparable row to row: stage 1 scores each config at its own
  peak concurrency, stage 3 pins one for all of them. Compare within a stage.
EOF
