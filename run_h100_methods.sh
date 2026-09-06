#!/usr/bin/env bash
#
# The three-method comparison on an H100 host, across all three models.
#
#   ./run_h100_methods.sh                 # all three
#   ./run_h100_methods.sh 1.7b            # just one
#
# H100 IS NOT GB10, and three defaults change with it:
#
#   gpu_memory_utilization  0.75 -> 0.90. On GB10 the fraction is of SYSTEM
#                           memory that the CPU, the benchmark client and the OS
#                           all share; 0.90 there left ~1.6GB and ran the box to
#                           the edge of the OOM killer. An H100's 80GB is the
#                           GPU's alone. hardware_defaults already keys this off
#                           unified_memory, so it needs no flag.
#   moe_backend             GB10 (sm121) has no FlashInfer kernels, which is why
#                           the MoE work there ran on triton and marlin. H100 is
#                           sm90 and has the full set; reconcile_moe_backend
#                           reads vLLM's own accepted list, so again no flag.
#   NVFP4                   needs Blackwell. On H100 those ladder rows are
#                           skipped automatically -- check_support.py says so
#                           before anything costs a model load.
#
# 80GB per card also changes what FITS: Qwen3-30B-A3B is 61.1GB resident, so it
# runs on one H100 but with little KV headroom. Pin ONE gpu; the search is
# single-GPU and vLLM will otherwise claim all eight.

set -uo pipefail
cd "$(dirname "$0")"
if ! "${PYTHON:-python}" -c "import inferopt" >/dev/null 2>&1; then
    export PYTHONPATH="$(cd "$(dirname "$0")" && pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
fi

PY=${PYTHON:-python}
export CUDA_VISIBLE_DEVICES=${GPU:-0}
TRACE=${TRACE:-data/trace_shared.jsonl}
QPS=${QPS:-16}
TTFT=${TTFT:-500}
ITL=${ITL:-250}

declare -A MODELS=(
  [1.7b]=Qwen/Qwen3-1.7B
  [14b]=Qwen/Qwen3-14B
  [30b]=Qwen/Qwen3-30B-A3B
)
WANT=${1:-"1.7b 14b 30b"}

echo
echo "  gpu       $CUDA_VISIBLE_DEVICES of $(nvidia-smi -L 2>/dev/null | wc -l) visible"
echo "  trace     $TRACE"
echo "  slo       ttft_p99 ${TTFT}ms  itl_p99 ${ITL}ms   qps $QPS"
$PY -m inferopt.check_support 2>/dev/null | sed 's/^/  unsupported: /' || true
echo

for tag in $WANT; do
    m=${MODELS[$tag]:-}
    [ -z "$m" ] && { echo "  unknown model tag: $tag"; continue; }

    echo "  ================================================================"
    echo "  == $tag  $m"
    echo "  ================================================================"

    # Stock, pinned -- the reference every method is read against. NOT ranked
    # beside them: a pinned point and a swept peak measure different things.
    if [ ! -f "runs/h100-$tag-baseline/eval.json" ]; then
        $PY -m inferopt.eval_repro --model "$m" --benchmark math_500 \
            --n 500 --repeats 3 --trace "$TRACE" --serving-concurrency 30 \
            --run-dir "runs/h100-$tag-baseline"
    fi

    # The three methods, from an IDENTICAL seed. --skip-predict is not optional
    # here: without it run.py seeds the walk from aiconfigurator while the other
    # two seed from seed_config(), which is a head start the provenance stamps
    # cannot show.
    MODEL="$m" TAG="h100-$tag" BASE="runs/h100-$tag-baseline" \
        PYTHON="$PY" TRACE="$TRACE" QPS="$QPS" TTFT="$TTFT" ITL="$ITL" \
        ./run_opt_ladder.sh

    # Lossy, seeded from the lossless answer rather than the raw seed. On the
    # 14B the conservative seed cannot meet the SLO at all -- goodput ~0, and
    # every downstream percentage would be a ratio against zero.
    if [ -f "runs/h100-$tag-seqdag/result.json" ]; then
        $PY -m inferopt.run optimize --model "$m" --trace "$TRACE" \
            --ttft-p99 "$TTFT" --itl-p99 "$ITL" --qps "$QPS" --allow-loss 0.1 \
            --seed-from-run "runs/h100-$tag-seqdag" \
            --run-dir "runs/h100-$tag-lossy"
    fi
done

echo
echo "  ================================================================"
for tag in $WANT; do
    D=""
    for s in seqdag yolo pb; do
        [ -f "runs/h100-$tag-$s/result.json" ] && D="$D runs/h100-$tag-$s"
    done
    [ -z "$D" ] && continue
    echo; echo "  == $tag"
    $PY -m inferopt.compare $D --baseline "runs/h100-$tag-baseline" \
        --plot "runs/h100-$tag-frontier.png" | tee "runs/h100-$tag-comparison.txt"
done
