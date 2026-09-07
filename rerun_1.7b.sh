#!/usr/bin/env bash
#
# Re-measure all five 1.7B arms against the CORRECTED load driver.
#
#   ./rerun_1.7b.sh                      # GB10, runs/rerun-1.7b-*
#   RUN_TAG=h100 ./rerun_1.7b.sh         # H100, runs_h100/rerun-1.7b-*
#
# WHY. Every number in runs/1.7b-* was measured through a driver with two
# defects: all closed-loop workers started in the same instant (the convoy,
# 62a2955) and every request asked for int(mean_output_tokens)=259 regardless
# of what its trace row said (5b72afb). Goodput at a FIXED config moved 1.15x
# launch to launch, which is wider than the 1.11x margin the pb-over-seqDAG
# result was claimed on. The configs those runs discovered are still good
# candidates; their measurements are not. This re-measures, it does not
# re-discover.
#
# Expect LOWER numbers and WORSE attainment than the old table. The workload is
# now genuinely harder: real lengths span 37..1464 tokens where the old driver
# issued a flat 259, so requests the tail was never asked to serve now exist.
#
# New directories throughout -- the old ones are the before side of the
# comparison and must not be overwritten.
#
# Idempotent: each step is skipped if its output already exists, so a killed
# run resumes by re-invoking this script.

set -uo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-/home/srikar/miniconda3/envs/verticalinference/bin/python}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
TRACE=data/trace_shared.jsonl
TAG=${RUN_TAG:-}
OUT=runs; [ -n "$TAG" ] && OUT="runs_${TAG}"
P=${OUT}/rerun-1.7b
mkdir -p "$OUT"

say() { printf '\n[%s] === %s ===\n' "$(date '+%F %T')" "$*"; }

# Do not start on top of a job that is still holding the GPU -- including the
# stage-3 turns. NOTE the pattern below must not match this script's own
# command line, which is how a previous version deadlocked its own queue.
wait_for_gpu() {
    local n=0
    while pgrep -f "inferopt\.pb_screen|inferopt\.run optimize|inferopt\.yolo_run|inferopt\.eval_repro|stage3_turn[0-9]" >/dev/null 2>&1; do
        [ $((n % 15)) -eq 0 ] && say "GPU busy, waiting (${n} min)"
        sleep 60; n=$((n+1))
        [ $n -gt 720 ] && { say "still busy after 12h, giving up"; return 1; }
    done
    say "GPU is free"
}

step() {
    local name=$1 marker=$2; shift 2
    if [ -e "$marker" ]; then say "$name: already done ($marker), skipping"; return 0; fi
    say "$name: starting"
    if "$@"; then say "$name: OK"; else say "$name: FAILED (continuing)"; fi
    wait_for_gpu
}

say "1.7B re-measurement starting -> ${P}-*"
$PY -c "import inferopt.evaluator as e, inspect
assert 'stagger_s' in inspect.signature(e._closed_loop).parameters, 'STALE: no stagger -- git pull'
assert hasattr(e.VllmEvaluator, 'replay_lengths'), 'STALE: no replay_lengths -- git pull'
print('  driver fixes present')" || exit 1
wait_for_gpu

# 1. stock -- pinned concurrency, the anchor everything else is read against
step "stock baseline" "${P}-baseline/eval.json" \
    $PY -m inferopt.eval_repro --model Qwen/Qwen3-1.7B --benchmark math_500 \
        --n 500 --repeats 3 --trace $TRACE --serving-concurrency 30 \
        --run-dir "${P}-baseline"

# 2. aiconfigurator -- what a user gets from the predictor today, on a b200_sxm
#    proxy because it has no GB10. A baseline, not anyone's head start.
cat > /tmp/aic-1.7b-rerun.json <<'JSON'
{"gpu_memory_utilization": 0.75, "max_num_seqs": 512, "max_model_len": 7168,
 "enable_prefix_caching": false, "enable_chunked_prefill": false,
 "enforce_eager": true}
JSON
step "aiconfigurator baseline" "${P}-aiconfig/eval.json" \
    $PY -m inferopt.eval_repro --model Qwen/Qwen3-1.7B --benchmark math_500 \
        --n 500 --repeats 3 --trace $TRACE --serving-concurrency 30 \
        --config /tmp/aic-1.7b-rerun.json --run-dir "${P}-aiconfig"

# 3. sequential DAG. --skip-predict so it starts from the same seed as yolo and
#    pb; without it the walk gets aiconfigurator's 512 and the three methods are
#    not comparable at all.
step "seqDAG" "${P}-seqdag/result.json" \
    $PY -m inferopt.run optimize --model Qwen/Qwen3-1.7B --trace $TRACE \
        --ttft-p99 500 --itl-p99 250 --qps 16 --lossless-only \
        --skip-predict --quality-every-node --run-dir "${P}-seqdag"

# 4. yolo
step "yolo" "${P}-yolo/result.json" \
    $PY -m inferopt.yolo_run --model Qwen/Qwen3-1.7B --trace $TRACE \
        --ttft-p99 500 --itl-p99 250 --qps 16 --run-dir "${P}-yolo"

# 5. Plackett-Burman. 20 launches, the long pole -- ~3.5h on GB10, ~1.75h on
#    H100. 3 of 20 failed here last time on illegal pinning; legality.py fixes
#    that, so expect 20 for 20.
step "PB screen" "${P}-pb/result.json" \
    $PY -m inferopt.pb_screen --model Qwen/Qwen3-1.7B --trace $TRACE \
        --ttft-p99 500 --itl-p99 250 --qps 16 --survivors 3 \
        --run-dir "${P}-pb"

say "comparison"
$PY -m inferopt.compare "${P}-seqdag" "${P}-yolo" "${P}-pb" \
    --baseline "${P}-baseline" --plot "${P}-frontier.png" \
    2>&1 | tee "${P}-comparison.txt"

say "done -- old table runs/1.7b-comparison.txt, new one ${P}-comparison.txt"
