#!/usr/bin/env bash
#
# The overnight queue. Runs unattended; nothing here needs a live session.
#
#   nohup ./overnight.sh > runs/overnight.log 2>&1 &
#
# SERIALISED ON PURPOSE. One GPU job at a time, in the order asked for. Two vLLM
# servers on one GPU do not fail cleanly -- they fight for memory and produce
# measurements that look real and are not.
#
# Each step is skipped if its output already exists, so this is safe to re-run
# after a kill, and a step that FAILS does not stop the queue: losing the 14B
# work because a 1.7B rerun died would be the worst outcome of a night's compute.
#
# Order is the one requested:
#   1. wait for the pb_screen run already in flight
#   2. seqDAG re-run with --skip-predict, so all three methods share a seed
#   3. the aiconfigurator config, measured as its own baseline row
#   4. the 14B: baseline, then all three methods, then the lossy walk

set -uo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-/home/srikar/miniconda3/envs/verticalinference/bin/python}
export CUDA_VISIBLE_DEVICES=0
TRACE=data/trace_shared.jsonl

say() { printf '\n[%s] === %s ===\n' "$(date '+%F %T')" "$*"; }

# Wait for any inferopt GPU job to exit. Polls rather than waits on a PID,
# because the job was started from another shell and is not our child.
wait_for_gpu() {
    local n=0
    while pgrep -f "pb_screen.py|run.py optimize|yolo_run.py|eval_repro.py" >/dev/null 2>&1; do
        [ $((n % 30)) -eq 0 ] && say "waiting for the in-flight job ($((n)) min so far)"
        sleep 60; n=$((n+1))
        [ $n -gt 480 ] && { say "STILL running after 8h; giving up on the wait"; return 1; }
    done
    say "GPU is free"
    return 0
}

# One step. Skipped when its marker file exists; a failure is logged, not fatal.
run_step() {
    local name=$1 marker=$2; shift 2
    if [ -e "$marker" ]; then say "$name: already done ($marker), skipping"; return 0; fi
    say "$name: starting"
    if "$@"; then say "$name: OK"; else say "$name: FAILED (continuing)"; fi
    wait_for_gpu
}

say "overnight queue starting"
wait_for_gpu

# ---- 1.7B: the same-seed re-run, so the three methods are comparable at all
run_step "1.7B seqDAG (--skip-predict, same seed as yolo/pb)" \
    runs/1.7b-seqdag-2/result.json \
    $PY run.py optimize --model Qwen/Qwen3-1.7B --trace $TRACE \
        --ttft-p99 500 --itl-p99 250 --qps 16 --lossless-only \
        --skip-predict --quality-every-node --run-dir runs/1.7b-seqdag-2

# ---- 1.7B: aiconfigurator as a BASELINE, not as anyone's head start.
# Exactly what stage 1.2 predicted for this model, one knob off the conservative
# seed: max_num_seqs 512 rather than 256. Predicted on a b200_sxm PROXY, since
# aiconfigurator has no GB10 -- so this is what a user would get today, not a
# recommendation tuned for this box.
cat > /tmp/aic-1.7b.json <<'JSON'
{"gpu_memory_utilization": 0.75, "max_num_seqs": 512, "max_model_len": 7168,
 "enable_prefix_caching": false, "enable_chunked_prefill": false,
 "enforce_eager": true}
JSON
run_step "1.7B aiconfigurator baseline" runs/1.7b-aiconfig/eval.json \
    $PY eval_repro.py --model Qwen/Qwen3-1.7B --benchmark math_500 \
        --n 500 --repeats 3 --trace $TRACE --serving-concurrency 30 \
        --config /tmp/aic-1.7b.json --run-dir runs/1.7b-aiconfig

say "1.7B comparison (with the same-seed walk)"
$PY compare.py runs/1.7b-seqdag-2 runs/1.7b-yolo runs/1.7b-pb \
    --baseline runs/1.7b-baseline --plot runs/1.7b-frontier.png \
    2>&1 | tee runs/1.7b-comparison.txt

# ---- 14B: stock baseline first, so the methods have an anchor
run_step "14B stock baseline" runs/14b-baseline/eval.json \
    $PY eval_repro.py --model Qwen/Qwen3-14B --benchmark math_500 \
        --n 500 --repeats 3 --trace $TRACE --serving-concurrency 30 \
        --run-dir runs/14b-baseline

# ---- 14B: all three methods. run_opt_ladder is resumable per method and now
# passes --skip-predict, so the three share a seed.
say "14B: three methods via run_opt_ladder.sh"
MODEL=Qwen/Qwen3-14B TAG=14b BASE=runs/14b-baseline PYTHON=$PY ./run_opt_ladder.sh
wait_for_gpu

# ---- 14B lossy walk. Never run on this model: every 14B traversal to date used
# --lossless-only, so its lossy numbers come from the ladder at a pinned L and
# are not comparable with any walk. This is last because it is the longest --
# autoquant@5.0 is not built for the 14B and will be produced here.
run_step "14B lossy walk (builds autoquant@5.0)" runs/14b-lossy-1/result.json \
    $PY run.py optimize --model Qwen/Qwen3-14B --trace $TRACE \
        --ttft-p99 500 --itl-p99 250 --qps 16 --allow-loss 0.1 \
        --skip-predict --run-dir runs/14b-lossy-1

say "queue finished"
for d in runs/1.7b-seqdag-2 runs/1.7b-aiconfig runs/14b-baseline \
         runs/14b-seqdag runs/14b-yolo runs/14b-pb runs/14b-lossy-1; do
    printf '  %-24s %s\n' "$(basename $d)" \
        "$([ -e "$d/result.json" ] || [ -e "$d/eval.json" ] && echo DONE || echo 'not reached')"
done
