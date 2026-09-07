#!/usr/bin/env bash
#
# Re-measure EVERYTHING against the corrected load driver.
#
#   ./rerun_all.sh                       # 1.7B, then 14B, then 30B-MoE
#   MODELS="1.7b" ./rerun_all.sh         # one family
#   RUN_TAG=h100 ./rerun_all.sh          # H100, into runs_h100/
#
# WHY. Every number on record went through a driver with two defects: all
# closed-loop workers started in the same instant (62a2955) and every request
# asked for int(mean_output_tokens) regardless of what its trace row said
# (5b72afb). At a FIXED config, goodput moved 1.15x launch to launch -- wider
# than the 1.11x margin the pb-over-seqDAG result was claimed on. The configs
# those runs discovered are still good candidates; their measurements are void.
# This re-measures, it does not re-discover.
#
# It also captures what the old runs did not: requests.jsonl.gz, the
# per-request record. Without it the SLO is frozen at whatever was guessed
# before the run -- 500ms/250ms here -- because goodput counts requests that
# individually met the bound and no p99 can be re-thresholded after the fact.
# With it, `python -m inferopt.slo_explore` re-derives goodput, attainment,
# replicas and cost at any TTFT/ITL without touching the GPU. That is the whole
# reason this script must not start on a stale checkout.
#
# Expect LOWER goodput and WORSE attainment than the old tables: real output
# lengths span 37..1464 tokens where the old driver issued a flat 259, so the
# tail is being asked to serve requests that previously did not exist.
#
# COST on GB10, from the recorded times of the runs being replaced:
#     1.7B    ~6.5h     baseline, aiconfig, seqDAG, yolo, PB
#     14B    ~21h       baseline, seqDAG, yolo, PB, lossy walk
#     MoE    ~14h       stock, lossless walk, lossy walk, PB
#                ------
#              ~41h     H100 is roughly half
#
# Idempotent per step: a killed run resumes by re-invoking this script.
# Everything lands in rerun-* directories; the old ones are the before side of
# the comparison and are never touched.

set -uo pipefail
cd "$(dirname "$0")"

# FIND AN INTERPRETER THAT ACTUALLY HAS vLLM, rather than hardcoding one.
# A baked-in conda path works on exactly the host it was written on: this
# script died on line 75 with "No such file or directory" for an interpreter
# that exists on the GB10 box and not on the H100. Running a 41-hour queue
# under the wrong python is worse than failing, so the choice is made once,
# reported, and checked.
pick_python() {
    local c
    # An EXPLICIT PYTHON is honoured or refused, never quietly replaced --
    # substituting a different interpreter for the one you named is how a run
    # ends up measuring a vLLM you did not choose.
    if [ -n "${PYTHON:-}" ]; then
        if command -v "$PYTHON" >/dev/null 2>&1 && "$PYTHON" -c "import vllm" >/dev/null 2>&1; then
            command -v "$PYTHON"; return 0
        fi
        echo "PYTHON=$PYTHON is not usable (missing, or no vllm in it)" >&2
        return 1
    fi
    for c in python python3 /home/srikar/miniconda3/envs/verticalinference/bin/python; do
        command -v "$c" >/dev/null 2>&1 || continue
        if "$c" -c "import vllm" >/dev/null 2>&1; then
            command -v "$c"
            return 0
        fi
    done
    return 1
}
PY=$(pick_python) || {
    cat >&2 <<'MSG'
No interpreter with vLLM found. Tried $PYTHON, python, python3, and this
project's usual conda env.

  PYTHON=/path/to/env/bin/python ./rerun_all.sh

Check the one you mean with:  <python> -c 'import vllm; print(vllm.__version__)'
Running the queue under an interpreter without vLLM fails 20 launches in.
MSG
    exit 1
}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
TRACE=data/trace_shared.jsonl
# ${MODELS-...}, not ${MODELS:-...}: an EMPTY MODELS means "no families, just
# run the preflight checks", which is how you verify a host before committing
# 41 hours to it. With :- an empty value silently became the full queue.
MODELS=${MODELS-"1.7b 14b moe"}
TAG=${RUN_TAG:-}
OUT=runs; [ -n "$TAG" ] && OUT="runs_${TAG}"
mkdir -p "$OUT"

say() { printf '\n[%s] === %s ===\n' "$(date '+%F %T')" "$*"; }

# Never start on top of a job still holding the GPU. The pattern deliberately
# cannot match this script's own command line -- that self-match is what
# deadlocked the overnight queue for 55 minutes.
wait_for_gpu() {
    local n=0
    while pgrep -f "inferopt\.pb_screen|inferopt\.run optimize|inferopt\.yolo_run|inferopt\.eval_repro|stage3_turn[0-9]" >/dev/null 2>&1; do
        [ $((n % 15)) -eq 0 ] && say "GPU busy, waiting (${n} min)"
        sleep 60; n=$((n+1))
        [ $n -gt 1440 ] && { say "still busy after 24h, giving up"; return 1; }
    done
    say "GPU is free"
}

# A vLLM server outliving the queue that started it is not a nuisance, it is a
# silent 41-hour failure. _serve launches with start_new_session=True and kills
# its process group in a `finally`; SIGKILL the queue, or kill it while a
# launch is in flight, and that finally never runs. The orphan keeps ~94 GB and
# every later launch dies with "Engine core initialization failed", which reads
# as a config problem and is not. Observed exactly that, twice.
#
# Only servers running from THIS interpreter's environment prefix, and only
# those already reparented to init. dextract's OCR server runs from a different
# conda env and is production traffic -- it must never match.
reap_orphans() {
    local prefix pid ppid n=0
    prefix=$(dirname "$(dirname "$PY")")          # .../envs/<env>
    for pid in $(pgrep -f "vllm serve" 2>/dev/null); do
        case "$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null)" in
            *"$prefix"*) ;;                        # ours
            *) continue ;;                         # someone else's -- leave it
        esac
        ppid=$(awk '{print $4}' /proc/$pid/stat 2>/dev/null)
        [ "$ppid" = "1" ] || continue              # still parented: in use
        say "reaping orphaned vLLM pid $pid (holding GPU memory from a killed run)"
        kill -INT "-$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')" 2>/dev/null
        n=$((n+1))
    done
    [ "$n" -gt 0 ] && sleep 15
    return 0
}
trap 'echo; say "interrupted -- reaping before exit"; reap_orphans; exit 130' INT TERM

step() {
    local name=$1 marker=$2; shift 2
    if [ -e "$marker" ]; then say "$name: already done, skipping"; return 0; fi
    reap_orphans                       # before, not after: a leftover from the
                                       # previous step is what breaks this one
    say "$name: starting"
    if "$@"; then say "$name: OK"; else say "$name: FAILED (continuing)"; fi
    wait_for_gpu
}

say "interpreter  $PY  (vLLM $("$PY" -c 'import vllm;print(vllm.__version__)' 2>/dev/null))"
$PY - <<'CHECK' || exit 1
import inspect
import inferopt.evaluator as e
assert "stagger_s" in inspect.signature(e._closed_loop).parameters, \
    "STALE CHECKOUT: no worker stagger. git pull before spending 41 GPU-hours."
assert hasattr(e.VllmEvaluator, "replay_lengths"), \
    "STALE CHECKOUT: no per-request replay lengths. git pull."
assert hasattr(e.VllmEvaluator, "dump_requests"), \
    "STALE CHECKOUT: no per-request capture -- the runs would finish with the SLO " \
    "frozen at 500/250 and need doing a third time. git pull."
import inferopt.slo_explore  # noqa: F401
print("  driver fixes and per-request capture present")
CHECK
wait_for_gpu

# --------------------------------------------------------------------------
run_family() {
    local fam=$1 model=$2 nbench=$3 conc=$4
    local P="${OUT}/rerun-${fam}"
    say "===== ${fam}: ${model} ====="

    step "${fam} stock baseline" "${P}-baseline/eval.json" \
        $PY -m inferopt.eval_repro --model "$model" --benchmark math_500 \
            --n "$nbench" --repeats 3 --trace $TRACE --serving-concurrency "$conc" \
            --run-dir "${P}-baseline"

    # --skip-predict so seqDAG starts from the SAME seed as yolo and pb.
    # Without it the walk gets aiconfigurator's max_num_seqs=512 and the three
    # methods are not comparable -- that asymmetry was invisible to provenance.
    step "${fam} seqDAG" "${P}-seqdag/result.json" \
        $PY -m inferopt.run optimize --model "$model" --trace $TRACE \
            --ttft-p99 500 --itl-p99 250 --qps 16 --lossless-only \
            --skip-predict --quality-every-node --run-dir "${P}-seqdag"

    step "${fam} yolo" "${P}-yolo/result.json" \
        $PY -m inferopt.yolo_run --model "$model" --trace $TRACE \
            --ttft-p99 500 --itl-p99 250 --qps 16 --run-dir "${P}-yolo"

    # The long pole: 20 launches. 3 of 20 failed on the 1.7B H100 last time on
    # illegal pinning; legality.py fixes that, so expect 20 for 20.
    step "${fam} PB screen" "${P}-pb/result.json" \
        $PY -m inferopt.pb_screen --model "$model" --trace $TRACE \
            --ttft-p99 500 --itl-p99 250 --qps 16 --survivors 3 \
            --run-dir "${P}-pb"

    say "${fam} comparison"
    local ARGS=""
    [ -f "${P}-baseline/eval.json" ] && ARGS="--baseline ${P}-baseline"
    $PY -m inferopt.compare "${P}-seqdag" "${P}-yolo" "${P}-pb" $ARGS \
        --plot "${P}-frontier.png" 2>&1 | tee "${P}-comparison.txt"
}

# --------------------------------------------------------------------------
for fam in $MODELS; do
case "$fam" in

1.7b)
    run_family 1.7b Qwen/Qwen3-1.7B 500 30
    # aiconfigurator as its own BASELINE row: what the predictor gives a user
    # today, on a b200_sxm proxy because it has no GB10. Not anyone's head start.
    cat > /tmp/aic-1.7b-rerun.json <<'JSON'
{"gpu_memory_utilization": 0.75, "max_num_seqs": 512, "max_model_len": 7168,
 "enable_prefix_caching": false, "enable_chunked_prefill": false,
 "enforce_eager": true}
JSON
    step "1.7b aiconfigurator baseline" "${OUT}/rerun-1.7b-aiconfig/eval.json" \
        $PY -m inferopt.eval_repro --model Qwen/Qwen3-1.7B --benchmark math_500 \
            --n 500 --repeats 3 --trace $TRACE --serving-concurrency 30 \
            --config /tmp/aic-1.7b-rerun.json --run-dir "${OUT}/rerun-1.7b-aiconfig"
    ;;

14b)
    run_family 14b Qwen/Qwen3-14B 500 30
    # The 14B lossy walk builds autoquant@5.0 for this model -- hours of GPU
    # before a single measurement. Last, because it is the longest thing here.
    step "14b lossy walk" "${OUT}/rerun-14b-lossy/result.json" \
        $PY -m inferopt.run optimize --model Qwen/Qwen3-14B --trace $TRACE \
            --ttft-p99 500 --itl-p99 250 --qps 16 --allow-loss 0.1 \
            --skip-predict --run-dir "${OUT}/rerun-14b-lossy"
    ;;

moe)
    # Smaller bench and lower pinned concurrency: 30B-A3B at 100 prompts and
    # L=8 is what the original stock row used, and changing it would break the
    # only comparison this arm has.
    step "moe stock baseline" "${OUT}/rerun-moe-baseline/eval.json" \
        $PY -m inferopt.eval_repro --model Qwen/Qwen3-30B-A3B \
            --config configs/stock.json --benchmark math_500 --n 100 --repeats 3 \
            --trace $TRACE --serving-concurrency 8 \
            --run-dir "${OUT}/rerun-moe-baseline"

    step "moe lossless walk" "${OUT}/rerun-moe-lossless/result.json" \
        $PY -m inferopt.run optimize --model Qwen/Qwen3-30B-A3B --trace $TRACE \
            --ttft-p99 500 --itl-p99 250 --qps 16 --lossless-only \
            --skip-predict --run-dir "${OUT}/rerun-moe-lossless"

    step "moe lossy walk" "${OUT}/rerun-moe-lossy/result.json" \
        $PY -m inferopt.run optimize --model Qwen/Qwen3-30B-A3B --trace $TRACE \
            --ttft-p99 500 --itl-p99 250 --qps 16 --allow-loss 0.1 \
            --skip-predict --run-dir "${OUT}/rerun-moe-lossy"

    step "moe PB screen" "${OUT}/rerun-moe-pb/result.json" \
        $PY -m inferopt.pb_screen --model Qwen/Qwen3-30B-A3B --trace $TRACE \
            --ttft-p99 500 --itl-p99 250 --qps 16 --survivors 3 \
            --run-dir "${OUT}/rerun-moe-pb"
    ;;

*) say "unknown family '$fam' -- expected one of: 1.7b 14b moe" ;;
esac
done

reap_orphans
say "all families done"
printf '\n  per-request capture written to:\n'
for d in "${OUT}"/rerun-*/requests.jsonl.gz; do
    [ -e "$d" ] && printf '    %-46s %s\n' "$d" "$(du -h "$d" | cut -f1)"
done
cat <<'MSG'

  The SLO is no longer frozen. To re-ask any of these runs at a different
  promise, with no GPU:

    python -m inferopt.slo_explore runs/rerun-1.7b-pb --ttft 300 --itl 200 \
        --demand-qps 16 --gpu-hourly-usd 3.0

  or dump the whole grid a slider indexes into:

    python -m inferopt.slo_explore runs/rerun-1.7b-pb --sweep \
        --demand-qps 16 --gpu-hourly-usd 3.0 --out runs/rerun-1.7b-pb-grid.json
MSG
