"""Stage 3, turn 1: is the operating point bound by prefill head of line blocking?

OBSERVE. The incumbent is max_model_len_rightsize at 1968.8 tok/s, L=256, with
enforce_eager true and chunked prefill off. At that peak kv_cache_util is 0.184
and preemptions is 0, so 82% of the KV pool is idle and nothing is evicted. The
curve is a cliff: L=256 gives ttft_p99 500ms at 0.99 attainment, L=512 gives
ttft_p99 36832ms at 0.00 attainment, while throughput only falls 2016.9 to
1830.2. The server keeps producing tokens at L=512. What collapses is TTFT.

HYPOTHESISE. That collapse is prefill head of line blocking. With chunked
prefill off, a step that admits a long prompt blocks every decode behind it, and
this trace has p99 input 2660 and p999 input 4380 against a mean of 620, so a
few long prompts dominate the queue.

THE LEVER, SECOND ATTEMPT. The first version used
long_prefill_token_threshold with max_num_partial_prefills and
max_long_partial_prefills. All three parse, so installed_flags() accepted them,
and then vLLM 0.26's V1 engine refused at startup with "Concurrent Partial
Prefill is not supported". A flag existing in the parser says nothing about the
engine implementing it, which is why every config here now gets a preflight
launch before it is allowed to consume a measurement slot.

What V1 does implement is chunked prefill plus a token budget per step.
max_num_batched_tokens caps how many tokens a step may contain, so a small
budget with chunking on bounds the prefill any single step can be dominated by.
That is the same head of line lever by a different route.

WHY THE DAG COULD NOT FIND THIS. It tests enable_chunked_prefill as a lone
boolean and max_num_batched_tokens as its own node, reverting each against the
incumbent, and never revisits a reverted node. A budget that only helps once
chunking is on is unreachable to a walk that measured them separately.

PREDICT. If right, a bounded step budget keeps TTFT finite at L=512 and moves
the cliff outward, so peak goodput clears 1.065x the re-measured control, which
is the worst pass to pass spread observed on this host and model. If wrong, TTFT
stays unbounded at L=512 or throughput falls, because chunking costs a prefill
that could have run in one shot.
"""
import subprocess
import time

from inferopt.evaluator import LaunchError
from inferopt.methods import MethodRunner, setup

INCUMBENT = {
    "gpu_memory_utilization": 0.75,
    "max_num_seqs": 256,
    "max_model_len": 6144,
    "enable_prefix_caching": True,
    "enable_chunked_prefill": False,
    "enforce_eager": True,
}

# 6144 is the served context, so a 1024 budget forces any prompt above that to
# span several steps, and 2048 splits the trace at roughly its p99 input of 2660.
ARMS = [
    ("control", dict(INCUMBENT)),
    ("chunk_budget_1024", {**INCUMBENT, "enable_chunked_prefill": True,
                           "max_num_batched_tokens": 1024}),
    ("chunk_budget_2048", {**INCUMBENT, "enable_chunked_prefill": True,
                           "max_num_batched_tokens": 2048}),
]

BAND = 1.065
BASELINE = 1968.8


def wait_for_gpu(poll_s: int = 120) -> None:
    """Block until no other inferopt job holds the accelerator."""
    waited = 0
    while True:
        busy = subprocess.run(
            ["pgrep", "-f", r"inferopt\.pb_screen|inferopt\.run optimize"
                            r"|inferopt\.yolo_run|inferopt\.eval_repro"],
            capture_output=True).returncode == 0
        if not busy:
            return
        if waited % 1800 == 0:
            print(f"  waiting for the stage 2 queue ({waited // 60}m)", flush=True)
        time.sleep(poll_s)
        waited += poll_s


def preflight(runner, name: str, cfg: dict) -> bool:
    """Does this configuration start at all? Two minutes, not a measurement slot.

    Turn 1's first attempt spent three launches discovering that two of its arms
    could not boot, then crashed formatting the result, so the failure was not
    even reported. A config that cannot start is a fact about the engine and is
    worth finding before the sweep, not during it.
    """
    try:
        with runner.ev._serve(cfg, f"preflight-{name}"):
            print(f"    preflight {name}: starts", flush=True)
            return True
    except LaunchError as e:
        first = str(e).strip().splitlines()[0]
        print(f"    preflight {name}: WILL NOT START, {first}", flush=True)
        return False


wait_for_gpu()
fp, slo, ctx = setup("Qwen/Qwen3-1.7B", "data/trace_shared.jsonl", 500, 250, qps=16)
runner = MethodRunner("stage3-t1", fp, slo, "data/trace_shared.jsonl",
                      "runs/stage3-t1-prefill", benchmarks=[], quality_every=False)

print("  preflight", flush=True)
live = [(n, c) for n, c in ARMS if preflight(runner, n, c)]
dead = [n for n, _ in ARMS if n not in {x for x, _ in live}]
if dead:
    print(f"  dropped, will not start: {', '.join(dead)}", flush=True)
if not any(n == "control" for n, _ in live):
    raise SystemExit("  control will not start, nothing to compare against")

rows = []
for name, cfg in live:
    print(f"\n  === {name} ===", flush=True)
    rows.append((name, runner.measure(cfg, name)))

print("\n  RESULT")
print(f"    {'arm':18s} {'goodput':>9s} {'L':>5s} {'ttft_p99':>10s} {'slo':>7s} {'vs control':>11s}")
control = next(t for n, t in rows if n == "control")
for name, t in rows:
    d = t.diagnostics or {}
    conc = t.concurrency if t.concurrency is not None else "-"
    ttft = f"{t.ttft_p99_ms:.0f}ms" if t.ttft_p99_ms not in (None, float("inf")) else "-"
    att = d.get("slo_attainment")
    ratio = (t.goodput / control.goodput) if control.goodput else 0.0
    print(f"    {name:18s} {t.goodput or 0:9.1f} {str(conc):>5} {ttft:>10s} "
          f"{(f'{att:.1%}' if att is not None else '-'):>7s} {ratio:10.3f}x")

print("\n  the cliff, per arm:")
for name, t in rows:
    pts = "  ".join(f"L{c['concurrency']}:{c['goodput']:.0f}/{c['ttft_p99_ms']:.0f}ms"
                    for c in (t.curve or [])) or "no curve"
    print(f"    {name:18s} {pts}")

scored = [(n, t) for n, t in rows if t.goodput]
best_name, best = max(scored, key=lambda r: r[1].goodput)
ratio = best.goodput / control.goodput if control.goodput else 0.0
print(f"\n  VERDICT  best arm {best_name} at {best.goodput:.1f} tok/s")
print(f"    control re-measured {control.goodput:.1f} against a recorded {BASELINE}")
if best_name == "control":
    print("    REFUTED: no bounded step budget beat the incumbent.")
elif ratio > BAND:
    print(f"    SUPPORTED: {ratio:.3f}x clears the {BAND}x band.")
else:
    print(f"    INSIDE THE NOISE BAND: {ratio:.3f}x against {BAND}x. Not a result.")
runner.finish(chosen=None)
