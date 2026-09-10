"""Stage 3, turn 2: the cliff is a residency cap, not a prefill problem.

OBSERVE, from turn 1. Bounding the per step token budget did not move the cliff
at all: ttft_p99 at L=512 measured 39021ms with chunking off, 38508ms at a 1024
token budget and 38846ms at 2048. Peak goodput moved 1797.8 to 1832.8, which is
1.019x against a 1.065x band, so nothing there either. Prefill head of line
blocking is refuted as the mechanism.

What the same runs show instead: max_num_seqs is 256 and the cliff appears at
exactly L=512, twice that. kv_cache_util at the peak is 0.184 and preemptions is
0, so the server has 82% of its KV pool idle and is evicting nothing. Throughput
at L=512 is still ~1830 tok/s while goodput collapses to 285.

HYPOTHESISE. At L=512 only max_num_seqs=256 requests can be resident, so the
other 256 are not slow, they are not running at all. Their TTFT is the wait for
a slot to free. At ~1830 tok/s across 256 residents and 260 output tokens per
request, a resident finishes in roughly 36 seconds, which is the 39 second TTFT
being measured. The number is not a latency, it is a queue.

PREDICT. If right, raising max_num_seqs to 512 lets every request be resident at
L=512, TTFT there falls from 39 seconds to seconds, and the peak moves off L=256
to a higher goodput. KV has room: 512 residents at the observed 0.184 per 256
lands near 0.37, well inside the pool. If wrong, TTFT stays high, which would
mean the limit is compute per step rather than residency, or throughput falls
because a wider batch costs more per token than it gains in parallelism.

WHY THE DAG COULD NOT FIND THIS. It retunes max_num_seqs only AFTER KV
quantization and AFTER weight quantization, as a repair for memory those steps
freed. This walk is lossless, so neither node ran, and max_num_seqs stayed at
the seed's 256 having never been swept on its own merits.
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

ARMS = [
    ("control_256", dict(INCUMBENT)),
    ("seqs_512", {**INCUMBENT, "max_num_seqs": 512}),
    ("seqs_1024", {**INCUMBENT, "max_num_seqs": 1024}),
]

BAND = 1.065
# Sweep past the cliff. The default bracket stops at 512, which is exactly where
# the collapse is, so a config that fixes the collapse would have nowhere to show
# it. measure() extends on its own when the peak sits at the top, but starting
# the bracket here saves that round trip.
LEVELS = (16, 32, 64, 128, 256, 512, 1024)


def wait_for_gpu(poll_s: int = 120) -> None:
    waited = 0
    while True:
        busy = subprocess.run(
            ["pgrep", "-f", r"inferopt\.pb_screen|inferopt\.run optimize"
                            r"|inferopt\.yolo_run|inferopt\.eval_repro|stage3_t1_"],
            capture_output=True).returncode == 0
        if not busy:
            return
        if waited % 1800 == 0:
            print(f"  waiting for the gpu ({waited // 60}m)", flush=True)
        time.sleep(poll_s)
        waited += poll_s


def preflight(runner, name: str, cfg: dict) -> bool:
    """Does it start? A parser that accepts a flag says nothing about the engine."""
    try:
        with runner.ev._serve(cfg, f"preflight-{name}"):
            print(f"    preflight {name}: starts", flush=True)
            return True
    except LaunchError as e:
        print(f"    preflight {name}: WILL NOT START, "
              f"{str(e).strip().splitlines()[0]}", flush=True)
        return False


wait_for_gpu()
fp, slo, ctx = setup("Qwen/Qwen3-1.7B", "data/trace_shared.jsonl", 500, 250, qps=16)
runner = MethodRunner("stage3-t2", fp, slo, "data/trace_shared.jsonl",
                      "runs/stage3-t2-residency", benchmarks=[], quality_every=False)

print("  preflight", flush=True)
live = [(n, c) for n, c in ARMS if preflight(runner, n, c)]
if not any(n == "control_256" for n, _ in live):
    raise SystemExit("  control will not start, nothing to compare against")

rows = []
for name, cfg in live:
    print(f"\n  === {name} ===", flush=True)
    rows.append((name, runner.measure(cfg, name, levels=LEVELS)))

print("\n  RESULT")
print(f"    {'arm':14s} {'goodput':>9s} {'L':>5s} {'ttft_p99':>10s} {'slo':>7s} {'vs control':>11s}")
control = next(t for n, t in rows if n == "control_256")
for name, t in rows:
    d = t.diagnostics or {}
    att = d.get("slo_attainment")
    ttft = f"{t.ttft_p99_ms:.0f}ms" if t.ttft_p99_ms not in (None, float("inf")) else "-"
    ratio = (t.goodput / control.goodput) if control.goodput else 0.0
    print(f"    {name:14s} {t.goodput or 0:9.1f} "
          f"{str(t.concurrency if t.concurrency is not None else '-'):>5} {ttft:>10s} "
          f"{(f'{att:.1%}' if att is not None else '-'):>7s} {ratio:10.3f}x")

print("\n  THE CLIFF. This is the claim: does ttft at L=512 stop being a queue?")
for name, t in rows:
    for c in (t.curve or []):
        if c["concurrency"] in (256, 512, 1024):
            print(f"    {name:14s} L={c['concurrency']:<5} goodput {c['goodput']:8.1f}  "
                  f"ttft_p99 {c['ttft_p99_ms']:9.0f}ms  slo {c['slo_attainment']:.2f}  "
                  f"kv {c.get('kv_cache_util', 0):.3f}  pre {c.get('preemptions', 0):.0f}")

scored = [(n, t) for n, t in rows if t.goodput]
best_name, best = max(scored, key=lambda r: r[1].goodput)
ratio = best.goodput / control.goodput if control.goodput else 0.0
print(f"\n  VERDICT  best arm {best_name} at {best.goodput:.1f} tok/s")
if best_name == "control_256":
    print("    REFUTED: raising residency did not help.")
elif ratio > BAND:
    print(f"    SUPPORTED: {ratio:.3f}x clears the {BAND}x band.")
else:
    print(f"    INSIDE THE NOISE BAND: {ratio:.3f}x against {BAND}x. Not a result.")
runner.finish(chosen=None)
