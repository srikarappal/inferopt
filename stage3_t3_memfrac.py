"""Stage 3, turn 3: is 0.75 of a UNIFIED memory pool the wrong reservation?

OBSERVE. Turns 1 and 2 located the binding constraint precisely. At the peak,
L=256, every arm measured ttft_p99 between 511 and 519ms against a 500ms target
with attainment 0.95, while itl_p99 sat at 134ms against a 250ms target. The
peak is pinned by TTFT crossing the SLO line. It is not pinned by capacity, by
prefill scheduling (turn 1, refuted) or by residency (turn 2, mechanism
confirmed and worth 32x on the cliff but nothing on the peak).

Meanwhile kv_cache_util at that peak is 0.184 across every arm tried. The server
reserves gpu_memory_utilization 0.75 of a 122GB pool and touches under a fifth
of the KV it carves out of it.

HYPOTHESISE. On GB10 that fraction is not GPU memory, it is SYSTEM memory the
CPU also uses. 0.75 is in the config as a boot requirement, chosen because 0.90
ran a 122GB box into the OOM killer, and it has never been swept as a
performance knob. Reserving 91GB to use 17GB of KV leaves the host competing for
what is left, and on a part where accelerator and host share one memory
controller that contention costs bandwidth the prefill needs. If so, reserving
less should lower TTFT at L=256 and let more requests land inside the target.

PREDICT. If right, a lower fraction raises goodput at the peak by pulling
ttft_p99 back under 500ms, and kv_cache_util rises without preemptions
appearing, since the pool shrinks while the working set does not. Clearing
1.065x over the control is the bar. If wrong, goodput is flat, which says the
sharing is not costing anything, or it falls with preemptions above zero, which
says the pool was not oversized after all and 0.184 was measuring the wrong
thing.

WHY THE DAG COULD NOT FIND THIS. hardware_defaults sets 0.75 on unified memory
as a launch requirement and no node varies it. Every configuration in every run
on this host, across all three models and all five methods, used 0.75.
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

# 0.45 still leaves ~55GB, and the peak touches 17GB of KV, so even the smallest
# arm has three times the working set. 0.85 tests the other direction: if
# contention is the mechanism, more reservation should be worse, not neutral.
ARMS = [
    ("frac_0.75_control", dict(INCUMBENT)),
    ("frac_0.45", {**INCUMBENT, "gpu_memory_utilization": 0.45}),
    ("frac_0.60", {**INCUMBENT, "gpu_memory_utilization": 0.60}),
    ("frac_0.85", {**INCUMBENT, "gpu_memory_utilization": 0.85}),
]

BAND = 1.065
LEVELS = (32, 64, 128, 256, 512)


def wait_for_gpu(poll_s: int = 120) -> None:
    waited = 0
    while True:
        busy = subprocess.run(
            ["pgrep", "-f", r"inferopt\.pb_screen|inferopt\.run optimize"
                            r"|inferopt\.yolo_run|inferopt\.eval_repro|stage3_t2_"],
            capture_output=True).returncode == 0
        if not busy:
            return
        if waited % 1800 == 0:
            print(f"  waiting for the gpu ({waited // 60}m)", flush=True)
        time.sleep(poll_s)
        waited += poll_s


def preflight(runner, name: str, cfg: dict) -> bool:
    """0.85 may simply not boot on this part. Find out in two minutes."""
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
runner = MethodRunner("stage3-t3", fp, slo, "data/trace_shared.jsonl",
                      "runs/stage3-t3-memfrac", benchmarks=[], quality_every=False)

print("  preflight", flush=True)
live = [(n, c) for n, c in ARMS if preflight(runner, n, c)]
if not any(n == "frac_0.75_control" for n, _ in live):
    raise SystemExit("  control will not start, nothing to compare against")

rows = []
for name, cfg in live:
    print(f"\n  === {name} ===", flush=True)
    rows.append((name, runner.measure(cfg, name, levels=LEVELS)))

print("\n  RESULT")
print(f"    {'arm':20s} {'goodput':>9s} {'L':>5s} {'ttft_p99':>10s} {'slo':>7s} "
      f"{'kv':>6s} {'pre':>5s} {'vs control':>11s}")
control = next(t for n, t in rows if n == "frac_0.75_control")
for name, t in rows:
    d = t.diagnostics or {}
    att, kv, pre = d.get("slo_attainment"), d.get("kv_cache_util"), d.get("preemptions")
    ttft = f"{t.ttft_p99_ms:.0f}ms" if t.ttft_p99_ms not in (None, float("inf")) else "-"
    ratio = (t.goodput / control.goodput) if control.goodput else 0.0
    print(f"    {name:20s} {t.goodput or 0:9.1f} "
          f"{str(t.concurrency if t.concurrency is not None else '-'):>5} {ttft:>10s} "
          f"{(f'{att:.1%}' if att is not None else '-'):>7s} "
          f"{(f'{kv:.3f}' if kv is not None else '-'):>6s} "
          f"{(f'{pre:.0f}' if pre is not None else '-'):>5s} {ratio:10.3f}x")

print("\n  ttft at the peak level, which is what the hypothesis is about:")
for name, t in rows:
    for c in (t.curve or []):
        if c["concurrency"] == 256:
            print(f"    {name:20s} L=256  goodput {c['goodput']:8.1f}  "
                  f"ttft_p99 {c['ttft_p99_ms']:7.0f}ms  slo {c['slo_attainment']:.3f}  "
                  f"kv {c.get('kv_cache_util', 0):.3f}  pre {c.get('preemptions', 0):.0f}")

scored = [(n, t) for n, t in rows if t.goodput]
best_name, best = max(scored, key=lambda r: r[1].goodput)
ratio = best.goodput / control.goodput if control.goodput else 0.0
print(f"\n  VERDICT  best arm {best_name} at {best.goodput:.1f} tok/s")
if best_name == "frac_0.75_control":
    print("    REFUTED: 0.75 is not leaving performance on the table.")
elif ratio > BAND:
    print(f"    SUPPORTED: {ratio:.3f}x clears the {BAND}x band.")
else:
    print(f"    INSIDE THE NOISE BAND: {ratio:.3f}x against {BAND}x. Not a result.")
runner.finish(chosen=None)
