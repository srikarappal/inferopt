"""Stage 3, turn 4 -- does the stagger hold ACROSS launches, not just within one?

Turn 3's dephased arm was four drives inside ONE server process. That controls
for the convoy but not for anything that varies at launch: CUDA graph capture,
allocator layout, where the KV blocks land. Turn 2's 10.6x spread was measured
across launches, so the fix has to be re-tested the same way or the comparison
is not like-for-like.

This is turn 2 re-run verbatim -- same config, same MethodRunner, same three
launches -- with the staggered driver underneath. Turn 2's numbers are printed
alongside so the delta is the only thing left to read.

Not a search. Three launches of ONE config, changing nothing but the driver.
"""
import statistics

from inferopt.methods import MethodRunner, setup

BEST = {
    "gpu_memory_utilization": 0.75, "max_num_seqs": 256, "max_model_len": 6144,
    "enable_prefix_caching": True, "enable_chunked_prefill": False,
    "enforce_eager": False,
}

# Turn 2, same config, convoying driver. Peak-of-curve per launch.
TURN2 = {
    "goodput": [1957.9, 1913.1, 2199.5],
    "ttft_p99_ms": [1115.0, 1250.0, 1057.0],
    "slo_attainment": [0.831, 0.928, 1.000],
}

fp, slo, ctx = setup("Qwen/Qwen3-1.7B", "data/trace_shared.jsonl", 500, 250, qps=16)
r = MethodRunner("stage3-t4", fp, slo, "data/trace_shared.jsonl",
                 "runs/stage3-turn4", benchmarks=[], quality_every=False)

rows = [r.measure(BEST, f"staggered-rep{i+1}") for i in range(3)]
print("\n  three launches, identical configuration, STAGGERED driver:")
for t in rows:
    d = t.diagnostics or {}
    print(f"    gp {t.goodput:8.1f}  L={t.concurrency:<4} ttft_p99 {t.ttft_p99_ms:7.0f}ms  "
          f"p95 {d.get('ttft_p95_ms', float('nan')):6.0f}ms  n={d.get('ttft_n', 0):<5}"
          f"itl {t.itl_p99_ms:5.1f}ms  slo {d.get('slo_attainment', 0):5.1%}")

print("\n  per-concurrency curves (the convoy showed up as non-monotonicity):")
for t in rows:
    pts = " ".join(f"{c['concurrency']}:{c['goodput']:.0f}/{c['ttft_p99_ms']:.0f}ms"
                   for c in (t.curve or []))
    print(f"    {t.node_id:16s} {pts}")


def spread(vals):
    vals = [v for v in vals if v is not None and v == v]
    if len(vals) < 2:
        return None, None, None
    lo, hi = min(vals), max(vals)
    return lo, hi, (hi / lo if lo else float("inf"))


print(f"\n  {'':16s} {'turn 2 (convoy)':>26s}   {'turn 4 (staggered)':>26s}   tightened")
for name, getter in (
    ("goodput", lambda t: t.goodput),
    ("ttft_p99_ms", lambda t: t.ttft_p99_ms),
    ("slo_attainment", lambda t: (t.diagnostics or {}).get("slo_attainment")),
):
    olo, ohi, ox = spread(TURN2[name])
    nlo, nhi, nx = spread([getter(t) for t in rows])
    if nx is None:
        continue
    gain = f"{ox / nx:5.1f}x" if nx else "   --"
    print(f"  {name:16s} {olo:9.1f}..{ohi:9.1f} ({ox:5.2f}x)   "
          f"{nlo:9.1f}..{nhi:9.1f} ({nx:5.2f}x)   {gain}")

print("\n  medians:")
for name, getter in (("goodput", lambda t: t.goodput),
                     ("ttft_p99_ms", lambda t: t.ttft_p99_ms)):
    new = [v for v in (getter(t) for t in rows) if v is not None]
    print(f"    {name:16s} turn 2 {statistics.median(TURN2[name]):8.1f}   "
          f"turn 4 {statistics.median(new):8.1f}")

print("\n  READ: if the turn-4 bands are tight, the search finally has a")
print("        reproducible objective and every stage-2 keep/revert decision")
print("        taken on a <1.3x margin needs re-checking against it.")
r.finish(chosen=None)
