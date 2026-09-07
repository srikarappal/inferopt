"""Stage 3, turn 5 -- turn 4 re-run with BOTH driver fixes.

Turn 4 had the stagger only: workers started apart, but every request still
asked for exactly int(mean_output_tokens) = 259 tokens, so identical durations
kept pulling them back into lockstep. Turn 5 adds the real fix -- each request
carries its own output length from the trace row its prompt came from (mean
259.6, sd 167.7, 37..1464).

Same config, same three launches, third time. The comparison is three-way:

    turn 2   convoy          all workers start together, constant length
    turn 4   stagger only    start apart, constant length pulls them back
    turn 5   stagger + real lengths

The prediction is NOT that turn 5 is tighter than turn 4 everywhere. Real
lengths make the workload genuinely more variable -- a 1464-token request and a
46-token one are not the same request -- so some spread is now signal rather
than artifact. What should hold is that the curve stays monotone and goodput
stays reproducible, because goodput aggregates over every completion.

Not a search. Three launches of ONE config, changing nothing but the driver.
"""
import statistics

from inferopt.methods import MethodRunner, setup

BEST = {
    "gpu_memory_utilization": 0.75, "max_num_seqs": 256, "max_model_len": 6144,
    "enable_prefix_caching": True, "enable_chunked_prefill": False,
    "enforce_eager": False,
}

# Peak-of-curve per launch, same config, earlier drivers.
PRIOR = {
    "turn 2 (convoy)": {
        "goodput": [1957.9, 1913.1, 2199.5],
        "ttft_p99_ms": [1115.0, 1250.0, 1057.0],
        "slo_attainment": [0.831, 0.928, 1.000],
    },
    "turn 4 (stagger)": {
        "goodput": [2025.4, 2096.7, 2159.6],
        "ttft_p99_ms": [273.5, 1229.6, 1392.0],
        "slo_attainment": [1.000, 0.874, 0.909],
    },
}
# Per-concurrency, where the comparison is actually like-for-like: turn 4's
# peak flipped between L=128 and L=256, so its headline row mixes two levels.
TURN4_BY_L = {
    4: ([223.9, 222.4, 222.2], [123, 125, 125]),
    8: ([413.4, 412.0, 411.2], [147, 146, 148]),
    16: ([740.3, 739.4, 731.6], [125, 105, 151]),
    32: ([1161.6, 1162.9, 1163.0], [179, 206, 180]),
    64: ([1728.4, 1710.5, 1706.3], [172, 131, 129]),
    128: ([2025.4, 2058.9, 2010.7], [274, 286, 274]),
}

fp, slo, ctx = setup("Qwen/Qwen3-1.7B", "data/trace_shared.jsonl", 500, 250, qps=16)
r = MethodRunner("stage3-t5", fp, slo, "data/trace_shared.jsonl",
                 "runs/stage3-turn5", benchmarks=[], quality_every=False)

rows = [r.measure(BEST, f"bothfixes-rep{i+1}") for i in range(3)]
print("\n  three launches, identical configuration, STAGGER + REAL LENGTHS:")
for t in rows:
    d = t.diagnostics or {}
    print(f"    gp {t.goodput:8.1f}  L={t.concurrency:<4} ttft_p99 {t.ttft_p99_ms:7.0f}ms  "
          f"p95 {d.get('ttft_p95_ms', float('nan')):7.0f}ms  n={d.get('ttft_n', 0):<5}"
          f"itl {t.itl_p99_ms:6.2f}ms  slo {d.get('slo_attainment', 0):5.1%}")


def band(vals):
    vals = [v for v in vals if v is not None and v == v]
    if len(vals) < 2:
        return None
    lo, hi = min(vals), max(vals)
    return lo, hi, (hi / lo if lo else float("inf"))


print("\n  === per concurrency, turn 4 vs turn 5 (like-for-like) ===")
print(f"    {'L':>4}  {'turn 4 goodput':>24s}  {'turn 5 goodput':>24s}   "
      f"{'t4 ttft p99':>18s}  {'t5 ttft p99':>18s}")
for L in sorted(TURN4_BY_L):
    g5 = [c["goodput"] for t in rows for c in (t.curve or []) if c["concurrency"] == L]
    tt5 = [c["ttft_p99_ms"] for t in rows for c in (t.curve or []) if c["concurrency"] == L]
    g4, tt4 = TURN4_BY_L[L]
    def fmt(v):
        b = band(v)
        return f"{b[0]:8.0f}..{b[1]:8.0f} {b[2]:4.2f}x" if b else "            --"
    print(f"    {L:>4}  {fmt(g4)}  {fmt(g5)}   {fmt(tt4)[:18]:>18s}  {fmt(tt5)[:18]:>18s}")

print("\n  === peak-of-curve, all three drivers ===")
for metric in ("goodput", "ttft_p99_ms", "slo_attainment"):
    get = {"goodput": lambda t: t.goodput,
           "ttft_p99_ms": lambda t: t.ttft_p99_ms,
           "slo_attainment": lambda t: (t.diagnostics or {}).get("slo_attainment")}[metric]
    print(f"\n    {metric}")
    for name, d in PRIOR.items():
        b = band(d[metric])
        print(f"      {name:18s} {b[0]:9.1f}..{b[1]:9.1f} ({b[2]:5.2f}x)  "
              f"median {statistics.median(d[metric]):9.1f}")
    cur = [v for v in (get(t) for t in rows) if v is not None]
    b = band(cur)
    if b:
        print(f"      {'turn 5 (both)':18s} {b[0]:9.1f}..{b[1]:9.1f} ({b[2]:5.2f}x)  "
              f"median {statistics.median(cur):9.1f}")

print("\n  curves:")
for t in rows:
    pts = " ".join(f"{c['concurrency']}:{c['goodput']:.0f}/{c['ttft_p99_ms']:.0f}ms"
                   for c in (t.curve or []))
    print(f"    {t.node_id:18s} {pts}")

print("\n  READ: goodput reproducible and the curve monotone means the objective")
print("        is sound. Wider TTFT than turn 4 is EXPECTED and is not a")
print("        regression -- the workload now contains 1464-token requests that")
print("        the constant-259 driver never issued.")
r.finish(chosen=None)
