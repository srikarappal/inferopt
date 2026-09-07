"""Stage 3, turn 2 -- how noisy is TTFT p99 across launches?

Turn 1's CONTROL arm re-measured the config PB reported at 2337.4 tok/s /
301ms TTFT / 100% SLO and got 2194.5 / 1088ms / 91%. Goodput moved 6%; TTFT p99
moved 3.6x.

Every SLO-attainment figure in this project is derived from TTFT p99, and
several stage-2 keep/revert decisions were effectively decided by it. If its
launch-to-launch spread is of that order then "100% attainment" and "91%" are
not distinguishable, and no TTFT hypothesis is worth a launch until the band is
known.

Not a search. Three launches of ONE config, changing nothing.
"""
import statistics

from inferopt.methods import MethodRunner, setup

BEST = {
    "gpu_memory_utilization": 0.75, "max_num_seqs": 256, "max_model_len": 6144,
    "enable_prefix_caching": True, "enable_chunked_prefill": False,
    "enforce_eager": False,
}

fp, slo, ctx = setup("Qwen/Qwen3-1.7B", "data/trace_shared.jsonl", 500, 250, qps=16)
r = MethodRunner("stage3-t2", fp, slo, "data/trace_shared.jsonl",
                 "runs/stage3-turn2", benchmarks=[], quality_every=False)

rows = [r.measure(BEST, f"identical-rep{i+1}") for i in range(3)]
print("\n  three launches, identical configuration:")
for t in rows:
    d = t.diagnostics or {}
    print(f"    gp {t.goodput:8.1f}  L={t.concurrency:<4} ttft_p99 {t.ttft_p99_ms:7.0f}ms  "
          f"itl {t.itl_p99_ms:5.1f}ms  slo {d.get('slo_attainment', 0):5.1%}  "
          f"kv {d.get('kv_cache_util', 0):.3f}")

def spread(vals, name):
    vals = [v for v in vals if v is not None]
    if len(vals) < 2:
        return
    lo, hi = min(vals), max(vals)
    print(f"    {name:16s} {lo:9.1f} .. {hi:9.1f}   spread {hi/lo if lo else 0:5.2f}x   "
          f"median {statistics.median(vals):9.1f}")

print("\n  spread across identical launches:")
spread([t.goodput for t in rows], "goodput")
spread([t.ttft_p99_ms for t in rows], "ttft_p99_ms")
spread([t.itl_p99_ms for t in rows], "itl_p99_ms")
spread([(t.diagnostics or {}).get("slo_attainment") for t in rows], "slo_attainment")
r.finish(chosen=None)
