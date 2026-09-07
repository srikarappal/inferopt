"""Re-measure ONE config to confirm kv_cache_util is now non-zero.

The winning 1.7B config from PB: prefix_caching + graph_capture +
max_model_len_rightsize, which measured 2333.6 tok/s at L=128 with the gauge
reading 0.000 -- as did all 32 other trials.
"""
from inferopt import SLO, optimize

r = optimize(
    model="Qwen/Qwen3-1.7B", trace="data/trace_shared.jsonl",
    slo=SLO(ttft_p99_ms=500, itl_p99_ms=250), qps=16,
    strategy="yolo", repeats=1, lossless_only=True,
    benchmarks=[],                       # no scoring; this is a telemetry check
    run_dir="runs/kv-verify-1.7b",
)
print()
for t in r.trials:
    d = t.diagnostics or {}
    print(f"  {t.node_id:16s} goodput {t.goodput:8.1f} L={t.concurrency}"
          f"  kv_cache_util={d.get('kv_cache_util')}"
          f"  preemptions={d.get('preemptions')}"
          f"  prefix_hit={d.get('prefix_hit_rate')}")
