"""Smoke-test optimize() end to end on a GPU. The API path has never launched."""
from inferopt import SLO, optimize

result = optimize(
    model="Qwen/Qwen3-1.7B",
    trace="data/trace_shared.jsonl",
    slo=SLO(ttft_p99_ms=500, itl_p99_ms=250),
    qps=16,
    strategy="yolo",          # 4 launches, the cheapest way to exercise the path
    lossless_only=True,
    run_dir="runs/api-smoke-1.7b",
)
print()
print(result.summary())
print(f"\n  trials {len(result.trials)}  frontier {len(result.frontier)}")
print(f"  replicas at 16 qps: {result.replicas(16)}")
print(f"  saved -> {result.run_dir}/result.json")
