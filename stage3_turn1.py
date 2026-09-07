"""Stage 3, turn 1 -- is TTFT at high L head-of-line blocking from long prefills?

Baseline for this turn is the best config any stage-1/2 method found on the
1.7B: prefix caching + CUDA graphs + right-sized context, 2337.4 tok/s at L=128.

The walk reverted plain chunked_prefill at -5.7%, so this is not a re-proposal
of that node. It adds --long-prefill-token-threshold, which no stage-2 run had
available, on a background the walk never reached: it reverted chunked_prefill
while its incumbent was prefix_caching alone, before graph_capture and
right-sizing were in place.

The two arms differ in ONE thing. Anything else and a moved result says nothing.
"""
from inferopt.methods import MethodRunner, setup

BEST = {
    "gpu_memory_utilization": 0.75, "max_num_seqs": 256, "max_model_len": 6144,
    "enable_prefix_caching": True, "enable_chunked_prefill": False,
    "enforce_eager": False,
}
# p99 input is 2660 tokens against a mean of 620. 1024 splits the tail without
# chunking the median request, which would add overhead for no queueing gain.
CHUNKED = {**BEST, "enable_chunked_prefill": True,
           "max_num_batched_tokens": 6144,          # legal: >= max_model_len
           "long_prefill_token_threshold": 1024}

fp, slo, ctx = setup("Qwen/Qwen3-1.7B", "data/trace_shared.jsonl", 500, 250, qps=16)
r = MethodRunner("stage3-t1", fp, slo, "data/trace_shared.jsonl",
                 "runs/stage3-turn1", benchmarks=[], quality_every=False)

for label, cfg in (("control-best", BEST), ("chunked+threshold", CHUNKED)):
    t = r.measure(cfg, label)
    d = t.diagnostics or {}
    print(f"\n  {label:20s} goodput {t.goodput:8.1f}  L={t.concurrency}  "
          f"ttft_p99 {t.ttft_p99_ms:6.0f}ms  itl {t.itl_p99_ms:5.1f}ms  "
          f"slo {d.get('slo_attainment', 0):.0%}  kv {d.get('kv_cache_util', 0):.3f}")
r.finish(chosen=None)
