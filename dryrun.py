"""A simulated evaluator, so the traversal can be exercised without a GPU.

    python dryrun.py

Not a performance model -- it encodes plausible directional effects plus noise
at the measured level, which is enough to prove the machinery: predicates gate,
sweeps expand, keep/revert respects the band, quality tolerance is learned at
the checkpoint, the budget guard stops with a usable result, and the frontier
keeps reverted configs.

HISTORY

  Written after 3,889 lines of harness produced zero proposals. The lesson was
  that the traversal logic and the measurement rig fail in completely different
  ways, and mixing them means every logic bug costs a launch to find. This runs
  the whole state machine against a synthetic evaluator in about a second.

  It is also what surfaced revert != discard: three of five frontier points in
  the first dry run were REVERTED configs, including an INT4 variant at 2.9x
  throughput and a quarter the memory. A lineage-only frontier discards exactly
  the operating points a user would most want to see.
"""

from __future__ import annotations

import random
from typing import Any

from calibration import STORE
from fingerprint import (
    Context, NodeMeasurement, Fingerprint, HardwareFingerprint, ModelFingerprint, SLO, WorkloadFingerprint,
)
from traverse import Trial, report, traverse

BASE_GOODPUT = 480.0
NOISE = 0.019          # the measured across-launch spread


class DryRunEvaluator:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def measure(self, config: dict[str, Any], *, probes, benchmarks, node_id,
                concurrency=None, levels=None,
                fixed_concurrency=None) -> Trial:
        g, ttft, mem, quality = BASE_GOODPUT, 420.0, 18.0, {}

        if config.get("enable_chunked_prefill"):
            g *= 1.06
            ttft *= 1.35 if config.get("max_num_batched_tokens", 8192) <= 2048 else 0.9
        if config.get("enable_prefix_caching"):
            g *= 1.11
        if config.get("max_model_len", 32768) <= 8192:
            g *= 1.04
        spec = config.get("speculative_config") or {}
        if spec:
            k = spec.get("num_speculative_tokens", 3)
            g *= 1.0 + min(0.22, 0.05 * k) - (0.03 if k > 5 else 0.0)
        if config.get("kv_cache_dtype") == "fp8_e4m3":
            g *= 1.18
            mem *= 0.72
        if "fp8" in str(config.get("model", "")).lower():
            g *= 1.35
            mem *= 0.55
        if "awq" in str(config.get("model", "")).lower():
            g *= 1.6
            mem *= 0.32
        g *= 1.0 + 0.04 * min(2.0, config.get("max_num_seqs", 256) / 256 - 1)

        if "quality" in probes:
            for b in benchmarks or ["humaneval_plus"]:
                v = {"humaneval_plus": 0.58, "math_500": 0.44, "ruler_multineedle": 0.92}[b]
                if config.get("kv_cache_dtype") == "fp8_e4m3":
                    v -= {"ruler_multineedle": 0.021}.get(b, 0.004)
                if "fp8" in str(config.get("model", "")).lower():
                    v -= 0.010
                if "awq" in str(config.get("model", "")).lower():
                    v -= {"humaneval_plus": 0.055, "math_500": 0.048}.get(b, 0.02)
                quality[b] = round(v + self.rng.gauss(0, 0.004), 4)

        g *= 1 + self.rng.gauss(0, NOISE / 2)
        ttft *= 1 + self.rng.gauss(0, 0.03)
        return Trial(node_id=node_id, config=dict(config), goodput=round(g, 1),
                     ttft_p99_ms=round(ttft, 1), itl_p99_ms=round(1000 / g * 24, 2),
                     memory_gb=round(mem, 1), quality=quality,
                     diagnostics={"spec_acceptance_rate": 0.31 if spec else None},
                     slo_ok=ttft <= 500 * 1.35)


def build_context() -> Context:
    fp = Fingerprint(
        model=ModelFingerprint(
            id="Qwen/Qwen3.5-9B", architecture="Qwen3ForCausalLM", n_params_b=9.0,
            n_layers=48, hidden_size=4096, n_heads=32, n_kv_heads=8,
            attention_type="gqa", max_model_len=32768,
            can_quantize_fp8=True,
            can_quantize_int4_awq=True),
        hw=HardwareFingerprint(
            gpu_name="NVIDIA GB10", compute_capability="12.1", memory_gb=121.7,
            memory_bandwidth_gb_s=273.0, unified_memory=True,
            system_ram_gb=121.7, cpu_cores=20),
        workload=WorkloadFingerprint(
            n_requests=800, mean_input_tokens=494, p99_input_tokens=1997,
            p999_input_tokens=2710, mean_output_tokens=262, p99_output_tokens=839,
            request_rate_qps=16.7, max_concurrency=33,
            prefix_overlap=0.30, prefix_overlap_per_adapter=0.30),
    )
    return Context(
        fingerprint=fp,
        slo=SLO(ttft_p99_ms=500, itl_p99_ms=30, quality_budget=0.03),
        incumbent={"max_num_seqs": 256, "gpu_memory_utilization": 0.90,
                   "max_model_len": 32768, "block_size": 16,
                   "enable_prefix_caching": False, "enable_chunked_prefill": False,
                   "enforce_eager": False},
        # stage 1.3 measured these before the DAG started
        quality_baseline={"humaneval_plus": 0.58, "math_500": 0.44, "ruler_multineedle": 0.92},
        incumbent_metrics=NodeMeasurement(goodput=480.0, ttft_p99_ms=420.0, itl_p99_ms=50.0),
    )


if __name__ == "__main__":
    import json
    dag = json.load(open("dag/llm.json"))
    ctx = build_context()
    ev = DryRunEvaluator(seed=7)
    # A synthetic baseline so the report's anchor path is exercised. Without one
    # every frontier percentage prints '--', which is exactly the failure this
    # argument exists to prevent.
    base = ev.measure(ctx.incumbent, probes=["goodput"], benchmarks=[], node_id="stage_1_3")
    res = traverse(dag, ctx, ev, baseline=base, concurrency=32)
    report(res)
