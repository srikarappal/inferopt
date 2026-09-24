"""Where the time goes, per kernel, for a configuration the walk measured.

The walk answers "which configuration"; it did not answer "what is the
denominator": which kernels a step is made of and how much of the card they
keep busy. Stage 3 needs that to look past the DAG, and a kernel effort needs
it to aim at an op that is 40% of a step on the customer's card rather than
one that looked slow on a test rig.

Every engine ships a torch profiler behind its own switch, and every one of
them writes a Chrome trace. This module reads that trace: GPU kernel events,
summed by name, grouped into families, with the GPU's busy fraction over the
profiled span. The summary is small enough to ride on a trial's diagnostics,
into the journal and result.json; the trace itself stays beside the launch log
for anyone who wants the whole thing.
"""

from __future__ import annotations

import gzip
import json
import re
from collections import defaultdict
from pathlib import Path

# Families a kernel name is sorted into, first match wins. Names are what
# CUDA reports: cutlass and cuBLAS GEMM mangles, flash and flashinfer attention
# symbols, Triton kernel names, and the elementwise kernels PyTorch generates.
FAMILIES = (
    ("attention", re.compile(r"attn|attention|fmha|flash|sage|paged|mla|prefill.*kernel|decode.*kernel", re.I)),
    ("moe", re.compile(r"moe|grouped_gemm|group_gemm|expert|topk|fused_experts|silu_and_mul.*moe", re.I)),
    ("gemm", re.compile(r"gemm|matmul|cutlass|cublas|nvjet|xmma|mm_kernel|sm\d+_.*gemm|wgmma|linear", re.I)),
    ("conv", re.compile(r"conv|cudnn|winograd|implicit_gemm", re.I)),
    ("norm", re.compile(r"norm|rms|layer_norm|group_norm", re.I)),
    ("rope_embed", re.compile(r"rope|rotary|embedding|timestep|pos_?emb", re.I)),
    ("sampling", re.compile(r"sampl|softmax|argmax|topp|top_p|logits|gumbel|multinomial", re.I)),
    ("memory", re.compile(r"memcpy|memset|copy_|cudaMemcpy|Memcpy|Memset|cat_|index_|gather|scatter|fill_", re.I)),
    ("elementwise", re.compile(r"elementwise|vectorized|unrolled|silu|gelu|mul|add|sub|div|cast|to_copy|clamp|sigmoid|exp|sqrt|activation|reduce", re.I)),
    ("allreduce", re.compile(r"nccl|allreduce|all_reduce|all_gather|reduce_scatter", re.I)),
)
GPU_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset", "cuda_kernel"}


def find_traces(directory: str | Path) -> list[Path]:
    """Every Chrome trace under `directory`, newest first."""
    root = Path(directory)
    if not root.exists():
        return []
    found = [p for p in root.rglob("*") if p.is_file()
             and (p.name.endswith(".json.gz") or p.name.endswith(".json"))
             and "trace" in p.name]
    return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)


def _events(path: Path) -> list[dict]:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        data = json.load(fh)
    return data.get("traceEvents", data) if isinstance(data, dict) else data


def family_of(name: str) -> str:
    for family, pattern in FAMILIES:
        if pattern.search(name):
            return family
    return "other"


def summarise_events(events: list[dict], top: int = 25) -> dict | None:
    """The GPU side of a trace, reduced to what a reader can act on."""
    by_name: dict[str, list[float]] = defaultdict(lambda: [0.0, 0])
    span_start, span_end = None, None
    for e in events:
        if e.get("ph") != "X" or e.get("cat") not in GPU_CATEGORIES:
            continue
        dur = float(e.get("dur") or 0.0)        # microseconds
        ts = float(e.get("ts") or 0.0)
        name = str(e.get("name") or "?")
        by_name[name][0] += dur
        by_name[name][1] += 1
        span_start = ts if span_start is None else min(span_start, ts)
        span_end = ts + dur if span_end is None else max(span_end, ts + dur)
    if not by_name:
        return None
    total_us = sum(v[0] for v in by_name.values())
    wall_us = (span_end - span_start) if span_start is not None else 0.0
    families: dict[str, float] = defaultdict(float)
    for name, (us, _) in by_name.items():
        families[family_of(name)] += us
    ranked = sorted(by_name.items(), key=lambda kv: -kv[1][0])
    return {
        "total_gpu_ms": round(total_us / 1000, 2),
        "wall_ms": round(wall_us / 1000, 2),
        # How much of the profiled span the GPU was executing anything. On a
        # single stream this is the utilisation; overlapping streams can push
        # it past one, which is itself worth seeing.
        "gpu_busy": round(total_us / wall_us, 3) if wall_us else None,
        "families": {f: {"ms": round(us / 1000, 2), "pct": round(100 * us / total_us, 1)}
                     for f, us in sorted(families.items(), key=lambda kv: -kv[1])},
        "top_ops": [{"name": name[:120], "family": family_of(name),
                     "ms": round(us / 1000, 2), "pct": round(100 * us / total_us, 1), "calls": calls}
                    for name, (us, calls) in ranked[:top]],
        "distinct_kernels": len(by_name),
    }


def summarise_traces(directory: str | Path, top: int = 25) -> dict | None:
    """Every trace under `directory`, merged. None when there is none yet or
    none holds a GPU event, which the caller records as unprofiled rather
    than as zero."""
    traces = find_traces(directory)
    events: list[dict] = []
    for path in traces:
        try:
            events.extend(_events(path))
        except Exception:
            continue                    # a torn trace is skipped, not fatal
    summary = summarise_events(events, top=top)
    if summary is not None:
        summary["traces"] = [str(p) for p in traces]
    return summary
