"""How many GPUs a model needs, and how many it can use.

    tp = recommend_tp(fp)
    tp.size          # what to serve with
    tp.why           # one line, for the run banner

TENSOR PARALLELISM IS A PRECONDITION, NOT AN OPTIMISATION. Every DAG node is
measured against an incumbent that already serves; a model that does not fit on
one GPU has no incumbent to improve on. So TP is chosen before the search, not
inside it, and the minimum that FITS is arithmetic rather than preference --
refusing to pick one turns a computable answer into an out-of-memory stack
trace.

WHAT SETS THE CEILING, and it is not memory:

  KV HEADS. KV cache shards by kv_head, so aggregate KV capacity grows with TP
  only up to n_kv_heads. Past that vLLM REPLICATES kv heads across ranks and
  memory stops scaling: Qwen3-14B has 8, Qwen3-30B-A3B has 4. TP=16 on the 14B
  satisfies every memory check and buys no additional KV whatsoever.

  THE NODE BOUNDARY. Within a host the allreduce runs over NVLink at roughly
  900 GB/s. Cross-host it drops to InfiniBand or PCIe -- an order of magnitude
  -- and it sits on the critical path of every layer of every token. Past one
  node, more TP makes decode SLOWER, not merely more expensive.

  DIVISIBILITY. vLLM requires the attention head count to divide by TP, so the
  candidates are not arbitrary.

Above the fit floor, TP is a genuine trade the fingerprint cannot rank: it
lowers per-device weight traffic, which improves the decode roofline, and it
adds a collective per layer, which worsens it. That choice stays the caller's.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TPChoice:
    size: int
    fits: bool
    floor: int
    """Smallest TP the model fits on at all."""
    ceiling: int
    """Largest TP that buys anything -- min(kv heads, gpus per node)."""
    why: str
    warnings: tuple[str, ...] = ()


def _per_device_gb(fp, tp: int, ctx_tokens: int) -> float:
    """Weights plus one max-length sequence's KV, on ONE device at this TP."""
    w = fp.model.weight_gb / tp
    # KV shards by kv_head and REPLICATES past that, so the divisor saturates.
    kv_shards = min(tp, max(1, fp.model.n_kv_heads))
    kv = (fp.model.kv_bytes_per_token * ctx_tokens / 1e9) / kv_shards
    return w + kv


def recommend_tp(fp, *, util: float | None = None, gpus_per_node: int = 8,
                 overhead_gb: float = 4.0, ctx_tokens: int | None = None) -> TPChoice:
    """Lowest TP that fits, reported with the ceiling past which TP is waste."""
    util = util if util is not None else (0.75 if fp.hw.unified_memory else 0.90)
    ctx = ctx_tokens or min(fp.model.max_model_len, 8192)
    budget = fp.hw.memory_gb * util - overhead_gb

    kv_heads = max(1, fp.model.n_kv_heads)
    per_node = min(gpus_per_node, fp.hw.gpu_count)
    ceiling = max(1, min(kv_heads, per_node))

    # vLLM requires the attention head count to divide by TP.
    cands = [t for t in (1, 2, 4, 8, 16, 32, 64)
             if t <= fp.hw.gpu_count and fp.model.n_heads % t == 0]
    if not cands:
        cands = [1]

    floor = next((t for t in cands if _per_device_gb(fp, t, ctx) <= budget), 0)
    warn: list[str] = []

    if not floor:
        biggest = cands[-1]
        return TPChoice(
            size=biggest, fits=False, floor=0, ceiling=ceiling,
            why=(f"does NOT fit: {_per_device_gb(fp, biggest, ctx):.1f} GB per "
                 f"device at TP={biggest} against a {budget:.1f} GB budget"),
            warnings=("no TP on this host is enough; quantize the weights or add "
                      "hosts",))

    size = floor
    if floor > ceiling:
        # Fitting REQUIRES more TP than is useful. Legal, and worth saying out
        # loud: the extra ranks add collectives and no KV.
        if floor > kv_heads:
            warn.append(
                f"TP={floor} exceeds the model's {kv_heads} KV heads, so KV "
                f"capacity stops growing past TP={kv_heads} -- the extra ranks "
                f"replicate KV rather than adding any")
        if floor > per_node:
            warn.append(
                f"TP={floor} crosses the {per_node}-GPU node boundary, so every "
                f"layer's allreduce leaves NVLink for the fabric -- decode gets "
                f"slower, not just costlier")

    return TPChoice(
        size=size, fits=True, floor=floor, ceiling=ceiling,
        why=(f"TP={size}: {_per_device_gb(fp, size, ctx):.1f} GB per device "
             f"against a {budget:.1f} GB budget "
             f"({fp.model.weight_gb:.0f} GB weights, {ctx} ctx)"),
        warnings=tuple(warn))
