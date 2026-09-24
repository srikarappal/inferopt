"""Measure each kernel against what it replaces, on this card, before anyone
says it is faster. `python -m vi_kernels.bench` prints the table and writes
bench.json beside it.

Shapes are the ones the walk measured: LLaDA2.0-mini's experts, Wan 1.3B and
Z-Image's blocks, Wan's VAE. Reference is eager PyTorch; where SGLang's own
kernel imports, it is timed too, because beating eager is not the bar.
"""

from __future__ import annotations

import json
import sys
import time

import torch

from vi_kernels import dit_fusions, dllm_attention, moe_gemv, vae_norm


def timed(fn, iters=20, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _publish_sglang_context():
    """fused_experts reads SGLang's process-wide config (deterministic mode and
    the like), which only a server publishes. Publish a minimal one so the
    kernel runs standalone; the model path is only parsed, never loaded."""
    from sglang.srt import runtime_context
    if runtime_context._CONTEXT.is_config_namespace_published("exec"):
        return
    from sglang.srt.server_args import ServerArgs
    role = next(iter(runtime_context.ROLE_NAMESPACE_SETS))
    runtime_context.publish(ServerArgs(model_path="inclusionAI/LLaDA2.0-mini", trust_remote_code=True), role=role)


def bench_moe(dev, dtype):
    # LLaDA2.0-mini: hidden 2048, expert intermediate 512, 256 experts, top 8
    K, N, E, T = 2048, 512, 256, 8
    w1 = torch.randn(E, 2 * N, K, device=dev, dtype=dtype) * 0.02
    w2 = torch.randn(E, K, N, device=dev, dtype=dtype) * 0.02
    rows = []
    for M in (1, 4, 8, 16):
        x = torch.randn(M, K, device=dev, dtype=dtype)
        ids = torch.stack([torch.randperm(E, device=dev)[:T] for _ in range(M)]).to(torch.int32)
        wts = torch.softmax(torch.randn(M, T, device=dev), dim=-1)
        ours = timed(lambda: moe_gemv.moe_small_m(x, w1, w2, ids, wts))
        row = {"kernel": "moe_small_m", "M": M, "ours_ms": round(ours, 3)}
        try:
            _publish_sglang_context()
            from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_experts
            from sglang.srt.layers.moe.topk import StandardTopKOutput
            from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
            topk = StandardTopKOutput(topk_weights=wts, topk_ids=ids, router_logits=None)
            cfg = MoeRunnerConfig()
            theirs = fused_experts(x, w1, w2, topk, cfg)
            ours_out = moe_gemv.moe_small_m(x, w1, w2, ids, wts)
            row["max_abs_diff_vs_sglang"] = round((theirs.float() - ours_out.float()).abs().max().item(), 4)
            row["sglang_fused_moe_ms"] = round(timed(lambda: fused_experts(x, w1, w2, topk, cfg)), 3)
        except Exception as e:
            row["sglang_fused_moe_ms"] = f"n/a ({type(e).__name__}: {str(e)[:80]})"
        rows.append(row)
    return rows


def bench_dllm(dev, dtype):
    # LLaDA2.0-mini: 16 heads, 4 kv heads, head dim 128; prefix 512, block 32
    H, HKV, D, P, B = 16, 4, 128, 512, 32
    q = torch.randn(B, H, D, device=dev, dtype=dtype)
    k = torch.randn(P + B, HKV, D, device=dev, dtype=dtype)
    v = torch.randn(P + B, HKV, D, device=dev, dtype=dtype)
    rows = []
    for active in (32, 16, 8, 2):
        idx = torch.arange(active, device=dev, dtype=torch.int32)
        all_rows = torch.arange(B, device=dev, dtype=torch.int32)
        ours = timed(lambda: dllm_attention.dllm_block_attention(q, k, v, idx))
        full = timed(lambda: dllm_attention.dllm_block_attention(q, k, v, all_rows))
        sdpa = timed(lambda: torch.nn.functional.scaled_dot_product_attention(
            q.permute(1, 0, 2)[None], k.repeat_interleave(H // HKV, 1).permute(1, 0, 2)[None],
            v.repeat_interleave(H // HKV, 1).permute(1, 0, 2)[None]))
        rows.append({"kernel": "dllm_block_attention", "active_rows": active, "of": B,
                     "ours_ms": round(ours, 4), "ours_full_block_ms": round(full, 4),
                     "sdpa_full_block_ms": round(sdpa, 4)})
    return rows


def bench_dit(dev, dtype):
    rows = []
    # Z-Image at 1024^2: 4096 tokens x 3840; Wan 1.3B at 480p x 33 frames: ~32k tokens x 1536
    for name, S, C in (("z-image 1024", 4096, 3840), ("wan1.3b 480p33f", 32760, 1536)):
        x = torch.randn(1, S, C, device=dev, dtype=dtype)
        shift = torch.randn(1, 1, C, device=dev, dtype=torch.float32)
        scale = torch.randn(1, 1, C, device=dev, dtype=torch.float32)
        gate = torch.randn(1, 1, C, device=dev, dtype=torch.float32)
        y = torch.randn_like(x)
        rows.append({"kernel": "modulated_layernorm", "shape": name,
                     "ours_ms": round(timed(lambda: dit_fusions.modulated_layernorm(x, shift, scale)), 3),
                     "eager_ms": round(timed(lambda: dit_fusions.modulated_layernorm_reference(x, shift, scale)), 3)})
        rows.append({"kernel": "gated_residual", "shape": name,
                     "ours_ms": round(timed(lambda: dit_fusions.gated_residual(x, gate, y)), 3),
                     "eager_ms": round(timed(lambda: dit_fusions.gated_residual_reference(x, gate, y)), 3)})
    return rows


def bench_vae(dev, dtype):
    # Wan VAE decoder mid stage: 384 channels, 9 latent frames x 60 x 104 upsampled
    x = torch.randn(1, 384, 9, 120, 208, device=dev, dtype=dtype)
    gamma = torch.randn(384, device=dev, dtype=torch.float32)
    bias = torch.randn(384, device=dev, dtype=torch.float32)
    return [{"kernel": "channel_rmsnorm_silu", "shape": "1x384x9x120x208",
             "ours_ms": round(timed(lambda: vae_norm.channel_rmsnorm_silu(x, gamma, bias)), 3),
             "eager_ms": round(timed(lambda: vae_norm.channel_rmsnorm_silu_reference(x, gamma, bias)), 3)}]


def main(out="bench.json"):
    dev, dtype = "cuda", torch.bfloat16
    rows = bench_moe(dev, dtype) + bench_dllm(dev, dtype) + bench_dit(dev, dtype) + bench_vae(dev, dtype)
    for r in rows:
        print(json.dumps(r))
    json.dump({"device": torch.cuda.get_device_name(), "rows": rows, "at": time.time()},
              open(out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
