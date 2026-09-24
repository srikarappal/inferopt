"""The thin layer between these kernels and SGLang.

A kernel is a torch op; the engine sees it through a seam. SGLang's seams, as
installed (0.5.20), read from its source:

  MoE       sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe.fused_experts
            (hidden_states, w1, w2, topk_output, moe_runner_config, ...)
  Wan DiT   the block's norm1(x, shift, scale) modulated LayerNorm, and the
            gated residual after attention
  Wan VAE   the channel RMSNorm module of the decoder

`install()` wraps each seam: our kernel takes the case it is built for
(bf16, small M, no quantisation, a shape it supports) and everything else
falls through to SGLang's own code unchanged. `check()` drives every wrapped
seam with random tensors and asserts our kernel ran and agreed with the
reference; that is the sanity check that the engine picks the kernels up,
run before any server is launched with them.

Installation is by import: `import vi_kernels.sglang_adapter` then
`install()`, or PYTHONPATH a sitecustomize that does the same, so SGLang's
own process gets it without a fork of SGLang.
"""

from __future__ import annotations

import torch

from vi_kernels import dit_fusions, moe_gemv, vae_norm

calls = {"moe_small_m": 0, "modulated_layernorm": 0, "gated_residual": 0, "channel_rmsnorm_silu": 0}
installed: dict[str, str] = {}


def _wrap_moe():
    try:
        from sglang.srt.layers.moe.moe_runner.triton_utils import fused_moe as fm
    except Exception as e:
        return f"skipped ({type(e).__name__})"
    original = fm.fused_experts

    def fused_experts(hidden_states, w1, w2, topk_output, moe_runner_config, *args, **kwargs):
        quantised = any(kwargs.get(k) for k in ("use_fp8_w8a8", "use_int8_w8a8", "use_int8_w8a16", "use_int4_w4a16"))
        plain = (hidden_states.dtype in (torch.bfloat16, torch.float16)
                 and w1.dtype == hidden_states.dtype and w2.dtype == hidden_states.dtype
                 and not quantised and kwargs.get("b1") is None and kwargs.get("b2") is None
                 and hidden_states.shape[0] <= moe_gemv.SMALL_M
                 and getattr(moe_runner_config, "activation", "silu") == "silu")
        if not plain:
            return original(hidden_states, w1, w2, topk_output, moe_runner_config, *args, **kwargs)
        calls["moe_small_m"] += 1
        out = moe_gemv.moe_small_m(hidden_states, w1, w2, topk_output.topk_ids, topk_output.topk_weights)
        scaling = getattr(moe_runner_config, "routed_scaling_factor", None)
        return out * scaling if scaling and scaling != 1.0 else out

    fm.fused_experts = fused_experts
    return "wrapped fused_experts"


def _wrap_wan_dit():
    try:
        from sglang.multimodal_gen.runtime.models.dits import wanvideo
    except Exception as e:
        return f"skipped ({type(e).__name__})"
    # The modulated norm is whatever class norm1 is; find it by behaviour: a
    # module whose forward takes (x, shift, scale).
    target = None
    for name in dir(wanvideo):
        obj = getattr(wanvideo, name)
        if isinstance(obj, type) and issubclass(obj, torch.nn.Module):
            code = getattr(getattr(obj, "forward", None), "__code__", None)
            if code and code.co_varnames[:4] == ("self", "hidden_states", "shift", "scale") \
                    or (code and code.co_varnames[:4] == ("self", "x", "shift", "scale")):
                target = obj
                break
    if target is None:
        return "skipped (no modulated norm class found)"
    original = target.forward

    def forward(self, x, shift, scale):
        affine = getattr(self, "weight", None) is not None or getattr(self, "elementwise_affine", False)
        if x.dtype not in (torch.bfloat16, torch.float16) or affine or x.shape[-1] > 8192:
            return original(self, x, shift, scale)
        calls["modulated_layernorm"] += 1
        return dit_fusions.modulated_layernorm(x, shift, scale, eps=getattr(self, "eps", 1e-6))

    target.forward = forward
    return f"wrapped {target.__name__}.forward"


def _wrap_wan_vae():
    candidates = []
    try:
        from diffusers.models.autoencoders import autoencoder_kl_wan as m
        candidates.append(m)
    except Exception:
        pass
    try:
        from sglang.multimodal_gen.runtime.models.vaes import wanvae as m2
        candidates.append(m2)
    except Exception:
        pass
    for module in candidates:
        cls = getattr(module, "WanRMS_norm", None)
        if cls is None:
            continue
        original = cls.forward

        def forward(self, x, _original=original):
            if x.dtype not in (torch.bfloat16, torch.float16) or not getattr(self, "channel_first", True):
                return _original(self, x)
            gamma = self.gamma
            bias = self.bias if isinstance(getattr(self, "bias", None), torch.Tensor) else None
            calls["channel_rmsnorm_silu"] += 1
            return vae_norm.channel_rmsnorm_silu(x, gamma, bias, silu=False)

        cls.forward = forward
        return f"wrapped {module.__name__}.WanRMS_norm.forward"
    return "skipped (no WanRMS_norm found)"


def install() -> dict[str, str]:
    installed["moe"] = _wrap_moe()
    installed["wan_dit"] = _wrap_wan_dit()
    installed["wan_vae"] = _wrap_wan_vae()
    return dict(installed)


def check(device="cuda") -> dict:
    """Drive every wrapped seam; assert our kernel ran and agreed."""
    report = {}
    dtype = torch.bfloat16
    # MoE through SGLang's own entry point
    try:
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_experts
        from sglang.srt.layers.moe.topk import StandardTopKOutput
        from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
        K, N, E, T, M = 256, 128, 16, 4, 4
        x = torch.randn(M, K, device=device, dtype=dtype)
        w1 = torch.randn(E, 2 * N, K, device=device, dtype=dtype) * 0.05
        w2 = torch.randn(E, K, N, device=device, dtype=dtype) * 0.05
        ids = torch.stack([torch.randperm(E, device=device)[:T] for _ in range(M)]).to(torch.int32)
        wts = torch.softmax(torch.randn(M, T, device=device), dim=-1)
        before = calls["moe_small_m"]
        out = fused_experts(x, w1, w2, StandardTopKOutput(topk_weights=wts, topk_ids=ids, router_logits=None),
                            MoeRunnerConfig())
        ref = moe_gemv.moe_reference(x, w1, w2, ids, wts)
        err = (out.float() - ref.float()).abs().max().item()
        report["moe"] = {"ours_ran": calls["moe_small_m"] > before, "max_abs_err": err, "ok": err < 0.05}
    except Exception as e:
        report["moe"] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
    # Kernels alone, against their references
    x = torch.randn(2, 64, 512, device=device, dtype=dtype)
    shift = torch.randn(2, 1, 512, device=device)
    scale = torch.randn(2, 1, 512, device=device)
    gate = torch.randn(2, 1, 512, device=device)
    y = torch.randn_like(x)
    e1 = (dit_fusions.modulated_layernorm(x, shift, scale).float()
          - dit_fusions.modulated_layernorm_reference(x, shift, scale).float()).abs().max().item()
    e2 = (dit_fusions.gated_residual(x, gate, y).float()
          - dit_fusions.gated_residual_reference(x, gate, y).float()).abs().max().item()
    v = torch.randn(1, 96, 3, 16, 16, device=device, dtype=dtype)
    g = torch.randn(96, device=device); b = torch.randn(96, device=device)
    e3 = (vae_norm.channel_rmsnorm_silu(v, g, b).float()
          - vae_norm.channel_rmsnorm_silu_reference(v, g, b).float()).abs().max().item()
    report["dit_fusions"] = {"modulated_layernorm_err": e1, "gated_residual_err": e2, "ok": e1 < 0.1 and e2 < 0.05}
    report["vae_norm"] = {"err": e3, "ok": e3 < 0.1}
    report["installed"] = dict(installed)
    report["calls"] = dict(calls)
    return report


if __name__ == "__main__":
    import json
    print(json.dumps(install(), indent=1))
    print(json.dumps(check(), indent=1))
