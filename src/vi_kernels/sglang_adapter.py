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

import os
import sys

import torch

from vi_kernels import dit_fusions, moe_gemv, vae_norm

calls = {"moe_small_m": 0, "modulated_layernorm": 0, "gated_residual": 0, "channel_rmsnorm_silu": 0}
installed: dict[str, str] = {}


def _note(kernel: str) -> None:
    """Count a call; announce the first one so a server log shows the pickup."""
    calls[kernel] += 1
    if calls[kernel] == 1:
        print(f"vi_kernels: {kernel} took its first call  [pid {os.getpid()}]", file=sys.stderr)


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
        _note("moe_small_m")
        out = moe_gemv.moe_small_m(hidden_states, w1, w2, topk_output.topk_ids, topk_output.topk_weights)
        scaling = getattr(moe_runner_config, "routed_scaling_factor", None)
        return out * scaling if scaling and scaling != 1.0 else out

    fm.fused_experts = fused_experts
    return "wrapped fused_experts"


def _wrap_wan_dit():
    """The DiT's modulated norm.

    Found on inspection of 0.5.20: SGLang Diffusion already fuses norm, scale
    and shift in CUDA (sglang.kernels.ops.diffusion.fused_norm_scale_shift)
    whenever the hidden size is a multiple of 256 up to 8192, which covers
    Wan's 1536 and Z-Image's 3840, and it fuses the gated residual into the
    next block's norm beside it. Item 5 of the list is theirs. Our Triton
    version takes only the fallback, forward_native, the shapes their kernel
    refuses; the bench decides whether it should take more.
    """
    try:
        from sglang.multimodal_gen.runtime.layers import layernorm as ln
    except Exception as e:
        return f"skipped ({type(e).__name__})"
    target = getattr(ln, "_NormScaleShift", None)
    if target is None:
        return "skipped (no _NormScaleShift)"
    original = target.forward_native

    def forward_native(self, x, shift, scale):
        weight = getattr(self.norm, "weight", None)
        bias = getattr(self.norm, "bias", None)
        if (getattr(self, "norm_type", "layer") != "layer" or weight is not None or bias is not None
                or x.dtype not in (torch.bfloat16, torch.float16) or x.shape[-1] > 16384):
            return original(self, x, shift, scale)
        _note("modulated_layernorm")
        return dit_fusions.modulated_layernorm(x, shift, scale, eps=getattr(self, "eps", 1e-6))

    target.forward_native = forward_native
    return "wrapped _NormScaleShift.forward_native (the fallback; their CUDA fusion keeps the aligned shapes)"


_vae_seams_done: set[str] = set()


def _wrap_norm_forward(cls, silu: bool, label: str) -> bool:
    """Wrap `cls.forward(self, x)` where the class computes
    F.normalize(x, dim=1) * scale * gamma + bias, optionally followed by SiLU.
    Ours takes channel-first contiguous 5D input in bf16/fp16/fp32; anything
    else goes to the original."""
    if label in _vae_seams_done:
        return False
    original = cls.forward

    def forward(self, x):
        gamma = getattr(self, "gamma", None)
        bias = getattr(self, "bias", None)
        gate = getattr(self, "_sgl_gate", None)
        theirs_would_run = gate is not None and gate.enabled and x.is_contiguous(memory_format=torch.channels_last_3d)
        if (gamma is None or x.ndim != 5 or not x.is_contiguous() or theirs_would_run
                or x.dtype not in (torch.bfloat16, torch.float16, torch.float32)
                or torch.compiler.is_compiling()):
            return original(self, x)
        _note("channel_rmsnorm_silu")
        b = bias.reshape(-1) if isinstance(bias, torch.Tensor) else None
        return vae_norm.channel_rmsnorm_silu(x, gamma.reshape(-1), b, silu=silu)

    cls.forward = forward
    _vae_seams_done.add(label)
    return True


def _wrap_wan_vae():
    """The Wan VAE's RMS norm, in SGLang's own decoder.

    Found on inspection of 0.5.20: SGLang Diffusion already fuses RMSNorm+SiLU
    in the Wan decoder (wan_vae_cuda_opt.FusedWanRMSNormSiLU, a Triton kernel)
    but only when the request's `quality` allows kernel fusions AND the tensor
    is channels_last_3d; the default request runs the eager chain. Ours takes
    the eager cases: the fused class's off path (norm+SiLU), the attention
    block's bare norm (WanRMS_norm, no SiLU) and diffusers' WanRMS_norm for a
    diffusers-backend pipeline. Called once per module as each is imported."""
    done = []
    try:
        from sglang.multimodal_gen.runtime.models.vaes import wan_vae_cuda_opt as opt
        if _wrap_norm_forward(opt.FusedWanRMSNormSiLU, silu=True, label="sglang fused off-path"):
            done.append("sglang FusedWanRMSNormSiLU off-path")
    except Exception:
        pass
    try:
        from sglang.multimodal_gen.runtime.models.vaes import wanvae
        if _wrap_norm_forward(wanvae.WanRMS_norm, silu=False, label="sglang bare norm"):
            done.append("sglang WanRMS_norm")
    except Exception:
        pass
    try:
        from diffusers.models.autoencoders import autoencoder_kl_wan as dwan
        if _wrap_norm_forward(dwan.WanRMS_norm, silu=False, label="diffusers bare norm"):
            done.append("diffusers WanRMS_norm")
    except Exception:
        pass
    if done:
        return "wrapped " + ", ".join(done)
    return "already wrapped" if _vae_seams_done else "skipped (no Wan VAE norm class importable)"


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
    # The DiT seam, through SGLang's own module, at a shape their CUDA kernel refuses
    try:
        from sglang.multimodal_gen.runtime.layers.layernorm import LayerNormScaleShift
        norm = LayerNormScaleShift(384, eps=1e-6, elementwise_affine=False, dtype=dtype).to(device)
        xd = torch.randn(2, 16, 384, device=device, dtype=dtype)
        sh = torch.randn(2, 1, 384, device=device); sc = torch.randn(2, 1, 384, device=device)
        before = calls["modulated_layernorm"]
        got = norm(xd, sh, sc)
        ref = dit_fusions.modulated_layernorm_reference(xd, sh, sc)
        err = (got.float() - ref.float()).abs().max().item()
        report["wan_dit"] = {"ours_ran": calls["modulated_layernorm"] > before, "max_abs_err": err, "ok": err < 0.1}
    except Exception as e:
        report["wan_dit"] = {"error": f"{type(e).__name__}: {str(e)[:160]}"}
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
    # The VAE seam through SGLang's own classes: the fused class with its gate
    # off (the default request) and the bare norm of the attention block.
    try:
        from sglang.multimodal_gen.runtime.models.vaes.wanvae import WanRMS_norm
        from sglang.multimodal_gen.runtime.models.vaes.wan_vae_cuda_opt import FusedWanRMSNormSiLU
        from sglang.multimodal_gen.runtime.models.vaes.fast_path_gate import VaeFastPathGate
        norm = WanRMS_norm(96, images=False).to(device)
        fused = FusedWanRMSNormSiLU(norm, VaeFastPathGate()).to(device)
        xv = torch.randn(1, 96, 3, 16, 24, device=device, dtype=dtype)
        before = calls["channel_rmsnorm_silu"]
        got_bare = norm(xv)
        got_fused = fused(xv)
        ref_bare = torch.nn.functional.normalize(xv.float(), dim=1) * norm.scale * norm.gamma.float()
        ref_fused = torch.nn.functional.silu(ref_bare)
        err_bare = (got_bare.float() - ref_bare).abs().max().item()
        err_fused = (got_fused.float() - ref_fused).abs().max().item()
        report["wan_vae"] = {"ours_ran": calls["channel_rmsnorm_silu"] - before == 2,
                             "bare_err": err_bare, "fused_off_path_err": err_fused,
                             "ok": err_bare < 0.1 and err_fused < 0.1}
    except Exception as e:
        report["wan_vae"] = {"error": f"{type(e).__name__}: {str(e)[:160]}"}
    report["installed"] = dict(installed)
    report["calls"] = dict(calls)
    return report


if __name__ == "__main__":
    import json
    print(json.dumps(install(), indent=1))
    print(json.dumps(check(), indent=1))
