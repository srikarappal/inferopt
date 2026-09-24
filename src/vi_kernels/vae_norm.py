"""A video VAE's channel RMSNorm and SiLU, one pass.

Wan's decoder normalises each spatial position's channel vector,
F.normalize(x, dim=1) * sqrt(C) * gamma + bias, then applies SiLU, on
tensors laid out [N, C, T, H, W]. Channels are the strided dimension, so the
norm is a reduction across strides while SiLU is a second full pass. Here
one program takes a tile of spatial positions and all their channels: loads
are coalesced along the spatial axis, the reduction runs along channels in
registers, and the activation is applied before the single write.

Decoding is where a video pipeline goes memory bound, which is why the pass
count is the cost.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _channel_rmsnorm_silu_kernel(x_ptr, gamma_ptr, bias_ptr, y_ptr,
                                 C, S, scale, eps,
                                 stride_n, stride_c,
                                 HAS_BIAS: tl.constexpr, SILU: tl.constexpr,
                                 BLOCK_C: tl.constexpr, BLOCK_S: tl.constexpr):
    n = tl.program_id(0)
    pid_s = tl.program_id(1)
    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = offs_s < S
    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C
    ptrs = x_ptr + n * stride_n + offs_c[:, None] * stride_c + offs_s[None, :]
    mask = mask_c[:, None] & mask_s[None, :]
    x = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
    norm = tl.sqrt(tl.sum(x * x, axis=0))                 # [BLOCK_S], L2 over channels
    inv = 1.0 / tl.maximum(norm, eps)
    gamma = tl.load(gamma_ptr + offs_c, mask=mask_c, other=1.0).to(tl.float32)
    y = x * inv[None, :] * scale * gamma[:, None]
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_c, mask=mask_c, other=0.0).to(tl.float32)
        y = y + bias[:, None]
    if SILU:
        y = y * tl.sigmoid(y)
    tl.store(y_ptr + n * stride_n + offs_c[:, None] * stride_c + offs_s[None, :],
             y.to(y_ptr.dtype.element_ty), mask=mask)


def channel_rmsnorm_silu(x: torch.Tensor, gamma: torch.Tensor, bias: torch.Tensor | None = None,
                         eps: float = 1e-12, silu: bool = True) -> torch.Tensor:
    """x [N, C, *spatial], gamma [C] (or [C, 1, 1, 1]), bias likewise or None."""
    N, C = x.shape[0], x.shape[1]
    x = x.contiguous()
    S = x[0, 0].numel()
    y = torch.empty_like(x)
    g = gamma.reshape(-1).contiguous()
    b = bias.reshape(-1).contiguous() if bias is not None else g
    BLOCK_S = 64
    _channel_rmsnorm_silu_kernel[(N, triton.cdiv(S, BLOCK_S))](
        x, g, b, y, C, S, math.sqrt(C), eps, x.stride(0), x.stride(1),
        HAS_BIAS=bias is not None, SILU=silu,
        BLOCK_C=triton.next_power_of_2(C), BLOCK_S=BLOCK_S)
    return y


def channel_rmsnorm_silu_reference(x, gamma, bias=None, eps=1e-12, silu=True):
    C = x.shape[1]
    shape = (1, C) + (1,) * (x.dim() - 2)
    y = torch.nn.functional.normalize(x.float(), dim=1, eps=eps) * math.sqrt(C) * gamma.float().reshape(shape)
    if bias is not None:
        y = y + bias.float().reshape(shape)
    if silu:
        y = torch.nn.functional.silu(y)
    return y.to(x.dtype)
