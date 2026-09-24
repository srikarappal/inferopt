"""Two passes a DiT block makes over its hidden states, each done once.

Every block of a Wan or Z-Image transformer normalises its input, then
modulates it by the timestep embedding, norm(x) * (1 + scale) + shift, and
after attention adds the result back through a gate, x + gate * y. In eager
PyTorch that is a LayerNorm kernel, two elementwise kernels, and another two,
each reading and writing the full [batch, tokens, channels] tensor. At 1024
squared on a 6B DiT those tensors are the traffic. Here each is one kernel:
one read, one write.

Shapes: x [R, C] (rows are batch x tokens), shift and scale [R, C] or
broadcast per batch through `row_to_mod`; outputs in x's dtype, statistics
in fp32.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _modulated_layernorm_kernel(x_ptr, shift_ptr, scale_ptr, rowmap_ptr, y_ptr,
                                C, eps,
                                stride_x, stride_mod, stride_y,
                                BLOCK_C: tl.constexpr):
    r = tl.program_id(0)
    mod_row = tl.load(rowmap_ptr + r).to(tl.int64)
    offs = tl.arange(0, BLOCK_C)
    mask = offs < C
    x = tl.load(x_ptr + r * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / C
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / C
    xn = xc * tl.rsqrt(var + eps)
    shift = tl.load(shift_ptr + mod_row * stride_mod + offs, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr + mod_row * stride_mod + offs, mask=mask, other=0.0).to(tl.float32)
    y = xn * (1.0 + scale) + shift
    tl.store(y_ptr + r * stride_y + offs, y.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _gated_residual_kernel(x_ptr, gate_ptr, y_ptr, rowmap_ptr, out_ptr,
                           C, stride_x, stride_gate, stride_y, stride_out,
                           BLOCK_C: tl.constexpr):
    r = tl.program_id(0)
    mod_row = tl.load(rowmap_ptr + r).to(tl.int64)
    offs = tl.arange(0, BLOCK_C)
    mask = offs < C
    x = tl.load(x_ptr + r * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(gate_ptr + mod_row * stride_gate + offs, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + r * stride_y + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + r * stride_out + offs, (x + g * y).to(out_ptr.dtype.element_ty), mask=mask)


def _rows_and_map(x: torch.Tensor, mod: torch.Tensor):
    """x as [R, C] and, for each row, which modulation row applies.

    Modulation comes per batch ([B, C] or [B, 1, C]) or per token ([B, S, C]);
    the map says which, so one kernel serves both without a broadcast copy.
    """
    if x.dim() == 3:
        B, S, C = x.shape
        x2 = x.reshape(B * S, C)
        if mod.dim() == 3 and mod.shape[1] == S:
            rowmap = torch.arange(B * S, device=x.device, dtype=torch.int32)
            mod2 = mod.reshape(B * S, C)
        else:
            rowmap = torch.arange(B, device=x.device, dtype=torch.int32).repeat_interleave(S)
            mod2 = mod.reshape(B, C)
        return x2, mod2, rowmap
    R, C = x.shape
    return x, mod.reshape(-1, C), torch.arange(R, device=x.device, dtype=torch.int32)


def modulated_layernorm(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor,
                        eps: float = 1e-6) -> torch.Tensor:
    x2, shift2, rowmap = _rows_and_map(x, shift)
    _, scale2, _ = _rows_and_map(x, scale)
    R, C = x2.shape
    y = torch.empty_like(x2)
    _modulated_layernorm_kernel[(R,)](
        x2, shift2.contiguous(), scale2.contiguous(), rowmap, y, C, eps,
        x2.stride(0), shift2.stride(0) if shift2.dim() == 2 else C, y.stride(0),
        BLOCK_C=triton.next_power_of_2(C))
    return y.reshape(x.shape)


def gated_residual(x: torch.Tensor, gate: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x2, gate2, rowmap = _rows_and_map(x, gate)
    y2 = y.reshape(x2.shape)
    R, C = x2.shape
    out = torch.empty_like(x2)
    _gated_residual_kernel[(R,)](
        x2, gate2.contiguous(), y2, rowmap, out, C,
        x2.stride(0), gate2.stride(0), y2.stride(0), out.stride(0),
        BLOCK_C=triton.next_power_of_2(C))
    return out.reshape(x.shape)


def modulated_layernorm_reference(x, shift, scale, eps=1e-6):
    xn = torch.nn.functional.layer_norm(x.float(), (x.shape[-1],), eps=eps)
    return (xn * (1.0 + scale.float()) + shift.float()).to(x.dtype)


def gated_residual_reference(x, gate, y):
    return (x.float() + gate.float() * y.float()).to(x.dtype)
