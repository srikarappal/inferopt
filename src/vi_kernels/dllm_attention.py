"""Attention for the rows of a diffusion LM's block that still need it.

A masked diffusion LM denoises a block of B positions over several forward
passes. Every pass, the block attends to the whole prefix and to itself, both
directions, no mask; after the pass, the positions whose confidence cleared
the threshold are committed and never change again. SGLang runs the full
block through attention on every pass. The committed rows' outputs are
recomputed and thrown away.

This kernel takes the index list of the rows still masked and computes
attention for those alone: a flash style forward, online softmax over the
prefix and the block, GQA by head ratio, with the query rows gathered by
index. Savings scale with the committed fraction, which at a 0.95 threshold
is most of the block for most of the passes.

Shapes: q [B, H, D] the block's queries; k, v [L, Hkv, D] the prefix followed
by the block (L = P + B); active [A] int32 row indices into q; out [A, H, D].
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _dllm_attn_kernel(q_ptr, k_ptr, v_ptr, idx_ptr, o_ptr,
                      L, H, HKV, scale,
                      stride_qb, stride_qh, stride_kl, stride_kh, stride_vl, stride_vh,
                      stride_oa, stride_oh,
                      D: tl.constexpr, BLOCK_L: tl.constexpr):
    a = tl.program_id(0)                  # which active row
    h = tl.program_id(1)                  # which head
    row = tl.load(idx_ptr + a).to(tl.int64)
    hkv = h // (H // HKV)
    offs_d = tl.arange(0, D)
    q = tl.load(q_ptr + row * stride_qb + h * stride_qh + offs_d).to(tl.float32) * scale
    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([D], dtype=tl.float32)
    for l0 in range(0, L, BLOCK_L):
        offs_l = l0 + tl.arange(0, BLOCK_L)
        mask_l = offs_l < L
        k = tl.load(k_ptr + offs_l[:, None] * stride_kl + hkv * stride_kh + offs_d[None, :],
                    mask=mask_l[:, None], other=0.0).to(tl.float32)
        s = tl.sum(k * q[None, :], axis=1)
        s = tl.where(mask_l, s, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        v = tl.load(v_ptr + offs_l[:, None] * stride_vl + hkv * stride_vh + offs_d[None, :],
                    mask=mask_l[:, None], other=0.0).to(tl.float32)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m_i = m_new
    tl.store(o_ptr + a * stride_oa + h * stride_oh + offs_d, (acc / l_i).to(o_ptr.dtype.element_ty))


def dllm_block_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                         active: torch.Tensor) -> torch.Tensor:
    """Attention outputs for q[active] over all of k and v."""
    B, H, D = q.shape
    L, HKV, Dk = k.shape
    assert Dk == D and v.shape == (L, HKV, D) and H % HKV == 0
    assert D in (32, 64, 128, 256), "head dim must be a power of two up to 256"
    active = active.to(torch.int32).contiguous()
    A = active.numel()
    out = torch.empty((A, H, D), device=q.device, dtype=q.dtype)
    if A == 0:
        return out
    _dllm_attn_kernel[(A, H)](
        q, k, v, active, out, L, H, HKV, 1.0 / math.sqrt(D),
        q.stride(0), q.stride(1), k.stride(0), k.stride(1), v.stride(0), v.stride(1),
        out.stride(0), out.stride(1),
        D=D, BLOCK_L=64)
    return out


def dllm_reference(q, k, v, active) -> torch.Tensor:
    """Full attention in torch for the active rows, GQA expanded."""
    B, H, D = q.shape
    L, HKV, _ = k.shape
    rep = H // HKV
    kk = k.repeat_interleave(rep, dim=1).float()      # [L, H, D]
    vv = v.repeat_interleave(rep, dim=1).float()
    qa = q[active.long()].float()                    # [A, H, D]
    s = torch.einsum("ahd,lhd->ahl", qa, kk) / math.sqrt(D)
    p = torch.softmax(s, dim=-1)
    return torch.einsum("ahl,lhd->ahd", p, vv).to(q.dtype)
