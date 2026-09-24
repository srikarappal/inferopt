"""MoE expert projection at small batch, as a GEMV per (token, expert).

fused_moe tiles a [tokens x hidden] block against an expert's weights, which
is right when many tokens share an expert. At decode with top-8 of 256 experts
and a handful of tokens, almost every expert sees one token, the tile is
mostly padding, and the kernel is bound by wasted MMA rather than by the
weight bytes it has to read anyway. Below `SMALL_M` the shape is a GEMV: read
each selected expert's rows once, multiply by the token's vector, done.

Two kernels, the way the layer is shaped: up projection with SiLU and the
multiply fused into its epilogue, then down projection accumulating the
routing weight into the token's output with atomics, since eight experts
land on one row.

Layout follows vLLM and SGLang: w1 [E, 2N, K] with the gate rows first,
w2 [E, K, N], bf16 or fp16 weights, fp32 accumulation.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

SMALL_M = 16


@triton.jit
def _silu(x):
    return x * tl.sigmoid(x)


@triton.jit
def _moe_up_kernel(x_ptr, w1_ptr, ids_ptr, h_ptr,
                   T, N, K,
                   stride_xm, stride_we, stride_wn,
                   BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_mt = tl.program_id(0)
    pid_n = tl.program_id(1)
    m = pid_mt // T
    e = tl.load(ids_ptr + pid_mt).to(tl.int64)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    acc_g = tl.zeros([BLOCK_N], dtype=tl.float32)
    acc_u = tl.zeros([BLOCK_N], dtype=tl.float32)
    w_base = w1_ptr + e * stride_we
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        xv = tl.load(x_ptr + m * stride_xm + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        wmask = mask_n[:, None] & mask_k[None, :]
        wg = tl.load(w_base + offs_n[:, None] * stride_wn + offs_k[None, :], mask=wmask, other=0.0).to(tl.float32)
        wu = tl.load(w_base + (offs_n + N)[:, None] * stride_wn + offs_k[None, :], mask=wmask, other=0.0).to(tl.float32)
        acc_g += tl.sum(wg * xv[None, :], axis=1)
        acc_u += tl.sum(wu * xv[None, :], axis=1)
    h = _silu(acc_g) * acc_u
    tl.store(h_ptr + pid_mt * N + offs_n, h, mask=mask_n)


@triton.jit
def _moe_down_kernel(h_ptr, w2_ptr, ids_ptr, wts_ptr, out_ptr,
                     T, N, K,
                     stride_we, stride_wk, stride_om,
                     BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_mt = tl.program_id(0)
    pid_k = tl.program_id(1)
    m = pid_mt // T
    e = tl.load(ids_ptr + pid_mt).to(tl.int64)
    wt = tl.load(wts_ptr + pid_mt).to(tl.float32)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = offs_k < K
    acc = tl.zeros([BLOCK_K], dtype=tl.float32)
    w_base = w2_ptr + e * stride_we
    for n0 in range(0, N, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        hv = tl.load(h_ptr + pid_mt * N + offs_n, mask=mask_n, other=0.0)
        w = tl.load(w_base + offs_k[:, None] * stride_wk + offs_n[None, :],
                    mask=mask_k[:, None] & mask_n[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(w * hv[None, :], axis=1)
    tl.atomic_add(out_ptr + m * stride_om + offs_k, acc * wt, mask=mask_k)


def moe_small_m(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor,
                topk_ids: torch.Tensor, topk_weights: torch.Tensor) -> torch.Tensor:
    """out[m] = sum_t topk_weights[m, t] * (w2[e] @ (silu(w1g[e] @ x[m]) * (w1u[e] @ x[m])))."""
    assert x.dim() == 2 and w1.dim() == 3 and w2.dim() == 3
    M, K = x.shape
    E, N2, K1 = w1.shape
    N = N2 // 2
    assert K1 == K and w2.shape == (E, K, N)
    T = topk_ids.shape[1]
    x = x.contiguous()
    ids = topk_ids.reshape(-1).contiguous()
    wts = topk_weights.reshape(-1).contiguous().to(torch.float32)
    h = torch.empty((M * T, N), device=x.device, dtype=torch.float32)
    out = torch.zeros((M, K), device=x.device, dtype=torch.float32)
    BLOCK_N, BLOCK_K = 64, 128
    _moe_up_kernel[(M * T, triton.cdiv(N, BLOCK_N))](
        x, w1, ids, h, T, N, K, x.stride(0), w1.stride(0), w1.stride(1),
        BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
    _moe_down_kernel[(M * T, triton.cdiv(K, 64))](
        h, w2, ids, wts, out, T, N, K, w2.stride(0), w2.stride(1), out.stride(0),
        BLOCK_K=64, BLOCK_N=128)
    return out.to(x.dtype)


def moe_reference(x, w1, w2, topk_ids, topk_weights) -> torch.Tensor:
    """The same thing in plain torch, fp32, for the test and the bench."""
    M, K = x.shape
    N = w1.shape[1] // 2
    out = torch.zeros((M, K), device=x.device, dtype=torch.float32)
    xf = x.float()
    for m in range(M):
        for t in range(topk_ids.shape[1]):
            e = int(topk_ids[m, t])
            g = w1[e, :N].float() @ xf[m]
            u = w1[e, N:].float() @ xf[m]
            hidden = torch.nn.functional.silu(g) * u
            out[m] += float(topk_weights[m, t]) * (w2[e].float() @ hidden)
    return out.to(x.dtype)
