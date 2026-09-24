"""Each kernel against its torch reference. Needs a CUDA card and Triton;
skipped without them, and run on the DGX before any claim is made."""

import math

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA card")


@pytest.fixture(scope="module")
def dev():
    torch.manual_seed(0)
    return "cuda"


def test_moe_small_m_matches_reference(dev):
    from vi_kernels import moe_gemv
    K, N, E, T = 256, 96, 12, 3
    for M in (1, 5, 16):
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        w1 = torch.randn(E, 2 * N, K, device=dev, dtype=torch.bfloat16) * 0.05
        w2 = torch.randn(E, K, N, device=dev, dtype=torch.bfloat16) * 0.05
        ids = torch.stack([torch.randperm(E, device=dev)[:T] for _ in range(M)]).to(torch.int32)
        wts = torch.softmax(torch.randn(M, T, device=dev), dim=-1)
        out = moe_gemv.moe_small_m(x, w1, w2, ids, wts)
        ref = moe_gemv.moe_reference(x, w1, w2, ids, wts)
        assert torch.allclose(out.float(), ref.float(), atol=0.05, rtol=0.05), (out - ref).abs().max()


def test_dllm_attention_matches_reference_and_only_active_rows(dev):
    from vi_kernels import dllm_attention
    H, HKV, D, P, B = 8, 2, 64, 100, 16
    q = torch.randn(B, H, D, device=dev, dtype=torch.bfloat16)
    k = torch.randn(P + B, HKV, D, device=dev, dtype=torch.bfloat16)
    v = torch.randn(P + B, HKV, D, device=dev, dtype=torch.bfloat16)
    active = torch.tensor([0, 3, 7, 15], device=dev, dtype=torch.int32)
    out = dllm_attention.dllm_block_attention(q, k, v, active)
    ref = dllm_attention.dllm_reference(q, k, v, active)
    assert out.shape == (4, H, D)
    assert torch.allclose(out.float(), ref.float(), atol=0.03, rtol=0.03), (out - ref).abs().max()
    # The prefix is attended: change a prefix value and the output moves.
    v2 = v.clone(); v2[:P] += 1.0
    assert not torch.allclose(dllm_attention.dllm_block_attention(q, k, v2, active).float(), out.float(), atol=0.1)
    # No rows, no work.
    assert dllm_attention.dllm_block_attention(q, k, v, active[:0]).shape == (0, H, D)


def test_modulated_layernorm_and_gate(dev):
    from vi_kernels import dit_fusions
    for S, C in ((37, 384), (64, 1536)):
        x = torch.randn(2, S, C, device=dev, dtype=torch.bfloat16)
        shift = torch.randn(2, 1, C, device=dev)
        scale = torch.randn(2, 1, C, device=dev)
        gate = torch.randn(2, 1, C, device=dev)
        y = torch.randn_like(x)
        got = dit_fusions.modulated_layernorm(x, shift, scale)
        ref = dit_fusions.modulated_layernorm_reference(x, shift, scale)
        assert torch.allclose(got.float(), ref.float(), atol=0.1, rtol=0.05), (got - ref).abs().max()
        got2 = dit_fusions.gated_residual(x, gate, y)
        ref2 = dit_fusions.gated_residual_reference(x, gate, y)
        assert torch.allclose(got2.float(), ref2.float(), atol=0.05, rtol=0.05)
    # per token modulation too (Wan 2.2 TI2V hands [B, S, 6, C])
    x = torch.randn(1, 10, 128, device=dev, dtype=torch.bfloat16)
    shift = torch.randn(1, 10, 128, device=dev); scale = torch.randn(1, 10, 128, device=dev)
    assert torch.allclose(dit_fusions.modulated_layernorm(x, shift, scale).float(),
                          dit_fusions.modulated_layernorm_reference(x, shift, scale).float(), atol=0.1, rtol=0.05)


def test_channel_rmsnorm_silu(dev):
    from vi_kernels import vae_norm
    x = torch.randn(2, 96, 3, 12, 20, device=dev, dtype=torch.bfloat16)
    gamma = torch.randn(96, device=dev); bias = torch.randn(96, device=dev)
    for silu in (True, False):
        got = vae_norm.channel_rmsnorm_silu(x, gamma, bias, silu=silu)
        ref = vae_norm.channel_rmsnorm_silu_reference(x, gamma, bias, silu=silu)
        assert torch.allclose(got.float(), ref.float(), atol=0.1, rtol=0.05), (got - ref).abs().max()
    got = vae_norm.channel_rmsnorm_silu(x, gamma, None)
    ref = vae_norm.channel_rmsnorm_silu_reference(x, gamma, None)
    assert torch.allclose(got.float(), ref.float(), atol=0.1, rtol=0.05)
