"""Kernels for the operations the walk found no engine covering.

Written as Triton torch extensions, so they compile on any card Triton
supports, and register into an engine through a thin adapter rather than a
fork. Each module carries its own torch reference and is tested against it;
each is measured by `bench.py` before it is believed to be faster than what
it replaces. Nothing here is faster until the table says so.

  moe_gemv        MoE expert projection at small batch: a GEMV per (token,
                  expert) that streams each selected expert's weights once,
                  for the decode regime the fused_moe tiles were not cut for
  dllm_attention  attention for the still masked rows of a diffusion LM's
                  block only; the committed rows need no new logits
  dit_fusions     modulated LayerNorm and the gated residual of a DiT block
                  as one pass each
  vae_norm        channel RMSNorm and SiLU of a video VAE as one pass
"""

__version__ = "0.1.0"


def available() -> bool:
    try:
        import torch
        import triton  # noqa: F401
        return torch.cuda.is_available()
    except Exception:
        return False
