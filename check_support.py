"""Print the artifact kinds THIS GPU cannot run, space separated. Empty if all.

    python check_support.py        ->  "nvfp4"   on H100
                                   ->  ""        on GB10/B200

Used by eval_ladder.sh to skip rows that would otherwise cost a model load and a
launch timeout to discover what the fingerprint already knows. NVFP4 needs
Blackwell FP4 hardware; on sm90 the engine simply refuses the checkpoint.

Kept as its own file rather than an inline heredoc inside the shell script: a
heredoc nested inside another heredoc terminates the outer one, which is exactly
how the first attempt at this broke.
"""

from __future__ import annotations

import sys


def unsupported() -> list[str]:
    """Reads the device directly rather than building a fingerprint.

    detect_hardware() takes an InferOptRequest, which validates that the model
    and trace exist -- neither of which this question depends on. Compute
    capability is the whole input.
    """
    import torch
    major, minor = torch.cuda.get_device_capability(0)
    bad = []
    if major < 10:            # FP4 tensor cores are Blackwell (sm100/sm120) and later
        bad.append("nvfp4")
    return bad


if __name__ == "__main__":
    try:
        print(" ".join(unsupported()))
    except Exception as e:
        # Never block a ladder because capability detection failed -- let the
        # launch itself be the judge.
        print("", file=sys.stdout)
        print(f"check_support: {type(e).__name__}: {e}", file=sys.stderr)
