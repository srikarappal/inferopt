"""Everything needed to reproduce a run, or to distrust it later.

    meta = provenance(ap, args, fingerprint, extra={"configs": ...})
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str))

Written BEFORE the first measurement, so a crashed run still records what was
attempted.

A results file that does not say which model, which flags, which code and which
machine produced it is a number without a claim attached. This project has
already lost measurements twice for want of that: run four's baseline existed
only in terminal scrollback, and runs seven and nine reported a quality drift
whose cause could not be checked afterwards because nothing recorded the state
that produced it.

Arguments are recorded with their DEFAULT alongside the value and a flag for
whether they were explicit. Reading `"repeats": 5` back in a month does not say
whether someone chose 5 or whether 5 was the default then -- and defaults change.

Shared by run.py and eval_repro.py deliberately. Two implementations of the same
record diverge, and this project has paid for duplication before: the prompt text
was built in two places, they disagreed, and it crashed a run nine launches in.
"""

from __future__ import annotations

import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent


def _version(module: str) -> str | None:
    try:
        return __import__(module).__version__
    except Exception:
        return None


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True,
                              timeout=10, cwd=_HERE).stdout.strip()
    except Exception:
        return None


def environment() -> dict[str, Any]:
    """Interpreter, platform, the versions that decide behaviour, and the code.

    `dirty` matters as much as the commit: a result produced from an uncommitted
    tree cannot be reproduced from the commit alone, and that should be visible
    on the record rather than discovered when someone tries.
    """
    return {
        "python": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "vllm": _version("vllm"),
        "torch": _version("torch"),
        "transformers": _version("transformers"),
        "inferopt_commit": _git("rev-parse", "HEAD"),
        "inferopt_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "inferopt_dirty": bool(_git("status", "--porcelain")),
    }


def provenance(ap, args, fp=None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """The full record. `extra` carries whatever the caller resolved at runtime.

    Anything computed rather than passed belongs in `extra` -- the port actually
    bound when the requested one was taken, the config after a predictor
    modified it, the problem count after filtering. Those are the values the run
    used, and they are not recoverable from the arguments.
    """
    rec: dict[str, Any] = {
        "when": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "command": " ".join(sys.argv),
        "cwd": str(Path.cwd()),
        "args": {k: {"value": v, "default": ap.get_default(k),
                     "explicit": v != ap.get_default(k)}
                 for k, v in sorted(vars(args).items())},
        "environment": environment(),
    }
    if fp is not None:
        rec["fingerprint"] = {
            "model": fp.model.model_dump(),
            "hardware": fp.hw.model_dump(),
            "workload": fp.workload.model_dump(),
            "lora": fp.lora.model_dump(),
        }
    if extra:
        rec["resolved"] = extra
    return rec


def banner(meta: dict[str, Any], path: Path) -> str:
    """One line for the console, so a dirty tree is visible while it still matters."""
    env = meta["environment"]
    commit = (env.get("inferopt_commit") or "?")[:8]
    dirty = "  DIRTY TREE" if env.get("inferopt_dirty") else ""
    return (f"  provenance -> {path}  (commit {commit}{dirty}, "
            f"vllm {env.get('vllm')}, torch {env.get('torch')})")


def trial_stamp(fp, trace: str | Path | None = None, slo=None) -> dict[str, Any]:
    """The compact identity a single MEASUREMENT must carry to be poolable.

    run_meta.json already records everything about a run. That is the wrong
    granularity for one job: comparing optimizers means pooling trials from many
    runs into one table, and a row that does not say what produced it cannot be
    pooled -- it can only be trusted or discarded wholesale. 80 trials were
    accumulated across nine runs before anyone noticed that not one of them
    recorded its model, its trace or its vLLM version.

    SLO IS PART OF THE IDENTITY, not context. Goodput counts only the requests
    that met the SLO, so the same config measured against a 500 ms TTFT target
    and a 200 ms one produces two different numbers that are not comparable and
    do not average. Pooling those silently is the failure this is here to stop.

    `key` is a hash of everything above, so a reader can group by one value
    instead of comparing seven fields and getting it subtly wrong.
    """
    import hashlib
    import json as _json

    rec: dict[str, Any] = {}
    if fp is not None:
        rec["model"] = fp.model.id
        rec["gpu"] = fp.hw.gpu_name
        rec["gpu_count"] = fp.hw.gpu_count
    env = environment()
    rec["vllm"] = env["vllm"]
    rec["commit"] = (env["inferopt_commit"] or "")[:12] or None
    rec["dirty"] = env["inferopt_dirty"]

    if trace:
        p = Path(trace)
        try:
            raw = p.read_bytes()
            rec["trace"] = p.name
            rec["trace_sha"] = hashlib.sha256(raw).hexdigest()[:12]
            rec["trace_rows"] = raw.count(b"\n")
        except Exception:
            # A trace we cannot read is recorded as unknown rather than omitted:
            # an absent field reads as "not applicable", which is a lie here.
            rec["trace"] = str(trace)
            rec["trace_sha"] = None
    if slo is not None:
        rec["slo"] = {"ttft_p99_ms": getattr(slo, "ttft_p99_ms", None),
                      "itl_p99_ms": getattr(slo, "itl_p99_ms", None)}
    rec["key"] = hashlib.sha256(
        _json.dumps(rec, sort_keys=True, default=str).encode()).hexdigest()[:12]
    return rec
