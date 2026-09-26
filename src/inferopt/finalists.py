"""Stage 2.1: the finalists, swept densely.

The walk (stage 2) ranks configurations at one operating point and a coarse
bracket around it, which is enough to CHOOSE between them: most goodput curves
sit uniformly above or below each other rather than crossing. It is not enough
to draw one. A curve of two to four points is a dot with a hint, and the shape
that sets capacity, and therefore the replica count and the price, is the whole
trade-off: few users fast against many users slower, on the configuration that
ships.

So the configurations anyone might deploy are launched once more and measured
at every rung of a ladder, past the first level that misses the target, with
the profiler window at the peak exactly as a walk trial gets it. What comes out
per finalist is what a DAG node gets: the curve, the peak, the profile
(families, top kernels, busy fraction) and a journal row, so a restart resumes
instead of re-sweeping and stage 3 (GEPA) reads the same diagnostics for a
finalist as for any node.

The ladder is capped at the configuration's own max_num_seqs. The sweep
measures the configuration that ships; raising a cap to see more curve would
measure a different one.
"""

from __future__ import annotations

import json
from pathlib import Path

from inferopt.traverse import Trial

LADDER_LLM = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256)
LADDER_DLLM = (1, 2, 3, 4, 6, 8, 12, 16)
LADDER_PIPELINE = (1, 2, 3, 4, 6, 8)


def ladder_for(fp, config: dict) -> tuple[int, ...]:
    """The concurrency levels a finalist is swept at.

    Sixteen for an autoregressive LLM, eight for a diffusion LM (a canvas per
    request makes every level slow), six for an image or video pipeline, each
    log spaced with a half step so the knee is not skipped, and cut at the
    configuration's max_num_seqs where it has one.
    """
    if getattr(fp, "diffusion", None) is not None:
        base = LADDER_PIPELINE
    elif getattr(fp.model, "decoding", "autoregressive") == "diffusion":
        base = LADDER_DLLM
    else:
        base = LADDER_LLM
    cap = (config or {}).get("max_num_seqs")
    if not cap:
        return base
    return tuple(level for level in base if level <= int(cap)) or (int(cap),)


def pick(trials: list[Trial], frontier: list[Trial], incumbent_node: str | None,
         n: int) -> list[Trial]:
    """The top n of the frontier, plus the incumbent when it is not among them.

    The incumbent is not necessarily on the frontier and the run ships the
    incumbent. On Qwen3-1.7B the walk kept prefix_caching, graph_capture
    dominated it and all three finalists swept were configurations the run did
    NOT deploy, so the shipped one had no capacity number at all.
    """
    chosen = list(frontier[:n])
    if incumbent_node and not any(t.node_id == incumbent_node for t in chosen):
        candidates = [t for t in trials if t.node_id == incumbent_node and t.goodput]
        if candidates:
            chosen.append(max(candidates, key=lambda t: t.goodput))
    return chosen


def sweep_finalists(evaluator, fp, trials: list[Trial], frontier: list[Trial],
                    incumbent_node: str | None, *, n: int = 3,
                    journal: Path | None = None, log=print) -> dict:
    """Stage 2.1. Returns {node_id: {"levels", "peak", "profile"}}.

    Each finalist's dense curve replaces the coarse one on its trial, and its
    operating point moves to the dense peak, so the result and the plot carry
    the measured curve of what ships. The walk's goodput is left as measured:
    that number decided the walk, and rewriting it would rewrite the decision.
    """
    finalists = pick(trials, frontier, incumbent_node, n) if n > 0 else []
    if not finalists:
        log("  stage 2.1  nothing met the target, so there is no finalist to sweep")
        return {}
    log(f"  stage 2.1  sweeping {len(finalists)} finalist(s) densely, past the first miss")
    out: dict = {}
    for trial in finalists:
        levels = ladder_for(fp, trial.config)
        log(f"  stage 2.1  {trial.node_id}: {len(levels)} levels, L={levels[0]}..{levels[-1]}")
        try:
            swept = evaluator.measure(trial.config, probes=["goodput"], benchmarks=[],
                                      node_id=f"finalist:{trial.node_id}", levels=levels)
        except Exception as error:
            log(f"  stage 2.1  {trial.node_id}: sweep failed ({type(error).__name__}: "
                f"{str(error)[:100]}); the walk's measurement stands")
            continue
        if journal is not None and not (swept.diagnostics or {}).get("replayed"):
            swept.provenance = dict(trial.provenance)
            with open(journal, "a") as fh:
                fh.write(json.dumps(swept.__dict__, default=str) + "\n")
        curve = swept.curve or []
        if not curve:
            log(f"  stage 2.1  {trial.node_id}: served nothing at any level")
            continue
        peak = max(curve, key=lambda m: m["goodput"])
        met = sum(1 for m in curve if m.get("goodput", 0) > 0)
        summary = {"levels": list(levels), "peak": peak,
                   "profile": (swept.diagnostics or {}).get("profile")}
        trial.curve = curve
        trial.concurrency = peak["concurrency"]
        trial.diagnostics = {**(trial.diagnostics or {}), "finalist": summary}
        out[trial.node_id] = summary
        log(f"  stage 2.1  {trial.node_id}: peak {peak['goodput']:.1f} at L={peak['concurrency']}, "
            f"{met} of {len(curve)} levels served within the target")
    return out
