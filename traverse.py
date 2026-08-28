"""Walk a technique DAG: apply, measure, keep or revert, follow the edge.

    result = traverse(dag, ctx, evaluator)

Greedy accumulation. Each node costs its own measurements, not a combinatorial
share, so N nodes cost N launches rather than 2^N. The two rules that make the
result usable:

  keep/revert    a variant must beat the incumbent by more than accept_band,
                 which is measured noise, not a guess

  revert != discard
                 a reverted config still enters the trial database and can
                 still land on the Pareto frontier. It may be exactly the
                 operating point the user wants -- "worse goodput, better
                 latency" is a trade, not a failure.

The Evaluator is injected, so the traversal logic is testable without a GPU.

HISTORY

  GEPA's pareto selector is not a Pareto frontier. It tracks candidates that are
  best on individual INSTANCES, which is a different thing from the classical
  multi-objective non-dominated set. So the trial database is ours and
  Result.frontier() computes the real thing.

  revert != discard. A reverted config still enters the frontier. In a dry run
  three of five frontier points were reverted configs -- including an INT4
  variant at 2.9x throughput and a quarter the memory, which failed the stated
  quality budget but is unambiguously an operating point someone would choose.
  A lineage-only frontier throws those away.

  Per-benchmark quality is the WORST benchmark, not the mean. A config that
  preserves its average by collapsing on one task is not an acceptable operating
  point.

  lossless_only was added to park the lossy branch while the conversion pipeline
  is still being proven, without deleting nodes from the DAG.
"""

from __future__ import annotations

import json

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from calibration import STORE
from fingerprint import Context, NodeMeasurement
from predicates import Predicate

# Frontier axes and their direction. Goodput already encodes the latency SLO
# (requests that miss it earn nothing), but ttft is kept as its own axis because
# two configs can hit the same goodput at very different tail latencies.
OBJECTIVES = {"goodput": "max", "quality": "max", "ttft_p99_ms": "min", "memory_gb": "min"}


@dataclass
class Trial:
    node_id: str
    config: dict[str, Any]
    goodput: float
    ttft_p99_ms: float
    itl_p99_ms: float
    memory_gb: float
    quality: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    equivalence_divergence: float | None = None
    kept: bool = False
    slo_ok: bool = True

    @property
    def min_quality(self) -> float:
        """Worst benchmark, not the mean. A config that keeps its average by
        collapsing on one task is not an acceptable operating point."""
        return min(self.quality.values()) if self.quality else 1.0

    def axes(self) -> dict[str, float]:
        return {"goodput": self.goodput, "quality": self.min_quality,
                "ttft_p99_ms": self.ttft_p99_ms, "memory_gb": self.memory_gb}


class Evaluator(Protocol):
    """Launch a config and measure it. Implementations own the GPU."""

    def measure(self, config: dict[str, Any], *, probes: list[str],
                benchmarks: list[str], node_id: str) -> Trial: ...


@dataclass
class Result:
    trials: list[Trial]
    incumbent: dict[str, Any]
    visited: list[str]
    skipped: list[tuple[str, str]]
    launches: int
    minutes: float
    stopped_early: str | None = None

    def frontier(self) -> list[Trial]:
        """True non-dominated set over every measurement taken, reverted ones
        included. Deliberately not GEPA's per-instance frontier and not just the
        accepted lineage."""
        out = []
        for a in self.trials:
            if not a.slo_ok:
                continue
            dominated = False
            for b in self.trials:
                if b is a or not b.slo_ok:
                    continue
                ax, bx = a.axes(), b.axes()
                better_any = any(
                    (bx[k] > ax[k]) if d == "max" else (bx[k] < ax[k])
                    for k, d in OBJECTIVES.items())
                worse_none = all(
                    (bx[k] >= ax[k]) if d == "max" else (bx[k] <= ax[k])
                    for k, d in OBJECTIVES.items())
                if worse_none and better_any:
                    dominated = True
                    break
            if not dominated:
                out.append(a)
        return sorted(out, key=lambda t: -t.goodput)


def _value(v: Any, ctx: Context) -> Any:
    """Resolve a config value that may be an expression string."""
    if isinstance(v, str) and any(c in v for c in "+-*/()") and not v.startswith("/"):
        try:
            return Predicate(v).evaluate(ctx)
        except Exception:
            return v          # a plain string that merely contains punctuation
    return v


def _variants(node, base: dict, ctx: Context) -> list[dict]:
    """base config + the node's action, then one variant per sweep entry."""
    applied = dict(base)
    for k, v in ((node.get("action") or {}).get("set") or {}).items():
        applied[k] = _value(v, ctx)
    sweep = node.get("sweep")
    if not sweep:
        return [applied]
    out = []
    for entry in sweep:
        var = dict(applied)
        for k, v in entry.items():
            # "speculative_config.num_speculative_tokens" addresses a nested key
            if "." in k:
                head, tail = k.split(".", 1)
                var[head] = {**(var.get(head) or {}), tail: _value(v, ctx)}
            else:
                var[k] = _value(v, ctx)
        out.append(var)
    return out


def traverse(dag: dict, ctx: Context, evaluator: Evaluator,
             *, log=print, lossless_only: bool = False,
             journal: str | Path | None = None) -> Result:
    """Walk the DAG, measuring each applicable node against the incumbent.

    `journal` is a path to append every Trial to as it completes. It exists
    because a KeyError in the quality probe once discarded nine launches and two
    hours of measurement: result.json is only written after traverse RETURNS, so
    any exception inside it loses everything. A measurement that cost eight
    minutes of GPU time should survive a bug in the code that reads it.
    """
    nodes = {n["id"]: n for n in dag["nodes"]}
    root = next(n["id"] for n in dag["nodes"] if n.get("class") == "root")
    guard = dag["traversal"]["budget_guard"]
    scenario = "multi_lora" if ctx.fingerprint.lora.multi_lora_active else "default"
    max_launches = guard["max_launches"][scenario] if isinstance(guard["max_launches"], dict) else guard["max_launches"]
    max_minutes = guard["max_minutes"][scenario] if isinstance(guard["max_minutes"], dict) else guard["max_minutes"]

    band, band_src = STORE.accept_band(ctx.fingerprint)
    ctx.accept_band = band
    log(f"accept_band {band:.1%}  [{band_src}]")
    log(f"budget      {max_launches} launches / {max_minutes} min  [{scenario}]\n")

    jpath = Path(journal) if journal else None
    if jpath:
        jpath.parent.mkdir(parents=True, exist_ok=True)
        jpath.write_text("")

    def record(t: Trial) -> None:
        """Append one measurement to the journal immediately.

        Flushed per line: a crash, a kill, or an OOM must not cost the trials
        already paid for. Journal failures are reported but never raised -- a
        broken journal must not take down a run that is otherwise fine."""
        if not jpath:
            return
        try:
            with open(jpath, "a") as fh:
                fh.write(json.dumps(t.__dict__, default=str) + "\n")
                fh.flush()
        except Exception as e:
            log(f"  (journal write failed: {type(e).__name__}: {e})")

    trials: list[Trial] = []
    visited: list[str] = []
    skipped: list[tuple[str, str]] = []
    launches, t0, stopped = 0, time.time(), None
    incumbent_cfg = dict(ctx.incumbent)
    if ctx.incumbent_metrics and ctx.incumbent_metrics.goodput:
        incumbent_goodput = ctx.incumbent_metrics.goodput
        log(f"incumbent   {incumbent_goodput:.1f} goodput  [stage 1.3]")
    else:
        # No stage-1.3 measurement: measure the incumbent here rather than let
        # the first node be kept for free. Costs one launch and makes every
        # keep/revert decision in the run comparable.
        t = evaluator.measure(incumbent_cfg, probes=["goodput"], benchmarks=[], node_id="incumbent")
        record(t)
        launches += 1
        trials.append(t)
        incumbent_goodput = t.goodput
        log(f"incumbent   {incumbent_goodput:.1f} goodput  [measured here, "
            f"no stage 1.3 result was supplied]")
    cur, last_kept = root, True

    while cur:
        node = nodes[cur]
        visited.append(cur)
        elapsed = (time.time() - t0) / 60

        if node.get("class") == "root":
            cur = (node.get("on_keep") or [None])[0]
            continue

        if node.get("class") == "terminal":
            break

        # --- budget: check BEFORE spending, and stop with a usable result ---
        cost = node.get("cost_launches", 0)
        if launches + cost > max_launches or elapsed > max_minutes:
            stopped = (f"budget guard at {cur}: {launches}+{cost} launches "
                       f"vs {max_launches}, {elapsed:.0f} min vs {max_minutes}")
            log(f"  STOP  {stopped}")
            break

        # --- skip: inactive, or the fingerprint says it cannot help ---
        why = None
        if node.get("status") != "active":
            why = f"status={node.get('status')}"
        elif lossless_only and node.get("class") == "lossy":
            why = "lossless_only: the lossy branch is parked"
        elif node.get("applicable_when"):
            try:
                if not Predicate(node["applicable_when"]).evaluate(ctx):
                    why = node["applicable_when"]
            except Exception as e:
                why = f"predicate error: {e}"
        if why:
            skipped.append((cur, why))
            log(f"  skip  {cur:32s} {why[:70]}")
            ctx.measurements[cur] = NodeMeasurement(kept=False)
            cur = (node.get("on_keep") or [None])[0]
            continue

        # --- measure every variant ---
        variants = _variants(node, incumbent_cfg, ctx)
        probes = node.get("probes", [])
        benches = node.get("quality_benchmarks", [])
        measured: list[Trial] = []
        for var in variants:
            t = evaluator.measure(var, probes=probes, benchmarks=benches, node_id=cur)
            launches += 1
            measured.append(t)
            trials.append(t)
            record(t)

        # --- keep or revert ---
        eligible = [t for t in measured if t.slo_ok]
        best = max(eligible, key=lambda t: t.goodput) if eligible else None
        threshold = incumbent_goodput * (1 + band)
        keep = bool(best) and (incumbent_goodput == 0 or best.goodput > threshold)

        if keep and node.get("class") == "lossy" and best.quality:
            # Two thresholds, and both are needed. tolerance is the measured
            # noise floor: a delta under it is not real, so rejecting on it
            # would reject on noise. budget is the user's allowance: a delta
            # over it is unacceptable however real. Between them the loss is
            # genuine AND affordable, which is exactly a frontier point.
            budget = ctx.slo.quality_budget
            for b, v in best.quality.items():
                ref = ctx.quality_baseline.get(b)
                if ref is None:
                    continue
                delta = ref - v
                tol, _ = STORE.quality_tolerance(ctx.fingerprint, b)
                tol = ctx.quality_tolerance.get(b, tol)
                if delta <= tol:
                    continue                       # within noise; not a real loss
                if budget is not None and delta > budget:
                    keep = False
                    log(f"        quality gate: {b} {ref:.4f} -> {v:.4f} "
                        f"(-{delta:.4f}) exceeds allow_loss {budget:.1%}")
                    break
                log(f"        quality:      {b} -{delta:.4f} "
                    f"(> tolerance {tol:.4f}, within budget)")

        if best and node.get("class") == "checkpoint" and best.quality:
            # lossless_complete: quality cannot have moved, so whatever moved IS
            # the noise. Adopt it as the tolerance every later lossy node uses,
            # and re-anchor the baseline to the post-lossless incumbent.
            for b, v in best.quality.items():
                ref = ctx.quality_baseline.get(b)
                if ref is not None:
                    ctx.quality_tolerance[b] = max(abs(ref - v), 1e-4)
            log(f"        measured quality tolerance: "
                + "  ".join(f"{k}={v:.4f}" for k, v in ctx.quality_tolerance.items()))
            ctx.quality_baseline.update(best.quality)

        if best:
            best.kept = keep
            ctx.measurements[cur] = NodeMeasurement(
                kept=keep, goodput=best.goodput, ttft_p99_ms=best.ttft_p99_ms,
                itl_p99_ms=best.itl_p99_ms, quality=best.quality,
                spec_acceptance_rate=best.diagnostics.get("spec_acceptance_rate"),
                config=best.config)
            delta = (best.goodput / incumbent_goodput - 1) if incumbent_goodput else 0.0
            log(f"  {'KEEP' if keep else 'revert':>5} {cur:32s} "
                f"{best.goodput:8.1f} goodput  {delta:+7.1%}  "
                f"({len(variants)} variant{'s' if len(variants) > 1 else ''})")
            if keep:
                incumbent_cfg, incumbent_goodput = dict(best.config), best.goodput
        else:
            ctx.measurements[cur] = NodeMeasurement(kept=False)
            log(f"  revert {cur:32s} no variant satisfied the SLO")

        last_kept = keep
        nxt = node.get("on_keep") if keep else node.get("on_revert")
        cur = (nxt or [None])[0]

    return Result(trials=trials, incumbent=incumbent_cfg, visited=visited,
                  skipped=skipped, launches=launches,
                  minutes=(time.time() - t0) / 60, stopped_early=stopped)


def report(res: Result, log=print) -> None:
    log(f"\n{'='*74}")
    log(f"visited {len(res.visited)} nodes, skipped {len(res.skipped)}, "
        f"{res.launches} launches, {res.minutes:.1f} min")
    if res.stopped_early:
        log(f"stopped early: {res.stopped_early}")
    log(f"\nPARETO FRONTIER  ({len(res.frontier())} of {len(res.trials)} measurements)\n")
    log(f"  {'node':32s} {'goodput':>9s} {'quality':>8s} {'ttft p99':>9s} {'mem':>7s}")
    log("  " + "-" * 70)
    for t in res.frontier():
        q = f"{t.min_quality:.4f}" if t.quality else "     --"
        log(f"  {t.node_id:32s} {t.goodput:9.1f} {q:>8s} "
            f"{t.ttft_p99_ms:8.0f}ms {t.memory_gb:6.1f}G"
            f"{'' if t.kept else '   (reverted)'}")
