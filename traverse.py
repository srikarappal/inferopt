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
    concurrency: int | None = None
    """In-flight requests this was measured at. NOT a tuning knob -- it is the
    operating point the stage 1.3 sweep found, and by Little's Law (L = lambda*W)
    an outcome of arrival rate and service time rather than an input."""
    curve: list[dict] = field(default_factory=list)
    """Extra (concurrency, goodput) points, for nodes whose curves cross."""
    quality_inherited: bool = False
    """True when quality was carried forward from the baseline rather than
    measured. Lossless nodes cannot move quality, and the equivalence probe is a
    stronger check than a 100-sample benchmark would be -- but the plot still
    needs a quality coordinate for every point, so it is inherited and flagged
    rather than left empty."""

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
                benchmarks: list[str], node_id: str,
                concurrency: int | None = None,
                levels: tuple[int, ...] | list[int] | None = None,
                fixed_concurrency: int | None = None) -> Trial: ...


@dataclass
class Result:
    trials: list[Trial]
    incumbent: dict[str, Any]
    visited: list[str]
    skipped: list[tuple[str, str]]
    launches: int
    minutes: float
    stopped_early: str | None = None
    concurrency: int | None = None
    """The operating point every node was measured at, from the stage 1.3
    sweep. Recorded because a goodput number without the concurrency it was
    taken at is a point on an unnamed curve -- which is what made run four's
    middle rows uninterpretable."""
    baseline: Trial | None = None
    """The stage 1.3 seed measurement. EVERY percentage in this run is a ratio
    against it, so a Result without it cannot be interpreted -- run four
    reported '+307%' and the 307% was of a number that existed only in the
    operator's terminal scrollback."""

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
             journal: str | Path | None = None,
             baseline: Trial | None = None,
             concurrency: int | None = None,
             fixed_concurrency: int | None = None) -> Result:
    """Walk the DAG, measuring each applicable node against the incumbent.

    `journal` is a path to APPEND every Trial to as it completes; the caller
    truncates it and writes the stage 1.3 baseline first, so the file holds the
    whole run. It exists
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

    def _decide(node_id: str, keep: bool, best: Trial, prev: float, band: float) -> None:
        """Append the keep/revert verdict, which record() cannot know yet.

        Written to the same journal as a `"decision"` line so a live run can be
        read without waiting for result.json."""
        if not jpath:
            return
        try:
            with open(jpath, "a") as fh:
                fh.write(json.dumps({
                    "decision": node_id, "kept": keep,
                    "goodput": best.goodput, "prev_goodput": prev,
                    "delta": (best.goodput / prev - 1) if prev else None,
                    "accept_band": band, "concurrency": best.concurrency,
                }, default=str) + "\n")
                fh.flush()
        except Exception as e:
            log(f"        journal: could not record decision ({e})")

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
        t = evaluator.measure(incumbent_cfg, probes=["goodput"], benchmarks=[],
                              node_id="incumbent", concurrency=concurrency,
                              fixed_concurrency=fixed_concurrency)
        # Same inheritance as any lossless node. Without it the incumbent lands
        # on the frontier plot with no accuracy coordinate and is silently
        # dropped from the quality view.
        if not t.quality and ctx.quality_baseline:
            t.quality = dict(ctx.quality_baseline)
            t.quality_inherited = True
        if t.concurrency:
            concurrency = t.concurrency
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
            # Crossing-prone nodes get extra concurrency levels. Everything
            # else is judged at the operating point the stage 1.3 sweep found.
            # `concurrency` anchors a BRACKET the evaluator sweeps, not a fixed
            # point it measures at. It moves whenever a node is kept -- see
            # below. Anchoring it permanently to the seed is what broke run five.
            t = evaluator.measure(var, probes=probes, benchmarks=benches, node_id=cur,
                                  concurrency=concurrency,
                                  fixed_concurrency=fixed_concurrency)
            # Every point needs a quality coordinate for the frontier plot. A
            # lossless node cannot move quality -- that is what makes it
            # lossless, and the equivalence probe verifies it -- so it inherits
            # the baseline's scores rather than spending 5 minutes re-measuring
            # what cannot have changed.
            if not t.quality and node.get("class") != "lossy" and ctx.quality_baseline:
                t.quality = dict(ctx.quality_baseline)
                t.quality_inherited = True
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

            # THE ZERO-WIDTH GATE CHECK. Both numbers exist for the first time
            # here, so this is the first moment the lossy branch can be known to
            # be un-passable.
            #
            # The gate is two thresholds: a delta at or under `tolerance` is
            # dismissed as unresolvable, a delta over `budget` is rejected as
            # unaffordable. If tolerance >= budget there is nothing in between --
            # every lossy node is either ignored or rejected, and the branch
            # cannot keep anything no matter how good it is.
            #
            # Run nine hit exactly this: measured tolerance 0.03 against an
            # allow_loss of 0.03. It is the same shape as the accuracy gate that
            # could never pass, and it must be loud rather than discovered after
            # four launches produce nothing.
            budget = ctx.slo.quality_budget
            if budget is not None:
                blocked = {b: t for b, t in ctx.quality_tolerance.items() if t >= budget}
                if blocked:
                    worst = max(blocked.values())
                    log(f"\n  LOSSY BRANCH CANNOT PASS: measured quality tolerance "
                        f"{worst:.4f} >= allow_loss {budget:.4f}")
                    for b, t in sorted(blocked.items(), key=lambda kv: -kv[1]):
                        log(f"    {b:22s} tolerance {t:.4f}")
                    log(f"  Any loss at or under the tolerance is dismissed as "
                        f"unresolvable; any loss above the budget is rejected as too "
                        f"expensive.\n  With tolerance >= budget there is no band "
                        f"between them, so no lossy node can be kept whatever it "
                        f"measures.")
                    log(f"  Raise --allow-loss above {worst:.4f}, raise the quality "
                        f"sample size so the tolerance shrinks, or run "
                        f"--lossless-only deliberately.")
                    stopped = (f"lossy branch un-passable: tolerance {worst:.4f} >= "
                               f"allow_loss {budget:.4f}")
                    break
            log(f"        measured quality tolerance: "
                + "  ".join(f"{k}={v:.4f}" for k, v in ctx.quality_tolerance.items()))
            ctx.quality_baseline.update(best.quality)

        if best:
            best.kept = keep
            # THE JOURNAL RECORDS THE DECISION SEPARATELY, because record(t) runs
            # right after the measurement -- before this line -- so every trial
            # in trials.jsonl carries kept=False, its dataclass default. Reading
            # the journal mid-run therefore says "nothing was kept" no matter
            # what the traversal decided; the real answer only appears in
            # result.json once the whole run finishes, which is exactly when a
            # journal is least needed. This appends the verdict as its own line
            # so a live run can be read honestly.
            _decide(cur, keep, best, incumbent_goodput, band)
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
                # The operating point moves with the incumbent. These techniques
                # work by raising the concurrency the server can sustain, so a
                # kept node's peak IS the new operating point -- and the next
                # node's bracket must be centred there or it re-measures the old
                # config's range and cannot see any further improvement.
                if (not fixed_concurrency and best.concurrency
                        and best.concurrency != concurrency):
                    log(f"         operating point {concurrency} -> {best.concurrency} "
                        f"(kept config sustains more)")
                    concurrency = best.concurrency
        else:
            ctx.measurements[cur] = NodeMeasurement(kept=False)
            log(f"  revert {cur:32s} no variant satisfied the SLO")

        last_kept = keep
        nxt = node.get("on_keep") if keep else node.get("on_revert")
        cur = (nxt or [None])[0]

    return Result(concurrency=concurrency, baseline=baseline, trials=trials, incumbent=incumbent_cfg, visited=visited,
                  skipped=skipped, launches=launches,
                  minutes=(time.time() - t0) / 60, stopped_early=stopped)


def report(res: Result, log=print, demand_tok_s: float | None = None,
           incumbent_curve: list[dict] | None = None) -> None:
    log(f"\n{'='*74}")
    log(f"visited {len(res.visited)} nodes, skipped {len(res.skipped)}, "
        f"{res.launches} launches, {res.minutes:.1f} min")
    if res.stopped_early:
        log(f"stopped early: {res.stopped_early}")

    b = res.baseline
    if b:
        d = b.diagnostics or {}
        log(f"\nBASELINE  (stage 1.3, the seed config -- every % below is against this)\n")
        log(f"  goodput          {b.goodput:9.1f} tok/s"
            f"   ({d.get('goodput_req_s', 0):.2f} req/s)")
        log(f"  throughput       {d.get('throughput', float('nan')):9.1f} tok/s")
        log(f"  ttft p99         {b.ttft_p99_ms:9.0f} ms")
        log(f"  itl p99          {b.itl_p99_ms:9.0f} ms")
        log(f"  slo attainment   {d.get('slo_attainment', 0):9.0%}")
        for k, v in (b.quality or {}).items():
            log(f"  {k:16s} {v:9.4f}")
    else:
        log(f"\nBASELINE  NOT RECORDED -- percentages below have no anchor.")

    log(f"\nPARETO FRONTIER  ({len(res.frontier())} of {len(res.trials)} measurements)\n")
    if res.concurrency:
        log(f"  measured at concurrency {res.concurrency} (the stage 1.3 sweep's peak)\n")
    hdr = (f"  {'node':28s} {'goodput':>9s} {'vs base':>9s} {'quality':>8s} "
           f"{'ttft p99':>9s} {'L':>5s} {'replicas':>9s}")
    log(hdr)
    log("  " + "-" * (len(hdr) - 2))
    for t in res.frontier():
        q = f"{t.min_quality:.4f}" if t.quality else "     --"
        rel = f"{(t.goodput/b.goodput - 1)*100:+8.1f}%" if b and b.goodput else "       --"
        if t.quality_inherited:
            q = q + "~"                       # inherited, not measured
        rep = (f"{__import__('math').ceil(demand_tok_s / max(1e-9, t.goodput)):9d}"
               if demand_tok_s else "       --")
        log(f"  {t.node_id:28s} {t.goodput:9.1f} {rel:>9s} {q:>8s} "
            f"{t.ttft_p99_ms:8.0f}ms {str(t.concurrency or '-'):>5s} {rep}"
            f"{'' if t.kept else '  (reverted)'}")

    log(f"\n  quality marked ~ was inherited from the baseline: a lossless node "
        f"cannot move it,\n  and the equivalence probe is a stronger check than a "
        f"100-sample benchmark.")
    if b and b.goodput:
        best = max((t for t in res.trials if t.slo_ok), key=lambda t: t.goodput, default=None)
        if best:
            log(f"\n  best measured   {best.goodput:.1f} tok/s at {best.node_id} "
                f"({best.goodput/b.goodput:.2f}x the baseline's {b.goodput:.1f})")
            if demand_tok_s:
                import math
                r0 = math.ceil(demand_tok_s / max(1e-9, b.goodput))
                r1 = math.ceil(demand_tok_s / max(1e-9, best.goodput))
                log(f"  fleet           {r0} replicas -> {r1} to serve "
                    f"{demand_tok_s:.0f} tok/s at the SLO")

    # THE CURVE OF THE CONFIG ACTUALLY CHOSEN. The seed's sweep is not it: on the
    # MoE run the seed collapsed from 41.8 goodput at L=2 to 7.2 at L=8, while
    # the incumbent PEAKS at L=8 with 65.6. Printing the seed's curve beside the
    # incumbent's config understates the deployed capacity roughly ninefold.
    if incumbent_curve:
        log(f"\n  CAPACITY OF THE CHOSEN CONFIG  (its own sweep, not the seed's)")
        log(f"    {'L':>5s} {'goodput':>9s} {'ttft p99':>9s} {'slo':>6s}")
        peak = max(incumbent_curve, key=lambda pt: pt.get("goodput", 0))
        for pt in incumbent_curve:
            mark = "   <- peak" if pt is peak else ""
            log(f"    {pt.get('concurrency', 0):5d} {pt.get('goodput', 0):9.1f} "
                f"{pt.get('ttft_p99_ms', 0):8.0f}m {pt.get('slo_attainment', 0):6.0%}{mark}")
        if peak is incumbent_curve[0] or peak is incumbent_curve[-1]:
            log(f"    NOTE: the peak sits at an EDGE of the swept range, so the true")
            log(f"    optimum may lie outside it -- treat this as a lower bound.")

