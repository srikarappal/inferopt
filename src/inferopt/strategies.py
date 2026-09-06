"""One protocol the three search strategies implement, so results compare.

    strategy = get_strategy("sequential")
    out = strategy.search(ctx, evaluator, seed_config, budget)

The data model was already shared -- methods.MethodRunner gives yolo and PB an
identical Trial and result.json, and compare.py successfully translates the
sequential walk's output into the same shape. That translation working is the
evidence these three can sit behind one interface. What was missing is an entry
point and a return type, which is all this file adds.

THE ASYMMETRY IS DECLARED, NOT DESIGNED AWAY

The sequential walk chains: each node is measured against the incumbent, and a
kept node becomes the background for the next. yolo and PB do not: every one of
yolo's two cells and every one of PB's twelve design rows is built from the SAME
fixed seed.

That is not an inconsistency to iron out. PB's arithmetic requires it -- the
difference of means only isolates a factor if every row shares a background --
and chaining PB would destroy the property the method exists for. So the
protocol exposes `chains_incumbent` and lets callers reason about it, rather
than forcing one behaviour on all three.

The practical consequence, which cost a night of GPU: --seed-from-run applies to
the sequential walk only.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class SearchResult:
    """What every strategy returns, whatever it did internally."""

    method: str
    trials: list[Any] = field(default_factory=list)
    chosen: Any = None
    """The config this strategy would SHIP. Recorded explicitly rather than
    inferred as max(goodput), because the strategies disagree about what
    shipping means: the walk ships its incumbent, which is not always the best
    trial it saw, while yolo ships whichever of its two cells read higher."""
    best_seen: Any = None
    """The best config MEASURED. A gap against `chosen` is a property of the
    strategy -- it walked past something better than it shipped -- and is worth
    reporting rather than hiding."""
    frontier: list[Any] = field(default_factory=list)
    launches: int = 0
    minutes: float = 0.0
    extra: dict = field(default_factory=dict)
    """Strategy-specific detail: PB's effects and design, yolo's cells, the
    walk's visited/skipped lists. Deliberately not flattened into the common
    fields -- forcing them into one schema would either lose information or
    invent empty keys for two strategies out of three."""

    @property
    def failed_launches(self) -> int:
        return sum(1 for t in self.trials if not getattr(t, "goodput", 0))


@runtime_checkable
class Measurer(Protocol):
    """What a strategy needs in order to measure a config.

    Deliberately NARROWER than evaluator.Evaluator, whose measure() takes
    probes, benchmarks, levels and a concurrency anchor. A strategy has no
    business choosing those -- they are the measurement contract, and letting
    each strategy pick its own is exactly how "goodput" came to mean different
    things in the three implementations. methods.MethodRunner satisfies this
    and fixes those choices in one place.
    """

    def measure(self, config: dict, label: str) -> Any: ...


@runtime_checkable
class Strategy(Protocol):
    name: str
    chains_incumbent: bool

    def search(self, ctx, runner: Measurer, seed: dict, *,
               budget_launches: int | None = None,
               log=print) -> SearchResult: ...


# ------------------------------------------------------------------ adapters

class SequentialStrategy:
    """The greedy DAG walk. Cheapest real search: one launch per factor.

    Chains an incumbent, so it is the only strategy that can usefully continue
    from a previous run's answer -- and the only one for which the lossy stage
    inherits the lossless one rather than starting over.
    """

    name = "sequential"
    chains_incumbent = True

    def __init__(self, dag: dict, *, lossless_only: bool = False,
                 force_benchmarks: list[str] | None = None,
                 concurrency: int | None = None,
                 baseline=None, provenance: dict | None = None):
        self.dag = dag
        self.lossless_only = lossless_only
        self.force_benchmarks = force_benchmarks
        self.concurrency = concurrency
        self.baseline = baseline
        self.provenance = provenance

    def search(self, ctx, runner, seed: dict, *,
               budget_launches: int | None = None, log=print) -> SearchResult:
        # traverse() drives the RAW evaluator, because the DAG chooses probes,
        # benchmarks and a concurrency bracket per node -- that per-node choice
        # is what the DAG is for. So this reaches through the runner for it,
        # while yolo and screen use the narrow measure(config, label).
        from inferopt.traverse import traverse
        evaluator = getattr(runner, "ev", runner)
        t0 = time.time()
        ctx.incumbent = dict(seed)
        res = traverse(self.dag, ctx, evaluator, log=log,
                       lossless_only=self.lossless_only,
                       baseline=self.baseline,
                       concurrency=self.concurrency,
                       provenance=self.provenance,
                       force_benchmarks=self.force_benchmarks)
        trials = ([self.baseline] if self.baseline else []) + list(res.trials)
        kept = [t for t in res.trials if t.kept]
        ok = [t for t in trials if t.goodput]
        return SearchResult(
            method=self.name,
            trials=trials,
            chosen=(kept[-1] if kept else self.baseline),
            best_seen=(max(ok, key=lambda t: t.goodput) if ok else None),
            frontier=list(res.frontier()),
            launches=res.launches,
            minutes=res.minutes if res.minutes else (time.time() - t0) / 60,
            extra={"incumbent": res.incumbent, "visited": res.visited,
                   "skipped": res.skipped, "stopped_early": res.stopped_early},
        )


class YoloStrategy:
    """Everything on, against everything off. Two configs, whatever the budget.

    The cheapest method by a wide margin, and the RIGHT answer whenever every
    factor helps -- there the all-on cell is the optimum and no search can beat
    two launches. It is included as a real candidate for that reason, not as a
    foil. Its weakness is that it spends the whole budget on one contrast, so
    when one factor in the bundle is harmful it ships that harm and cannot see
    it.
    """

    name = "yolo"
    chains_incumbent = False

    def __init__(self, factors: list[dict], *, repeats: int = 2):
        self.factors = factors
        self.repeats = repeats

    def search(self, ctx, runner: Measurer, seed: dict, *,
               budget_launches: int | None = None, log=print) -> SearchResult:
        import statistics
        t0 = time.time()
        all_on = dict(seed)
        for f in self.factors:
            all_on.update(f["on"])
        cells = {"all_off": dict(seed), "all_on": all_on}

        trials: list[Any] = []
        got: dict[str, list] = {k: [] for k in cells}
        for rep in range(self.repeats):
            for name, cfg in cells.items():
                if budget_launches and len(trials) >= budget_launches:
                    break
                t = runner.measure(cfg, f"{name}-rep{rep + 1}")
                trials.append(t)
                got[name].append(t)

        # The cell's value is the MEAN of its launches, not the best of them.
        # Taking the max would keep whichever launch drew luckiest, which is the
        # failure the repeats exist to remove.
        means = {}
        for name, ts in got.items():
            ok = [t.goodput for t in ts if t.goodput]
            means[name] = statistics.fmean(ok) if ok else 0.0
        winner = max(means, key=means.get) if means else None
        pool = [t for t in got.get(winner, []) if t.goodput] or got.get(winner, [])
        chosen = (min(pool, key=lambda t: abs(t.goodput - means[winner]))
                  if pool else None)
        ok = [t for t in trials if t.goodput]
        return SearchResult(
            method=self.name, trials=trials, chosen=chosen,
            best_seen=(max(ok, key=lambda t: t.goodput) if ok else None),
            launches=len(trials), minutes=(time.time() - t0) / 60,
            extra={"cells": {k: {"mean_goodput": means[k],
                                 "goodput": [t.goodput for t in got[k]]}
                             for k in cells},
                   "winner": winner,
                   "lift_all_on_vs_all_off": (
                       means["all_on"] / means["all_off"] - 1
                       if means.get("all_off") else None)},
        )


class ScreenStrategy:
    """Plackett-Burman screening, then a factorial over what survives.

    Measures every factor across six different backgrounds instead of one, so a
    keep/revert call does not rest on a single pair of launches against
    across-launch spread of the same size. Costs the most: twelve launches
    before it recommends anything, and it recommends rather than chooses --
    stage 2 is what turns a ranking into a config.
    """

    name = "screen"
    chains_incumbent = False

    def __init__(self, factors: list[dict], *, repeats: int = 1,
                 survivors: int = 3, stage2: bool = True):
        self.factors = factors
        self.repeats = repeats
        self.survivors = survivors
        self.stage2 = stage2

    def search(self, ctx, runner: Measurer, seed: dict, *,
               budget_launches: int | None = None, log=print) -> SearchResult:
        import statistics

        from inferopt.pb_screen import aliases, effects, lenth, pb_design
        t0 = time.time()
        design, n = pb_design(len(self.factors))
        trials: list[Any] = []
        rows: list[float | None] = []

        for r, row in enumerate(design):
            cfg = dict(seed)
            for c, f in enumerate(self.factors):
                if row[c]:
                    cfg.update(f["on"])
            vals = []
            for rep in range(self.repeats):
                if budget_launches and len(trials) >= budget_launches:
                    break
                t = runner.measure(cfg, f"pb-row{r + 1}-rep{rep + 1}")
                t.diagnostics = dict(t.diagnostics or {})
                t.diagnostics["pb_row"] = r + 1
                t.diagnostics["pb_factors_on"] = {
                    self.factors[c]["id"]: bool(row[c])
                    for c in range(len(self.factors))}
                trials.append(t)
                if t.goodput:
                    vals.append(t.goodput)
            rows.append(statistics.fmean(vals) if vals else None)

        eff = effects(design, self.factors, rows)
        usable = [x["effect"] for x in eff if x.get("effect") is not None]
        L = lenth(usable) if usable else {}
        # Lenth decides when there is no replication: with one launch per row the
        # two groups' spread is mostly the OTHER factors moving, which a balanced
        # design has already cancelled out of the estimate.
        if self.repeats < 2 and L.get("me"):
            surv = [x["id"] for x in eff if x.get("effect") is not None
                    and abs(x["effect"]) > L["me"]]
        else:
            surv = [x["id"] for x in eff if x.get("effect") is not None
                    and x.get("se") and abs(x["effect"]) > 2 * x["se"]]

        chosen = None
        top: list[str] = []
        pinned: dict[str, bool] = {}
        if self.stage2:
            ranked = [x["id"] for x in eff if x.get("effect") is not None]
            top = (surv or ranked)[: max(1, self.survivors)]
            pinned = {x["id"]: x["effect"] > 0 for x in eff
                      if x["id"] not in top and x.get("effect") is not None}
            by_id = {f["id"]: f for f in self.factors}
            for i in range(2 ** len(top)):
                if budget_launches and len(trials) >= budget_launches:
                    break
                bits = [(i >> j) & 1 for j in range(len(top))]
                cfg = dict(seed)
                for fid, on in pinned.items():
                    if on:
                        cfg.update(by_id[fid]["on"])
                for fid, on in zip(top, bits):
                    if on:
                        cfg.update(by_id[fid]["on"])
                label = "s2-" + ("+".join(f for f, b in zip(top, bits) if b)
                                 or "pinned_only")
                trials.append(runner.measure(cfg, label))
            s2 = [t for t in trials if str(t.node_id).startswith("s2-") and t.goodput]
            chosen = max(s2, key=lambda t: t.goodput) if s2 else None

        ok = [t for t in trials if t.goodput]
        return SearchResult(
            method=self.name, trials=trials,
            chosen=chosen or (max(ok, key=lambda t: t.goodput) if ok else None),
            best_seen=(max(ok, key=lambda t: t.goodput) if ok else None),
            launches=len(trials), minutes=(time.time() - t0) / 60,
            extra={"design": [[bool(x) for x in r] for r in design],
                   "factors": eff, "survivors": surv, "varied": top,
                   "pinned": pinned, "lenth": L,
                   "aliases": aliases(design, self.factors)},
        )


STRATEGIES = {
    "sequential": SequentialStrategy,
    "yolo": YoloStrategy,
    "screen": ScreenStrategy,
}
