"""Score optimizer STRATEGIES against a measured table, with no GPU.

    python replay.py --seeds 400 --budget 24
    python replay.py --table runs/table-14b.json --seeds 400

The problem this exists to solve is circular. To claim "screening beats the
sequential walk" you have to run both many times, and each run is 3-9 GPU-hours.
So the comparison never gets made, and the project decides between approaches on
argument instead of measurement. That is exactly how two identical MoE runs came
to disagree with each other about spec_decode_depth -- reverted at 60.0 in one,
kept at 62.6 in the other -- with no way to say which was the accident.

The way out is the tabular-benchmark pattern from NAS-Bench and HPOBench:
enumerate the configuration space ONCE, pay the GPU cost once, and afterwards a
"run" of any optimizer is a table lookup. Thousands of seeds then cost seconds,
and questions that were unaffordable become routine.

WHAT IS BEING SCORED, AND WHAT IS NOT

Every strategy here is a real candidate, not a foil. The sequential DAG walk is
not a legacy baseline being retired: it is cheap, it never needs a second stage,
and on a space with one dominant factor it is hard to beat. Screening costs 12
launches before it recommends anything. Which of those is right depends on the
shape of the space, and the shape of the space is an empirical question -- which
is the whole point of having a table.

REGRET IS MEASURED AGAINST TRUTH, NOT AGAINST WHAT THE OPTIMIZER SAW

A strategy picks the config with the best OBSERVED goodput, and is then scored
on that config's TRUE goodput. This is deliberate and it is the only honest
accounting. An optimizer that chases noise will pick a cell that got a lucky
draw, and it should be charged for the config it actually shipped, not for the
number it happened to read. Scoring on observed values would reward variance.

NOISE IS PART OF THE BENCHMARK

Cells hold k separate launches, and a lookup draws one at random. Replaying
against noiseless cell means would flatter every strategy, and would flatter
most the ones that decide on small differences -- the sequential walk resolves
keep/revert at a 5% band, and measured across-launch spread IS 5%. Removing the
noise would quietly delete the exact effect under study. NAS-Bench-201 stores 3
seeds per architecture for the same reason.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path
from typing import Callable

from pb_screen import pb_design, effects


# ---------------------------------------------------------------- the table

class Table:
    """Goodput for every point in a binary factor space, with repeats.

    A cell key is a bitstring over `factors`, so "101000" means factors 0 and 2
    ON. Cells may be MISSING: a half-fraction table measures 32 of 64 points,
    and a strategy that asks for an unmeasured cell has to be told so rather
    than handed a fabricated number.
    """

    def __init__(self, factors: list[str], cells: dict[str, list[float]],
                 meta: dict | None = None):
        self.factors = factors
        self.cells = cells
        self.meta = meta or {}
        if not cells:
            raise ValueError("a table with no cells cannot score anything")

    # ---- construction

    @classmethod
    def from_file(cls, path: str | Path) -> "Table":
        d = json.loads(Path(path).read_text())
        return cls(d["factors"], d["cells"], d.get("meta"))

    @classmethod
    def synthetic(cls, effects: dict[str, float], *, base: float = 29.7,
                  interactions: dict[tuple[str, str], float] | None = None,
                  noise: float = 0.05, repeats: int = 3, seed: int = 0) -> "Table":
        """A full table from a stated effect model. For testing the harness and
        for asking what-if questions the real table cannot answer.

        This is NOT a substitute for measurement. It answers "if the space had
        this shape, which strategy would win", which is worth knowing before
        spending 15 GPU-hours -- and worth distrusting afterwards.

        Defaults are set from what we measured on Qwen3-30B-A3B: a seed of 29.7
        tok/s and 5% across-launch spread.
        """
        rng = random.Random(seed)
        names = list(effects)
        inter = interactions or {}

        # ONE implementation of the model. It was written twice -- once to
        # generate the noisy cells and once to record the noiseless truth -- and
        # a mutation that deleted interactions from the first copy left the
        # second copy agreeing with the tests. Two copies of a definition
        # disagree eventually, and here the disagreement would be invisible:
        # every strategy would be scored against a truth the table did not have.
        def model(mask: str) -> float:
            on = {names[j] for j in range(len(names)) if mask[j] == "1"}
            g = base + sum(effects[f] for f in on)
            g += sum(v for (a, b), v in inter.items() if a in on and b in on)
            return max(g, 0.1)

        truth = {format(i, f"0{len(names)}b"): None for i in range(2 ** len(names))}
        truth = {m: model(m) for m in truth}
        cells = {m: [max(0.1, g * (1 + rng.gauss(0, noise))) for _ in range(repeats)]
                 for m, g in truth.items()}
        t = cls(names, cells, {"synthetic": True, "noise": noise,
                               "base": base, "effects": effects,
                               "interactions": {f"{a}*{b}": v
                                                for (a, b), v in inter.items()}})
        t._truth = truth
        return t

    # ---- lookup

    def has(self, mask: str) -> bool:
        return mask in self.cells

    def draw(self, mask: str, rng: random.Random) -> float | None:
        """One launch. Returns None for a cell the table never measured."""
        c = self.cells.get(mask)
        return rng.choice(c) if c else None

    def truth(self, mask: str) -> float:
        """The cell's true goodput, free of launch noise.

        For a synthetic table this is exact. For a measured one it is the mean
        of the repeats, which is the best estimate available -- and the reason
        repeats are not optional.
        """
        t = getattr(self, "_truth", None)
        if t is not None:
            return t[mask]
        return statistics.fmean(self.cells[mask])

    def virtual_best(self) -> tuple[str, float]:
        """The best cell in the table -- what a perfect optimizer would find
        with unlimited budget. Every regret is measured against this."""
        m = max(self.cells, key=self.truth)
        return m, self.truth(m)

    def true_effects(self) -> dict[str, float]:
        """Main effect of each factor over the WHOLE table -- a full factorial.

        This is what stage 1 is trying to estimate from 12 runs, and having it
        exactly is the single most valuable thing the table provides: comparing
        it to a screen's estimate measures whether resolution-III confounding
        actually cost us anything on this space, rather than leaving it as a
        caveat in a docstring.
        """
        out = {}
        for j, f in enumerate(self.factors):
            on = [self.truth(m) for m in self.cells if m[j] == "1"]
            off = [self.truth(m) for m in self.cells if m[j] == "0"]
            out[f] = statistics.fmean(on) - statistics.fmean(off) if on and off else 0.0
        return out


# ------------------------------------------------------------- strategies
#
# Signature: strategy(table, budget, rng) -> list[(mask, observed)]
#
# The returned list is every launch the strategy spent, in order. The harness
# scores it; a strategy never reports its own result, so none of them can score
# themselves generously.

Trace = list[tuple[str, float]]


def _all_off(table: Table) -> str:
    return "0" * len(table.factors)


def sequential_dag(table: Table, budget: int, rng: random.Random,
                   band: float = 0.05) -> Trace:
    """The DAG walk as traverse.py performs it. A first-class strategy.

    One launch per factor, in DAG order, each compared against the incumbent's
    STORED value rather than a fresh measurement -- which is what the real
    traversal does, and it is why a lucky incumbent measurement propagates
    forward through every later decision.

    A reverted factor is never revisited. That is the property screening was
    built to avoid, and it is also why this is cheap: 6 factors cost 7 launches
    where a screen costs 12 before it says anything at all.
    """
    if budget < 1:
        return []
    mask = _all_off(table)
    obs = table.draw(mask, rng)
    if obs is None:
        return []
    trace: Trace = [(mask, obs)]
    incumbent = obs
    for j in range(len(table.factors)):
        if len(trace) >= budget:
            break
        cand = mask[:j] + "1" + mask[j + 1:]
        o = table.draw(cand, rng)
        if o is None:
            continue
        trace.append((cand, o))
        if o > incumbent * (1 + band):
            mask, incumbent = cand, o
    return trace


def pb_then_factorial(table: Table, budget: int, rng: random.Random,
                      survivors: int = 3, repeats: int = 1) -> Trace:
    """Screen every factor, then sweep the survivors exhaustively.

    Stage 1 spends a fixed 12 launches (for up to 11 factors) and chooses
    nothing. Stage 2 spends 2^k on the k factors with the largest effects, with
    the rest pinned to whichever level the screen preferred. The screen is not
    the deliverable; it is what makes stage 2 affordable.

    A budget too small for stage 1 returns what it managed, which will score
    badly -- correctly so. Screening genuinely cannot deliver under 12 launches,
    and hiding that would be the kind of favourable accounting this file exists
    to prevent.
    """
    n = len(table.factors)
    design, _ = pb_design(n)
    trace: Trace = []
    rows: list[float | None] = []
    for row in design:
        if len(trace) >= budget:
            rows.append(None)
            continue
        mask = "".join("1" if row[j] else "0" for j in range(n))
        vals = []
        for _ in range(repeats):
            if len(trace) >= budget:
                break
            o = table.draw(mask, rng)
            if o is None:
                break
            trace.append((mask, o))
            vals.append(o)
        rows.append(statistics.fmean(vals) if vals else None)

    eff = effects(design, [{"id": f} for f in table.factors], rows)
    ranked = [e for e in eff if e.get("effect") is not None]
    if not ranked:
        return trace
    top = [e["id"] for e in ranked[:survivors]]
    # Non-survivors are pinned to the level the screen preferred. Pinning them
    # OFF instead would confound stage 2 with a change of background.
    pinned = {e["id"]: ("1" if e["effect"] > 0 else "0")
              for e in ranked[survivors:]}
    for i in range(2 ** len(top)):
        if len(trace) >= budget:
            break
        bits = format(i, f"0{len(top)}b")
        mask = "".join(
            bits[top.index(f)] if f in top else pinned.get(f, "0")
            for f in table.factors)
        o = table.draw(mask, rng)
        if o is not None:
            trace.append((mask, o))
    return trace


def random_search(table: Table, budget: int, rng: random.Random) -> Trace:
    """Uniform sampling. The reference every search method must beat.

    A strategy that cannot beat random search on a space is not exploiting
    structure, and reporting it without this comparison would be meaningless.
    """
    keys = list(table.cells)
    trace: Trace = []
    for _ in range(budget):
        m = rng.choice(keys)
        o = table.draw(m, rng)
        if o is not None:
            trace.append((m, o))
    return trace


def yolo(table: Table, budget: int, rng: random.Random) -> Trace:
    """Everything off, then everything on, repeated -- the original proposal.

    It measures one contrast with all the budget, so it answers "does the bundle
    help" precisely and "which part of it helped" not at all. Kept because it is
    the cheapest way to get a defensible headline number, and because it should
    be scored rather than dismissed.
    """
    off, on = _all_off(table), "1" * len(table.factors)
    trace: Trace = []
    while len(trace) < budget:
        for m in (off, on):
            if len(trace) >= budget:
                break
            o = table.draw(m, rng)
            if o is not None:
                trace.append((m, o))
    return trace


def pb_anchored(table: Table, budget: int, rng: random.Random,
                survivors: int = 3, repeats: int = 1) -> Trace:
    """Screen, then sweep the survivors AROUND THE BEST ROW THE SCREEN SAW.

    pb_then_factorial pins each non-survivor to the sign of its estimated main
    effect. That estimate is confounded -- resolution III aliases main effects
    with two-way interactions -- and replay measured the damage: on a space with
    two interactions, a factor with a true effect of +2.0 was estimated at +0.55,
    close enough to zero that finite repeats flipped its sign and stage 2 pinned
    it the wrong way, putting the best cell permanently out of reach.

    The screen has already measured 12 real configurations, and it throws that
    away when it decides levels from signs. Anchoring on the best row observed
    replaces a confounded inference with a direct measurement: whatever
    combination of the non-survivors that row happens to hold is known to work,
    interactions and all, because it was run.

    This does not fix the confounding. It stops stage 2 from depending on it.
    """
    n = len(table.factors)
    design, _ = pb_design(n)
    trace: Trace = []
    rows: list[float | None] = []
    for row in design:
        if len(trace) >= budget:
            rows.append(None)
            continue
        mask = "".join("1" if row[j] else "0" for j in range(n))
        vals = []
        for _ in range(repeats):
            if len(trace) >= budget:
                break
            o = table.draw(mask, rng)
            if o is None:
                break
            trace.append((mask, o))
            vals.append(o)
        rows.append(statistics.fmean(vals) if vals else None)

    eff = effects(design, [{"id": f} for f in table.factors], rows)
    ranked = [e for e in eff if e.get("effect") is not None]
    if not ranked or not trace:
        return trace
    top = [e["id"] for e in ranked[:survivors]]
    anchor = max(trace, key=lambda x: x[1])[0]
    for i in range(2 ** len(top)):
        if len(trace) >= budget:
            break
        bits = format(i, f"0{len(top)}b")
        mask = "".join(
            bits[top.index(f)] if f in top else anchor[j]
            for j, f in enumerate(table.factors))
        o = table.draw(mask, rng)
        if o is not None:
            trace.append((mask, o))
    return trace


STRATEGIES: dict[str, Callable[..., Trace]] = {
    "pb_anchored": pb_anchored,
    "sequential_dag": sequential_dag,
    "pb_then_factorial": pb_then_factorial,
    "random_search": random_search,
    "yolo": yolo,
}


# ---------------------------------------------------------------- scoring

def regret(table: Table, trace: Trace) -> float:
    """Fraction of the achievable goodput the strategy left on the table.

    0.0 is the virtual best; 0.12 means it shipped a config 12% short of the
    best one in the space. Scored on the TRUE value of the config the strategy
    would ship, which is the one with the best value it OBSERVED.
    """
    if not trace:
        return 1.0
    shipped = max(trace, key=lambda x: x[1])[0]
    _, best = table.virtual_best()
    return (best - table.truth(shipped)) / best if best else 1.0


def score(table: Table, name: str, budget: int, seeds: int,
          **kw) -> dict:
    fn = STRATEGIES[name]
    rs = []
    for s in range(seeds):
        rng = random.Random(s)
        rs.append(regret(table, fn(table, budget, rng, **kw)))
    rs.sort()
    return {
        "strategy": name, "budget": budget, "seeds": seeds,
        "mean_regret": statistics.fmean(rs),
        "median_regret": statistics.median(rs),
        "p90_regret": rs[min(len(rs) - 1, int(0.9 * len(rs)))],
        "solved": sum(1 for r in rs if r < 1e-9) / len(rs),
    }


def dominance(table: Table, a: str, b: str, budget: int, seeds: int) -> float:
    """Fraction of seeds where `a` ships a strictly better config than `b`.

    Reported alongside mean regret because a strategy can win on average while
    losing most of the time -- one catastrophic seed for the other strategy is
    enough. Both numbers, or neither.
    """
    wins = 0
    for s in range(seeds):
        ra = regret(table, STRATEGIES[a](table, budget, random.Random(s)))
        rb = regret(table, STRATEGIES[b](table, budget, random.Random(s)))
        wins += ra < rb - 1e-12
    return wins / seeds


def screen_fidelity(table: Table, seeds: int = 200) -> dict:
    """How close a 12-run screen gets to the full-factorial main effects.

    This is the experiment the table exists for. Stage 1 is resolution III, so
    its estimates are confounded with two-way interactions; that is a real
    limitation and this measures its size instead of restating it.

    Reports Spearman rank correlation, because a screen's output is a RANKING --
    which factors to spend stage 2 on. Getting the magnitudes wrong is
    tolerable; ranking a dead factor above a live one is not.
    """
    truth = table.true_effects()
    n = len(table.factors)
    design, _ = pb_design(n)
    hits, corrs = [], []
    top_true = {f for f, _ in sorted(truth.items(), key=lambda x: -abs(x[1]))[:3]}
    for s in range(seeds):
        rng = random.Random(s)
        rows = [table.draw("".join("1" if r[j] else "0" for j in range(n)), rng)
                for r in design]
        est = {e["id"]: (e.get("effect") or 0.0)
               for e in effects(design, [{"id": f} for f in table.factors], rows)}
        top_est = {f for f, _ in sorted(est.items(), key=lambda x: -abs(x[1]))[:3]}
        hits.append(len(top_true & top_est) / len(top_true))
        corrs.append(_spearman([truth[f] for f in table.factors],
                               [est[f] for f in table.factors]))
    return {"top3_recall": statistics.fmean(hits),
            "rank_correlation": statistics.fmean(corrs)}


def _spearman(a: list[float], b: list[float]) -> float:
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    ra, rb = rank(a), rank(b)
    n = len(a)
    if n < 2:
        return 0.0
    ma, mb = statistics.fmean(ra), statistics.fmean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den else 0.0


# ------------------------------------------------------------------- CLI

# Shaped from the Qwen3-30B-A3B lossless run: seed 29.7 tok/s, incumbent 66.6
# with prefix_caching + ngram, chunked_prefill reverted, and across-launch
# spread 5%. It is a MODEL of that space, not a record of it -- the real table
# replaces it. Written down explicitly so the assumptions can be argued with.
MEASURED_LIKE = {
    "prefix_caching": 14.0,
    "max_model_len_rightsize": 2.0,
    # NEGATIVE, and that matters more than its size. The first version of this
    # model gave every factor a positive effect, which made the all-on cell the
    # global optimum -- and YOLO, which measures exactly that cell, scored a
    # perfect 100% at every budget. It looked like a result and was an artifact:
    # on a space where "turn everything on" is the answer, the strategy that
    # tries only that will always win, and nothing has been learned about any of
    # them. Real spaces are not like that, which is why the walk reverts nodes.
    "chunked_prefill": -3.0,
    "max_num_batched_tokens": -2.5,
    "spec_decode_ngram": 20.0,
    "graph_capture": 4.0,
}
MEASURED_LIKE_INTERACTIONS = {
    # chunked_prefill reverted at L=2 on both MoE runs, and was never retried
    # once spec decode had moved the operating point to L=8. If it only pays off
    # in that combination, the sequential walk cannot find it by construction --
    # this is the interaction that motivated screening, so the default table
    # contains one.
    ("chunked_prefill", "spec_decode_ngram"): 6.0,
    ("prefix_caching", "spec_decode_ngram"): 3.0,
}


# Named shapes a configuration space can have. Which strategy wins is a
# property of the SHAPE, not of the strategy -- so the shape has to be varied
# deliberately rather than assumed. Every one of these is a plausible story
# about a serving stack, and they disagree about who wins.
SPACES = {
    "measured_like": (MEASURED_LIKE, MEASURED_LIKE_INTERACTIONS),
    # Nearly everything helps and the wins add up. all-on is near-optimal, so
    # measuring one cell answers the question and screening is 12 wasted
    # launches. This is the space where YOLO is the right tool.
    "mostly_helps": (
        {"a": 14.0, "b": 6.0, "c": 4.0, "d": 3.0, "e": 20.0, "f": 2.0}, {}),
    # One dominant factor, the rest actively harmful -- the shape you get when a
    # stack is already well tuned and most switches are regressions. all-on is
    # BAD here, so YOLO reports a loss and stops; a walk that tries one factor at
    # a time and reverts is well matched to it.
    "one_winner": (
        {"a": 22.0, "b": -4.0, "c": -6.0, "d": -3.0, "e": -5.0, "f": -2.0}, {}),
    # Main effects near zero, the value entirely in pairs. The adversarial case
    # for screening, which ranks by main effect and will therefore rank the live
    # factors as dead. Also the case the sequential walk cannot solve, since it
    # reverts each factor before ever seeing its partner.
    "interaction_trap": (
        {"a": 1.0, "b": -1.0, "c": 0.5, "d": -0.5, "e": 1.0, "f": -1.0},
        {("a", "b"): 18.0, ("c", "d"): 12.0, ("e", "f"): 9.0}),
}


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="replay", description=__doc__.split("\n\n")[0])
    ap.add_argument("--table", help="measured table JSON; default is synthetic")
    ap.add_argument("--seeds", type=int, default=400)
    ap.add_argument("--budgets", default="7,12,16,20,24,32")
    ap.add_argument("--noise", type=float, default=0.05,
                    help="across-launch spread for the synthetic table; 0.05 measured")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--space", default="measured_like", choices=sorted(SPACES),
                    help="which synthetic shape to score against")
    ap.add_argument("--all-spaces", action="store_true",
                    help="score every shape and show that the winner changes")
    ap.add_argument("--no-interactions", action="store_true",
                    help="synthetic table with additive effects only")
    args = ap.parse_args()

    if args.all_spaces:
        return _all_spaces(args)
    if args.table:
        table = Table.from_file(args.table)
        src = f"measured: {args.table}"
    else:
        eff, inter = SPACES[args.space]
        table = Table.synthetic(
            eff, noise=args.noise, repeats=args.repeats,
            interactions=None if args.no_interactions else inter)
        src = (f"synthetic, noise={args.noise:.0%}, "
               f"{'additive' if args.no_interactions else 'with 2 interactions'}")

    vb_mask, vb = table.virtual_best()
    print(f"\n  table      {src}")
    print(f"  factors    {len(table.factors)}  ->  {len(table.cells)} cells")
    print(f"  best cell  {vb_mask}  at {vb:.1f} tok/s")
    print(f"             {', '.join(f for j, f in enumerate(table.factors) if vb_mask[j]=='1') or '(none)'}")
    print(f"  all-off    {table.truth(_all_off(table)):.1f} tok/s")
    if vb_mask == "1" * len(table.factors):
        print(f"\n  WARNING: the all-on cell IS the best cell, so every factor helps and")
        print(f"  the space has no trade-off in it. On such a table YOLO scores a perfect")
        print(f"  0.000 by measuring one point, and the comparison below says nothing")
        print(f"  about any strategy. Treat this run as void unless the real measured")
        print(f"  table genuinely has that shape -- in which case no search is needed.")

    print(f"\n  TRUE main effects (full factorial over every cell)")
    for f, v in sorted(table.true_effects().items(), key=lambda x: -abs(x[1])):
        print(f"    {f:28s} {v:+7.2f}")

    budgets = [int(b) for b in args.budgets.split(",")]
    print(f"\n  MEAN REGRET vs BUDGET   ({args.seeds} seeds; 0.00 = found the best cell)")
    print(f"    {'launches':>10s}  " + "".join(f"{n:>20s}" for n in STRATEGIES))
    for b in budgets:
        row = [score(table, n, b, args.seeds) for n in STRATEGIES]
        best = min(r["mean_regret"] for r in row)
        cells = []
        for r in row:
            mark = "*" if abs(r["mean_regret"] - best) < 1e-9 else " "
            cells.append(f"{mark}{r['mean_regret']:.3f}".rjust(20))
        print(f"    {b:>10d}  " + "".join(cells))
    print(f"    (* = best at that budget)")

    print(f"\n  SOLVED RATE -- fraction of seeds that found the exact best cell")
    print(f"    {'launches':>10s}  " + "".join(f"{n:>20s}" for n in STRATEGIES))
    for b in budgets:
        row = [score(table, n, b, args.seeds) for n in STRATEGIES]
        print(f"    {b:>10d}  " + "".join(f"{r['solved']:>20.0%}" for r in row))

    print(f"\n  HEAD TO HEAD at {budgets[-1]} launches "
          f"(fraction of seeds where the row beats the column)")
    names = list(STRATEGIES)
    print(f"    {'':>20s}" + "".join(f"{n:>20s}" for n in names))
    for a in names:
        cells = "".join(
            f"{'--':>20s}" if a == b
            else f"{dominance(table, a, b, budgets[-1], args.seeds):>20.0%}"
            for b in names)
        print(f"    {a:>20s}{cells}")

    fid = screen_fidelity(table, seeds=min(args.seeds, 200))
    print(f"\n  SCREEN FIDELITY -- a 12-run screen vs the full factorial")
    print(f"    rank correlation with the true effects   {fid['rank_correlation']:.3f}")
    print(f"    top-3 factors correctly identified       {fid['top3_recall']:.0%}")
    print(f"    This is the resolution-III question, measured. A screen that")
    print(f"    ranks the live factors correctly has done its job even if the")
    print(f"    magnitudes are confounded -- stage 2 re-measures those anyway.")
    print()
    return 0

def _all_spaces(args) -> int:
    """Score every shape. The point is that the winner CHANGES.

    A single table can only ever say which strategy suits that one space. Read
    across the rows and the real conclusion appears: there is no strategy that
    wins everywhere, so the useful question is not "which optimizer" but "which
    shape is this stack in", and that is answered by measurement.
    """
    names = list(STRATEGIES)
    print(f"\n  MEAN REGRET BY SPACE SHAPE  "
          f"({args.seeds} seeds, {args.budgets.split(',')[-1]} launches, "
          f"noise {args.noise:.0%})")
    print(f"    {'space':>18s}" + "".join(f"{n:>19s}" for n in names))
    budget = int(args.budgets.split(",")[-1])
    for sname, (eff, inter) in SPACES.items():
        t = Table.synthetic(eff, noise=args.noise, repeats=args.repeats,
                            interactions=inter)
        rs = [score(t, n, budget, args.seeds)["mean_regret"] for n in names]
        best = min(rs)
        cells = "".join(
            (("*" if abs(r - best) < 1e-9 else " ") + f"{r:.3f}").rjust(19)
            for r in rs)
        vb, _ = t.virtual_best()
        print(f"    {sname:>18s}{cells}   best cell {vb}")
    print(f"\n    (* = best for that shape)")

    # The worst case is the number that matters when the shape is unknown, which
    # it always is for a (model, hardware) pair nobody has measured before. An
    # average over shapes would be a claim about how often each shape occurs,
    # and nothing here supports such a claim.
    print(f"\n  WORST CASE ACROSS SHAPES -- the honest statistic when you do not")
    print(f"  yet know which space you are in")
    worst = {}
    for n in names:
        worst[n] = max(
            score(Table.synthetic(eff, noise=args.noise, repeats=args.repeats,
                                  interactions=inter),
                  n, budget, args.seeds)["mean_regret"]
            for eff, inter in SPACES.values())
    for n, w in sorted(worst.items(), key=lambda x: x[1]):
        bar = "#" * int(w * 100)
        print(f"    {n:>19s}  {w:.3f}  {bar}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
