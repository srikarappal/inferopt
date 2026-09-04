"""Plackett-Burman screening: which techniques matter, before tuning any of them.

    python pb_screen.py --model Qwen/Qwen3-14B --trace data/trace_shared.jsonl \
        --ttft-p99 500 --itl-p99 250 --run-dir runs/pb

An ALTERNATIVE stage 1 to the greedy DAG walk in run.py, not a replacement for
it -- both are kept so they can be compared. The lossy ladder is untouched and
still runs afterwards: screening assumes independent on/off factors, and
quantization is a choice among mutually exclusive alternatives, each costing
hours to produce. There is nothing to screen there; you measure all of them,
which is what eval_ladder.sh does.

WHY SCREEN INSTEAD OF WALK

The greedy walk decides keep/revert from ONE measurement against a 5% accept
band -- and a single launch's goodput spread IS about 5%. Two identical
Qwen3-30B-A3B runs consequently disagreed: spec_decode_depth reverted in one
(60.0 against a 66.6 incumbent) and was kept in the other (62.6), because their
seed measurements differed. The final configs differed too.

Plackett-Burman removes that failure by construction. Every factor is ON in
half the runs and off in the other half, and every PAIR of factors is balanced
across those halves, so the difference of the two means isolates one factor
while the others average out:

    effect(f) = mean(goodput | f ON) - mean(goodput | f off)

With 6 runs per side the noise on each mean falls to ~5%/sqrt(6) = 2%, and on
the difference to ~2.9% -- so effects above roughly 6% are resolvable, while
the walk cannot distinguish anything from a 5% band at all.

It also never discards a factor. The walk reverts a node and never revisits it,
so a technique that only helps in combination is lost: chunked_prefill reverted
on both MoE runs at L=2, and we never learned whether it wins once spec decode
has pushed the operating point to L=8. Screening measures every factor across
six different backgrounds.

WHAT THIS DOES NOT DO

It is resolution III: main effects are CONFOUNDED with two-way interactions. A
large effect for chunked_prefill may really be chunked_prefill x spec_decode,
and 12 runs cannot separate them. Screening tells you WHICH factors to spend
the next measurements on; it does not tell you what they do, and it does not
choose a value. That is stage 2 -- a full factorial over the survivors, where
interactions come out cleanly and values get swept. Stage 2 is not optional
polish; it is what makes stage 1 interpretable.

The ON level is most of the experiment. `chunked_prefill = ON` needs a
max_num_batched_tokens; a poor choice reports "chunked_prefill does not help"
when the truth is "not at that value". Levels are therefore taken from the DAG's
own action/sweep entries rather than from vLLM defaults, so they inherit
whatever reasoning went into the nodes.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

# Plackett-Burman generating rows. Cyclic rotation of the row, then a final
# all-low run. N must be a multiple of 4 and handles up to N-1 factors.
PB_GENERATORS = {
    12: "++-+++---+-",
    20: "++--++++-+-+----++-",
    24: "+++++-+-++--++--+-+----",
}


def pb_design(n_factors: int) -> tuple[list[list[bool]], int]:
    """Rows of True/False, one row per run, one column per factor."""
    n = next((k for k in sorted(PB_GENERATORS) if k > n_factors), None)
    if n is None:
        raise ValueError(
            f"{n_factors} factors needs a Plackett-Burman design larger than "
            f"{max(PB_GENERATORS)}; add a generating row or drop factors")
    gen = PB_GENERATORS[n]
    rows = [[gen[(j - i) % (n - 1)] == "+" for j in range(n - 1)] for i in range(n - 1)]
    rows.append([False] * (n - 1))
    return [r[:n_factors] for r in rows], n


def factors_from_dag(dag: dict, ctx) -> list[dict]:
    """One factor per applicable lossless node, with its ON config.

    Taken from the DAG so the screen stays in step with the technique
    definitions rather than duplicating them. A node whose applicable_when is
    False for this workload is not a factor -- it would be off in all 12 runs
    and the column would be degenerate.
    """
    from predicates import Predicate
    from traverse import _variants

    out = []
    for n in dag["nodes"]:
        if n.get("class") != "lossless" or n.get("status") != "active":
            continue
        gate = n.get("applicable_when")
        if gate:
            try:
                if not Predicate(gate).evaluate(ctx):
                    continue
            except Exception:
                continue          # needs a measurement that does not exist yet
        # The ON state is the node's action plus its FIRST sweep entry. Screening
        # asks "does this technique help at a sensible setting", not "which
        # setting is best" -- that is stage 2.
        variants = _variants(n, {}, ctx)
        on = variants[0]
        if not on:
            continue
        out.append({"id": n["id"], "on": on,
                    "n_sweep": len(n.get("sweep") or []) or 1})
    return out


def effects(design, factors, results) -> list[dict]:
    """mean(ON) - mean(off) per factor, with the noise on that difference."""
    out = []
    for c, f in enumerate(factors):
        hi = [results[r] for r in range(len(design)) if design[r][c] and results[r] is not None]
        lo = [results[r] for r in range(len(design)) if not design[r][c] and results[r] is not None]
        if len(hi) < 2 or len(lo) < 2:
            out.append({**f, "effect": None, "reason": "too few usable runs"})
            continue
        mh, ml = statistics.fmean(hi), statistics.fmean(lo)
        # Pooled spread of the two groups, propagated to their difference.
        sh = statistics.stdev(hi) if len(hi) > 1 else 0.0
        sl = statistics.stdev(lo) if len(lo) > 1 else 0.0
        se = math.sqrt(sh ** 2 / len(hi) + sl ** 2 / len(lo))
        out.append({**f, "effect": mh - ml, "mean_on": mh, "mean_off": ml,
                    "se": se, "n_on": len(hi), "n_off": len(lo),
                    "relative": (mh - ml) / ml if ml else None})
    return sorted(out, key=lambda x: -abs(x.get("effect") or 0))


def main() -> int:
    ap = argparse.ArgumentParser(prog="pb_screen", description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--ttft-p99", type=float, default=500)
    ap.add_argument("--itl-p99", type=float, default=250)
    ap.add_argument("--dag", default="dag/llm.json")
    ap.add_argument("--run-dir", default="runs/pb")
    ap.add_argument("--repeats", type=int, default=1,
                    help="LAUNCHES per design row. Repeats must be separate "
                         "launches: across-launch spread measured 5x "
                         "within-launch, and the accept band is built on the "
                         "former, so repeated windows in one launch measure the "
                         "wrong noise.")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--port", type=int, default=8300)
    a = ap.parse_args()

    from evaluator import VllmEvaluator
    from fingerprint import Context
    from request import InferOptRequest, build_fingerprint
    from run import free_port, seed_config

    fp, slo = build_fingerprint(InferOptRequest(
        model=a.model, trace=a.trace, ttft_p99_ms=a.ttft_p99, itl_p99_ms=a.itl_p99))
    ctx = Context(fingerprint=fp, slo=slo)
    dag = json.loads(Path(a.dag).read_text())

    base = seed_config(fp)              # all factors OFF, same seed the walk uses
    factors = factors_from_dag(dag, ctx)
    if not factors:
        print("  no applicable lossless factors for this workload")
        return 1
    design, n = pb_design(len(factors))

    run_dir = Path(a.run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  model     {fp.model.id}  {fp.model.n_params_b:.1f}B")
    print(f"  design    Plackett-Burman N={n}, {len(factors)} factors, "
          f"{len(design)} rows x {a.repeats} launch(es) = "
          f"{len(design) * a.repeats} launches")
    print(f"  factors   {', '.join(f['id'] for f in factors)}")
    print(f"\n  Each factor is ON in {sum(design[r][0] for r in range(len(design)))} "
          f"of {len(design)} rows. No row is 'everything on' -- each is a different\n"
          f"  mix, which is what lets the difference of means isolate one factor.\n")

    from provenance import banner, provenance
    meta = provenance(ap, a, fp, extra={
        "design": "plackett-burman", "n": n,
        "factors": [f["id"] for f in factors],
        "rows": [[bool(x) for x in row] for row in design],
        "base_config": base,
    })
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(banner(meta, run_dir / "run_meta.json") + "\n")

    ev = VllmEvaluator(fp, slo, a.trace, run_dir, gpu=a.gpu, port=free_port(a.port))
    journal = run_dir / "rows.jsonl"
    journal.write_text("")
    results, t0 = [], time.time()

    for r, row in enumerate(design):
        cfg = dict(base)
        for c, f in enumerate(factors):
            if row[c]:
                cfg.update(f["on"])
        on = [factors[c]["id"] for c in range(len(factors)) if row[c]] or ["(none)"]
        print(f"  row {r+1:2d}/{len(design)}  +{(time.time()-t0)/60:5.1f}m  ON: "
              f"{', '.join(on)}")
        got = []
        for rep in range(a.repeats):
            try:
                t = ev.measure(cfg, probes=["goodput"], benchmarks=[],
                               node_id=f"pb-row{r+1}-rep{rep+1}")
                got.append(t.goodput)
                print(f"            launch {rep+1}: {t.goodput:7.1f} tok/s  "
                      f"ttft {t.ttft_p99_ms:5.0f}ms  L={t.concurrency}")
            except Exception as e:
                print(f"            launch {rep+1} FAILED: {type(e).__name__}: "
                      f"{str(e)[:90]}")
        val = statistics.fmean(got) if got else None
        results.append(val)
        with open(journal, "a") as fh:
            fh.write(json.dumps({"row": r + 1, "on": on, "config": cfg,
                                 "goodput": got, "mean": val}, default=str) + "\n")
            fh.flush()

    # ---------------------------------------------------------------- report
    eff = effects(design, factors, results)
    usable = [x for x in results if x is not None]
    noise = (statistics.stdev(usable) if len(usable) > 1 else 0.0)
    print(f"\n{'='*74}")
    print(f"  EFFECTS  (mean goodput with the factor ON, minus with it off)\n")
    print(f"  {'factor':26s} {'effect':>9s} {'rel':>8s} {'+/- se':>8s}  verdict")
    print("  " + "-" * 70)
    for x in eff:
        if x.get("effect") is None:
            print(f"  {x['id']:26s} {'--':>9s}  {x.get('reason','')}")
            continue
        sig = abs(x["effect"]) > 2 * x["se"] if x["se"] else abs(x["effect"]) > 0
        print(f"  {x['id']:26s} {x['effect']:+9.1f} {x['relative']:+7.0%} "
              f"{x['se']:8.1f}  {'RESOLVED' if sig else 'inside noise'}")
    print(f"\n  Effects are confounded with two-way interactions -- this is a")
    print(f"  resolution III design. A large effect says 'spend stage 2 here',")
    print(f"  not 'this factor causes it'.")

    survivors = [x["id"] for x in eff if x.get("effect") is not None
                 and x["se"] and abs(x["effect"]) > 2 * x["se"]]
    print(f"\n  STAGE 2: full factorial over the resolved factors, sweeping VALUES")
    if survivors:
        k = len(survivors[:4])
        print(f"    factors {survivors[:4]}")
        print(f"    {2**k} configs for the on/off factorial (interactions come out "
              f"clean), plus a value sweep on any that carry a setting.")
    else:
        print(f"    nothing resolved above the noise. Either the techniques do not")
        print(f"    help on this workload, or --repeats needs raising: one launch")
        print(f"    per row leaves ~{noise:.1f} tok/s of spread on each mean.")

    (run_dir / "effects.json").write_text(json.dumps(
        {"factors": eff, "results": results, "survivors": survivors,
         "design": [[bool(x) for x in row] for row in design]},
        indent=2, default=str))
    print(f"\n  wrote {run_dir}/effects.json and {journal}")
    print(f"  total {(time.time()-t0)/60:.0f} min\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
