"""Put the search methods side by side, on every config each of them tried.

    python compare.py runs/08b-seqdag runs/08b-yolo runs/08b-pb \
        --baseline runs/08b-baseline

Reads whatever the methods wrote and produces two tables: one row per METHOD
(what it shipped, what it cost) and one row per CONFIG (everything anyone
measured). No number here is recomputed -- if a column is empty the method did
not measure it, and that absence is itself a result.

WHAT MAKES A COMPARISON FAIR, AND WHAT DOES NOT

Same model, same trace, same SLO, same DAG. compare.py CHECKS this rather than
assuming it: every run carries a provenance stamp naming its model, trace hash,
vLLM version and SLO, and rows whose stamps disagree are reported as
incomparable instead of being quietly tabulated together. Goodput is defined
against the SLO, so two runs with different SLOs produce numbers that look like
they belong in one column and do not.

The instrument has to match too. A swept peak and a pinned point are different
measurements of different things; the baseline is produced by eval_repro, which
PINS concurrency, so it is reported separately and never ranked against the
methods.

WHAT TO READ

  ships       the config the method would deploy -- its actual answer
  best seen   the best config it MEASURED. A gap means the method walked past
              something better than it shipped, which is a property of the
              method, not an accident.
  launches    the cost. A method that wins by spending triple has not won.
  failed      launches that died. They cost the same and buy nothing.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load(d: str | Path) -> dict | None:
    f = Path(d) / "result.json"
    if not f.exists():
        return None
    r = json.loads(f.read_text())
    r["_dir"] = str(d)
    # run.py's traversal writes a different shape than MethodRunner does. Map it
    # rather than changing run.py: the walk's output is the format everything
    # else was built to match, and rewriting it would invalidate every run on
    # disk.
    if "method" not in r:
        inc = r.get("incumbent") or {}
        peak = r.get("incumbent_peak") or {}
        trials = r.get("trials") or []
        base = r.get("baseline")
        chosen = None
        kept = [t for t in trials if t.get("kept")]
        q_ship = (kept[-1].get("quality") if kept else None) or \
                 ((base or {}).get("quality") or {})
        ship_id = kept[-1].get("node_id") if kept else "stage_1_3 (seed)"
        if peak.get("goodput"):
            chosen = {"node_id": ship_id, "goodput": peak["goodput"],
                      "concurrency": peak.get("concurrency"),
                      "ttft_p99_ms": peak.get("ttft_p99_ms"),
                      "itl_p99_ms": peak.get("itl_p99_ms"),
                      "quality": (inc.get("quality") or q_ship),
                      "config": inc.get("config") or inc,
                      "diagnostics": {"slo_attainment": peak.get("slo_attainment")}}
        r = {**r, "method": "seqDAG",
             "provenance": (base or {}).get("provenance") or {},
             "launches": r.get("launches"), "minutes": r.get("minutes"),
             "failed_launches": sum(1 for t in trials if not t.get("goodput")),
             "chosen": chosen,
             "best_seen": max((t for t in trials if t.get("goodput")),
                              key=lambda t: t["goodput"], default=None),
             "trials": ([{**base, "node_id": "stage_1_3 (seed)"}] if base else []) + trials,
             "_dir": str(d)}
    return r


def g(t, *keys, default=None):
    for k in keys:
        if t and t.get(k) is not None:
            return t[k]
    return default


def num(v, spec="{:.1f}", dash="-"):
    try:
        return spec.format(v) if v is not None else dash
    except (TypeError, ValueError):
        return dash


def acc(t, bench):
    q = (t or {}).get("quality") or {}
    return q.get(bench)


# Axes for the joint frontier. Same directions the traversal uses, so a point
# that is non-dominated here would be non-dominated inside a single run too.
AXES = {"goodput": "max", "quality": "max", "ttft_p99_ms": "min"}


def pareto(points: list[dict], axes: dict) -> list[int]:
    """Indices of the non-dominated points over `axes`.

    Computed across ALL methods pooled, which is the comparison worth having:
    "who had the single highest goodput" rewards one lucky launch, while "whose
    points survive against everyone else's" asks whether a method found trades
    the others missed. A method can own several frontier points without ever
    holding the top one.

    A point missing any axis is dropped rather than defaulted. Filling a missing
    accuracy with 0 would dominate nothing and be dominated by everything;
    filling it with the baseline's score would assert a measurement nobody made.
    """
    ok = [i for i, p in enumerate(points)
          if all(p.get(k) is not None and math.isfinite(p[k]) for k in axes)]
    out = []
    for i in ok:
        a = points[i]
        beaten = False
        for j in ok:
            if i == j:
                continue
            b = points[j]
            better = any((b[k] > a[k]) if d == "max" else (b[k] < a[k])
                         for k, d in axes.items())
            no_worse = all((b[k] >= a[k]) if d == "max" else (b[k] <= a[k])
                           for k, d in axes.items())
            if no_worse and better:
                beaten = True
                break
        if not beaten:
            out.append(i)
    return out


def _label(t: dict) -> str:
    """node_id, plus the quantization variant when the node has several."""
    c = t.get("config") or {}
    q, qb = c.get("quantize"), c.get("quantize_bits")
    v = (f"{q}@{qb}" if q and qb else q) or ""
    return f"{t.get('node_id')}:{v}" if v else str(t.get("node_id"))


def collect(runs, bench, baseline=None) -> list[dict]:
    """Every measured point from every method, flattened for plotting."""
    pts = []
    for r in runs:
        ship = (r.get("chosen") or {}).get("node_id")
        live = [t for t in (r.get("trials") or []) if t.get("goodput")]
        # A node_id is not unique: weight_autoquantize covers four quantization
        # variants, and matching on the name alone starred all four. The shipped
        # config is the BEST variant of the node that was kept, so resolve it to
        # exactly one trial here rather than letting the plot guess.
        cand = [t for t in live if t.get("node_id") == ship]
        ship_t = max(cand, key=lambda t: t["goodput"]) if cand else None
        for t in live:
            pts.append({
                "method": r["method"],
                "label": _label(t),
                "goodput": t.get("goodput"),
                "quality": (t.get("quality") or {}).get(bench),
                "ttft_p99_ms": t.get("ttft_p99_ms"),
                "itl_p99_ms": t.get("itl_p99_ms"),
                "concurrency": t.get("concurrency"),
                "inherited": bool(t.get("quality_inherited")),
                "shipped": t is ship_t,
            })
    if baseline:
        pts.append({**baseline, "method": "baseline", "inherited": False,
                    "shipped": False})
    return pts


def plot_frontier(pts: list[dict], path: str, bench: str, title: str) -> None:
    """Two panels: the accuracy trade, and the latency trade.

    Two because they answer different questions and one plot cannot. Goodput
    against accuracy is the headline -- what did the speed cost? Goodput against
    TTFT is where the SLO actually bites, and a config can look free on the
    first panel while sitting against the latency wall on the second.
    """
    import matplotlib
    matplotlib.use("Agg")          # headless; this runs on the GPU box
    import matplotlib.pyplot as plt

    methods = sorted({p["method"] for p in pts})
    colour = dict(zip(methods, plt.cm.tab10.colors))
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.5))

    panels = [("quality", f"{bench} accuracy", "max", axs[0]),
              ("ttft_p99_ms", "TTFT p99 (ms)", "min", axs[1])]
    for key, ylab, direction, ax in panels:
        axes = {"goodput": "max", key: direction}
        front = set(pareto(pts, axes))
        for m in methods:
            idx = [i for i, p in enumerate(pts) if p["method"] == m
                   and p.get(key) is not None]
            if not idx:
                continue
            # Three draws, because the three states mean different things and
            # must not borrow each other's marker: measured, inherited (an
            # assumption, so hollow), and shipped (the method's actual answer).
            first = True
            for shipped, inherited, mk, size in (
                    (False, False, "o", 44), (False, True, "o", 44),
                    (True, None, "*", 260)):
                sel = [i for i in idx
                       if pts[i]["shipped"] == shipped
                       and (inherited is None or pts[i]["inherited"] == inherited)]
                if not sel:
                    continue
                ax.scatter([pts[i]["goodput"] for i in sel],
                           [pts[i][key] for i in sel], s=size, marker=mk,
                           facecolors="none" if inherited else colour[m],
                           edgecolors=colour[m], linewidths=1.4,
                           label=(m if first else None), zorder=4 if shipped else 3)
                first = False
        fp = sorted(front, key=lambda i: pts[i]["goodput"])
        if len(fp) > 1:
            ax.plot([pts[i]["goodput"] for i in fp], [pts[i][key] for i in fp],
                    color="0.35", lw=1.2, ls="--", zorder=2,
                    label="joint Pareto front")
        for i in set(front) | {i for i, q in enumerate(pts) if q["shipped"]}:
            if pts[i].get(key) is None:
                continue
            if pts[i]["shipped"] or len(front) < 8:
                ax.annotate(pts[i]["label"][:30], (pts[i]["goodput"], pts[i][key]),
                            fontsize=7, xytext=(4, 4), textcoords="offset points",
                            color="0.25")
        ax.set_xlabel("goodput (tok/s)  ->  better")
        ax.set_ylabel(ylab + ("  ->  better" if direction == "max" else "  <-  better"))
        ax.grid(alpha=0.25, lw=0.6)
        ax.set_title(f"goodput vs {ylab}", fontsize=10)
    axs[0].legend(fontsize=8, loc="best")
    fig.suptitle(title, fontsize=11)
    fig.text(0.5, 0.005, "star = the config that method ships   |   hollow = accuracy "
             "INHERITED, not measured   |   dashed = frontier over all methods pooled",
             ha="center", fontsize=7.5, color="0.35")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(path, dpi=150)
    print(f"\n  wrote {path}")


def main() -> int:
    ap = argparse.ArgumentParser(prog="compare", description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", help="method run dirs")
    ap.add_argument("--baseline", help="an eval_repro run dir, reported separately")
    ap.add_argument("--benchmark", default="math_500")
    ap.add_argument("--plot", nargs="?", const="auto", default=None,
                    help="write a Pareto frontier plot. Bare --plot derives the "
                         "path from the first run dir.")
    ap.add_argument("--demand", type=float, default=None,
                    help="tok/s of demand for replica counts; default reads it "
                         "from the runs' own fingerprints")
    a = ap.parse_args()

    runs = [r for r in (load(d) for d in a.runs) if r]
    if not runs:
        print("  no result.json in any of those directories")
        return 1

    # Two runs of the same method are a legitimate thing to compare -- three
    # seqDAG runs disagreeing with each other is how path dependence was found.
    # They must not collapse into one label.
    seen = {}
    for r in runs:
        seen.setdefault(r["method"], []).append(r)
    for m, group in seen.items():
        if len(group) > 1:
            for r in group:
                r["method"] = f"{m}/{Path(r['_dir']).name.split('-')[-1]}"[:14]

    # --- comparability, checked rather than assumed
    stamps = {}
    for r in runs:
        p = r.get("provenance") or {}
        stamps[r["method"]] = (p.get("model"), p.get("trace_sha"),
                               json.dumps(p.get("slo"), sort_keys=True),
                               p.get("vllm"))
    distinct = set(stamps.values())
    print()
    if len(distinct) > 1:
        print("  *** THESE RUNS ARE NOT COMPARABLE ***")
        for m, s in stamps.items():
            print(f"    {m:10s} model={s[0]} trace={s[1]} slo={s[2]} vllm={s[3]}")
        print("  Goodput is defined against the SLO, so a differing SLO alone makes")
        print("  these columns different measurements. Fix the inputs and re-run.\n")
    elif not any(distinct) or not list(distinct)[0][0]:
        print("  NOTE: runs carry no provenance stamp, so comparability could not")
        print("  be verified. Re-run with a current build to get one.\n")

    # --- per method
    print(f"  METHODS  ({a.benchmark} for accuracy)")
    hdr = (f"    {'method':14s} {'ships':>9s} {'L':>4s} {'TTFT p99':>9s} "
           f"{'ITL p99':>8s} {'SLO':>5s} {'acc':>7s} {'launch':>7s} {'fail':>5s} "
           f"{'min':>5s} {'best seen':>10s}")
    print(hdr); print("    " + "-" * (len(hdr) - 4))
    rows = []
    for r in sorted(runs, key=lambda x: -(g(x.get("chosen"), "goodput") or 0)):
        c, b = r.get("chosen"), r.get("best_seen")
        gp = g(c, "goodput")
        d = (c or {}).get("diagnostics") or {}
        rows.append((r["method"], gp))
        print(f"    {r['method']:14s} "
              f"{num(gp):>9s} "
              f"{str(g(c, 'concurrency') or '-'):>4s} "
              f"{num(g(c, 'ttft_p99_ms'), '{:.0f}ms'):>9s} "
              f"{num(g(c, 'itl_p99_ms'), '{:.1f}ms'):>8s} "
              f"{num(d.get('slo_attainment'), '{:.0%}'):>5s} "
              f"{num(acc(c, a.benchmark), '{:.4f}'):>7s} "
              f"{r.get('launches') or 0:>7d} {r.get('failed_launches') or 0:>5d} "
              f"{(r.get('minutes') or 0):>5.0f} "
              f"{num(g(b, 'goodput')):>10s}")

    if len(rows) > 1 and rows[0][1]:
        best_m, best_g = rows[0]
        print(f"\n    {best_m} ships the highest goodput. Margins:")
        for m, gp in rows[1:]:
            if gp:
                print(f"      vs {m:14s} {best_g/gp:.2f}x")

    # --- baseline, kept apart on purpose
    if a.baseline:
        f = Path(a.baseline) / "eval.json"
        if f.exists():
            e = json.loads(f.read_text())
            res = next(iter(e["results"].values()))
            s = res.get("serving") or {}
            print(f"\n  BASELINE (eval_repro, concurrency PINNED -- not ranked above)")
            print(f"    stock     {s.get('goodput', 0):9.1f} "
                  f"{str(s.get('concurrency','-')):>4s} "
                  f"{s.get('ttft_p99_ms',0):7.0f}ms {s.get('itl_p99_ms',0):6.1f}ms "
                  f"{s.get('slo_attainment',0):5.0%} {res['mean']:7.4f}")
            print(f"    A pinned point and a swept peak measure different things;")
            print(f"    this row says what stock does at ONE operating point.")
        else:
            print(f"\n  baseline: no eval.json in {a.baseline}")

    # --- every config anyone measured
    print(f"\n  EVERY CONFIG MEASURED")
    hdr = (f"    {'method':14s} {'label':30s} {'tok/s':>8s} {'L':>4s} "
           f"{'TTFT':>8s} {'ITL':>7s} {'SLO':>5s} {'acc':>7s}")
    print(hdr); print("    " + "-" * (len(hdr) - 4))
    for r in runs:
        for t in r.get("trials") or []:
            d = t.get("diagnostics") or {}
            gp = t.get("goodput")
            q = acc(t, a.benchmark)
            inh = "~" if t.get("quality_inherited") else " "
            print(f"    {r['method']:14s} {str(t.get('node_id'))[:30]:30s} "
                  f"{(num(gp) if gp else 'FAILED'):>8s} "
                  f"{str(t.get('concurrency') or '-'):>4s} "
                  f"{(num(t.get('ttft_p99_ms'), '{:.0f}ms') if gp else '-'):>8s} "
                  f"{(num(t.get('itl_p99_ms'), '{:.1f}ms') if gp else '-'):>7s} "
                  f"{num(d.get('slo_attainment'), '{:.0%}'):>5s} "
                  f"{(f'{q:.4f}{inh}' if q is not None else '-'):>7s}")
    print(f"\n    ~ = accuracy INHERITED from the baseline, not measured on that")
    print(f"        config. Run the walk with --quality-every-node to remove these.")

    # --- who owns the joint frontier
    bpt = None
    if a.baseline:
        f = Path(a.baseline) / "eval.json"
        if f.exists():
            e = json.loads(f.read_text())
            res = next(iter(e["results"].values()))
            sv = res.get("serving") or {}
            bpt = {"label": "stock (pinned)", "goodput": sv.get("goodput"),
                   "quality": res.get("mean"), "ttft_p99_ms": sv.get("ttft_p99_ms"),
                   "itl_p99_ms": sv.get("itl_p99_ms"),
                   "concurrency": sv.get("concurrency")}

    pts = collect(runs, a.benchmark, bpt)
    front = pareto(pts, AXES)
    if front:
        print(f"\n  JOINT PARETO FRONTIER  "
              f"(goodput max, {a.benchmark} max, TTFT p99 min)")
        own = {}
        for i in front:
            own[pts[i]["method"]] = own.get(pts[i]["method"], 0) + 1
        for m, c in sorted(own.items(), key=lambda x: -x[1]):
            print(f"    {m:14s} {c:2d} of {len(front)} non-dominated points")
        print(f"    Owning points is not the same as shipping the best one: a")
        print(f"    method can find trades the others missed and still ship a")
        print(f"    worse config than a rival's.")
        skipped = len(pts) - len([i for i, q in enumerate(pts)
                                  if all(q.get(k) is not None for k in AXES)])
        if skipped:
            print(f"    {skipped} measured point(s) omitted for want of an axis -- "
                  f"usually an unmeasured accuracy.")

    if a.plot:
        path = a.plot
        if path == "auto":
            path = f"runs/{Path(a.runs[0]).name.rsplit('-', 1)[0]}-frontier.png"
        try:
            model = ((runs[0].get("provenance") or {}).get("model")) or "?"
            plot_frontier(pts, path, a.benchmark,
                          f"{model} -- search methods compared "
                          f"({len(pts)} measured configs)")
        except Exception as e:
            print(f"\n  plot failed: {type(e).__name__}: {e}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
