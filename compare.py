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
        if peak.get("goodput"):
            chosen = {"node_id": "incumbent", "goodput": peak["goodput"],
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


def main() -> int:
    ap = argparse.ArgumentParser(prog="compare", description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", help="method run dirs")
    ap.add_argument("--baseline", help="an eval_repro run dir, reported separately")
    ap.add_argument("--benchmark", default="math_500")
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
    print(f"        config. Run the walk with --quality-every-node to remove these.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
