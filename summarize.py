"""One table across every variant measured, whenever and wherever it was measured.

    python summarize.py

Reads every eval.json under runs/ and prints one comparison. Standalone rather
than embedded in quantize_and_eval.sh for two reasons: the summary is worth
re-running after adding a variant without re-measuring anything, and a script
that is mid-run cannot be edited -- bash reads a script file incrementally as it
executes, so changing it in place corrupts the remainder.

WHAT IS AND IS NOT COMPARABLE

Rows are only comparable if they were measured by the same instrument. The
serving numbers all come from serving_metrics() -- open loop at the trace's
arrival rate, REPEATS windows, worst-value aggregation -- and the commit that
unified that is ae41dbe. Anything measured earlier used a different instrument
and is marked, rather than silently mixed in.

The accuracy column carries its own caveat: MATH-500's measured resolution limit
on this rig is 0.02-0.03 at n=100. A gap smaller than that is evidence of
neither damage nor safety.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

# The commit that made eval_repro use the traversal's serving measurement rather
# than a single closed-loop point. Anything older is not comparable on the
# serving columns.
SERVING_UNIFIED_AT = "ae41dbe"

# Where to look, and what to call it. Order is the story: what you start with,
# what costs nothing, then increasingly aggressive weight changes.
SOURCES = [
    ("stock",         "runs/quantize/q_stock",              "bf16, vLLM defaults"),
    ("stock (older)", "runs/eval_repro/base",               "bf16, earlier run"),
    ("lossless",      "runs/eval_repro/base_after_runNine", "flags only: prefix cache + ngram"),
    ("fp8",           "runs/quantize/q_fp8",                "load-time flag, no artifact"),
    ("autoquant@6.0", "runs/quantize/q_autoquant_6.0",      "mixed precision, 6 effective bits"),
    ("autoquant@5.0", "runs/quantize/q_autoquant_5.0",      "mixed precision, 5 effective bits"),
    ("w4a16",         "runs/quantize/q_w4a16",              "4-bit weights, 16-bit activations"),
    ("nvfp4",         "runs/quantize/q_nvfp4",              "4-bit weights AND activations"),
]

DEMAND_TOK_S = 15.36 * 259      # what this workload needs served on time


def artifact_gb(tag: str) -> str:
    for p in Path("artifacts").glob(f"*--{tag.replace('@', '_')}"):
        n = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        return f"{n / 1e9:.1f}"
    return ""


def main() -> int:
    rows = []
    for label, d, what in SOURCES:
        f = Path(d) / "eval.json"
        if not f.exists():
            rows.append((label, what, None, None, None))
            continue
        r = json.loads(f.read_text())
        res = next(iter(r["results"].values()))
        commit = (r.get("environment", {}).get("inferopt_commit") or "")[:7]
        rows.append((label, what, res, commit, r.get("when", "")[:10]))

    hdr = (f"  {'variant':16s} {'accuracy':>9s} {'spread':>7s} {'goodput':>9s} "
           f"{'thru':>8s} {'ttft p99':>9s} {'slo':>5s} {'GB':>6s} {'replicas':>8s}")
    print(f"\n{hdr}")
    print("  " + "-" * (len(hdr) - 2))

    base = None
    for label, what, res, commit, when in rows:
        if res is None:
            print(f"  {label:16s}     -- not measured --")
            continue
        sv = res.get("serving") or {}
        gp = sv.get("goodput")
        rep = f"{math.ceil(DEMAND_TOK_S / gp)}" if gp else "-"
        print(f"  {label:16s} {res['mean']:9.4f} {res['spread']:7.4f} "
              f"{(gp or 0):9.1f} {sv.get('throughput', 0):8.1f} "
              f"{sv.get('ttft_p99_ms', 0):8.0f}m {sv.get('slo_attainment', 0):5.0%} "
              f"{artifact_gb(label):>6s} {rep:>8s}")
        if label == "stock":
            base = res

    print()
    for label, what, res, commit, when in rows:
        if res is not None:
            print(f"  {label:16s} {what}   [{commit} {when}]")

    # Accuracy deltas, stated against what the eval can actually resolve.
    if base:
        print(f"\n  ACCURACY vs stock ({base['mean']:.4f})")
        limit = max(base["spread"], 0.02)
        for label, what, res, commit, when in rows:
            if res is None or label.startswith("stock"):
                continue
            d = res["mean"] - base["mean"]
            verdict = ("within the eval's resolution -- neither damage nor safety shown"
                       if abs(d) <= limit else
                       ("REAL LOSS" if d < 0 else "real gain"))
            print(f"    {label:16s} {d:+.4f}   {verdict}")
        print(f"\n  resolution limit {limit:.4f} -- measured, not assumed: three passes of the")
        print(f"  same config over the same 100 problems move by this much on their own.")

    # A quantized run that could not hit its requested budget says so.
    for p in sorted(Path("artifacts").glob("*/autoquantize_search.json")):
        a = json.loads(p.read_text())
        if a.get("clamped_to_model_floor"):
            print(f"\n  {p.parent.name}: requested {a['effective_bits_requested']} bits, "
                  f"produced at {a['effective_bits_achieved']} -- this model's floor.")
            print(f"    Embeddings, lm_head and routers are never quantized and set it.")

    stale = [l for l, w, r, c, wh in rows
             if r is not None and c and not c.startswith(SERVING_UNIFIED_AT[:7])]
    if stale:
        print(f"\n  NOTE: serving columns are only comparable across runs measured at or after")
        print(f"  {SERVING_UNIFIED_AT}, which unified eval_repro's measurement with the")
        print(f"  traversal's. Rows from other commits: {', '.join(stale)}.")
        print(f"  Accuracy is unaffected -- that path has not changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
