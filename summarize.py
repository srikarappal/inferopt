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
#
# Identified by MESSAGE, not by hash. The hash was hardcoded as ae41dbe and then
# every hash in the repo changed when history was rewritten to correct commit
# authorship -- so the ancestry check silently failed and flagged every row as
# incomparable, including the ones that share the instrument.
SERVING_UNIFIED_SUBJECT = "One serving measurement"


def _serving_unified_commit() -> str | None:
    import subprocess
    try:
        r = subprocess.run(
            ["git", "log", "--format=%H %s", "--all"],
            capture_output=True, text=True, timeout=10)
        for line in r.stdout.splitlines():
            h, _, subj = line.partition(" ")
            if subj.startswith(SERVING_UNIFIED_SUBJECT):
                return h
    except Exception:
        pass
    return None

# Where to look, and what to call it. Order is the story: what you start with,
# what costs nothing, then increasingly aggressive weight changes.
SOURCES = [
    # LOSSLESS -- weights untouched, flags only
    ("stock",          "runs/quantize/q_stock",              "bf16, vLLM defaults"),
    ("stock n=500",    "runs/quantize/q_stock_n500",         "bf16, 500-problem eval"),
    ("stock (older)",  "runs/eval_repro/base",               "bf16, earlier run"),
    ("lossless",       "runs/eval_repro/base_after_runNine", "prefix cache + ngram spec decode"),
    # LOSSY -- weights rewritten
    ("fp8",            "runs/quantize/q_fp8",                "8-bit, load-time flag, no artifact"),
    ("autoquant@6.0",  "runs/quantize/q_autoquant_6.0",      "mixed precision, 6.0 effective bits"),
    ("autoquant@5.15", "runs/quantize/q_autoquant_5.15",     "mixed precision, at this model's floor"),
    ("w4a16",          "runs/quantize/q_w4a16",              "4-bit weights, 16-bit activations"),
    ("nvfp4",          "runs/quantize/q_nvfp4",              "4-bit weights AND activations"),
    ("nvfp4 n=500",    "runs/quantize/q_nvfp4_n500",         "nvfp4, 500-problem eval"),
]

# Which rows changed the weights. The distinction the table exists to make: a
# lossless row cannot move quality by construction, so any movement there is
# measurement noise; a lossy row can, so movement there has to be judged.
LOSSY = {"fp8", "autoquant@6.0", "autoquant@5.15", "w4a16", "nvfp4", "nvfp4 n=500"}

DEMAND_TOK_S = 15.36 * 259      # what this workload needs served on time


MODEL_PREFIX = "Qwen__Qwen3-14B"


def artifact_gb(tag: str) -> str:
    """Size of THIS model's artifact.

    The glob was `*--{tag}`, which also matched the Qwen3-0.6B directories left
    by the smoke test -- so nvfp4 reported 0.6GB, the size of a different model.
    Anchoring on the model prefix stops a leftover test artifact being read as a
    production result.
    """
    p = Path("artifacts") / f"{MODEL_PREFIX}--{tag.replace('@', '_')}"
    if not p.is_dir():
        return ""
    n = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return f"{n / 1e9:.1f}"


def discover(root: Path) -> list[tuple[str, str, str]]:
    """Every eval.json under `root`, labelled from what actually ran.

    The SOURCES table below is a hand-written story about ONE ladder: which
    variants, in which order, with a sentence each. It cannot describe a ladder
    that has not been run yet, and hardcoding a second copy for every benchmark
    is how the artifact glob and the commit check both went stale.

    So: given a directory, read the labels off the runs themselves. run_meta.json
    records the model and benchmark that produced each result, which is exactly
    the description a discovered row needs.
    """
    out = []
    for f in sorted(root.glob("*/eval.json")):
        d = f.parent
        label = d.name[2:] if d.name.startswith("q_") else d.name
        what = ""
        meta = d / "run_meta.json"
        if meta.exists():
            try:
                m = json.loads(meta.read_text())
                args = m.get("args", {})
                model = (args.get("model") or {}).get("value", "")
                bench = (args.get("benchmark") or {}).get("value", "")
                n = (args.get("n") or {}).get("value", "")
                what = f"{Path(str(model)).name}   {bench} n={n}"
            except Exception:
                pass
        out.append((label, str(d), what))
    return out


def main() -> int:
    import sys
    if len(sys.argv) > 1:
        root = Path(sys.argv[1])
        if not root.is_dir():
            print(f"  {root} is not a directory")
            return 1
        sources = discover(root)
        if not sources:
            print(f"  no eval.json under {root} yet")
            return 0
    else:
        sources = SOURCES

    rows = []
    for label, d, what in sources:
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

    # Chronological, not a string match. The old check flagged every commit whose
    # hash merely differed from ae41dbe -- including every commit AFTER it, which
    # are exactly the comparable ones. Anything at or after the unifying commit
    # shares the instrument.
    import subprocess
    unified = _serving_unified_commit()
    def _at_or_after(commit: str) -> bool:
        if not commit or not unified:
            return False
        try:
            r = subprocess.run(["git", "merge-base", "--is-ancestor", unified, commit],
                               capture_output=True, timeout=10)
            return r.returncode == 0
        except Exception:
            return False
    stale = [l for l, w, r, c, wh in rows if r is not None and not _at_or_after(c)]
    if stale:
        print(f"\n  NOTE: serving columns are only comparable across runs measured at or after")
        print(f"  {(unified or '?')[:7]}, which unified eval_repro's measurement with the")
        print(f"  traversal's. Rows from other commits: {', '.join(stale)}.")
        print(f"  Accuracy is unaffected -- that path has not changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
