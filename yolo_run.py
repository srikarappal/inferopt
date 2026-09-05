"""Turn everything on, measure it, compare against turning nothing on.

    python yolo_run.py --model Qwen/Qwen3.5-0.8B --trace data/trace_shared.jsonl \
        --ttft-p99 500 --itl-p99 250 --run-dir runs/yolo

This is what a capable user does without a search: read the docs, switch on
every lossless technique that sounds applicable, and ship it. It is included
here as a REAL METHOD, not a strawman, because it has two genuine advantages
and one decisive weakness, and the comparison should show all three.

The advantages: it is the cheapest method by a wide margin -- two distinct
configurations against the walk's n+1 and screening's 12 -- and when every
factor helps, the all-on cell IS the optimum, so it finds the best possible
answer in two launches and no search can beat it.

The weakness: it measures one contrast with its entire budget. It answers "is
the bundle better than nothing" precisely and "which part of it helped" not at
all, so when one factor in the bundle is harmful, yolo ships that harm and
cannot see it. On the MoE, four of five lossless factors measured NEGATIVE
against the seed -- prefix_caching -22.9%, max_model_len_rightsize -17.3%,
chunked_prefill -17.0%, max_num_batched_tokens -24.3% -- with only
spec_decode_ngram positive at +42.2%. All-on there means switching on four
regressions to get one win.

Whether that is the common case is exactly what this comparison is for. It is
NOT established, and a simulation cannot establish it.

BOTH CELLS ARE REPEATED. With one launch each, yolo's answer is a single
difference measured against ~5% across-launch spread -- the same fragility that
made the greedy walk disagree with itself three times. Repeating both cells
costs two launches and removes the objection that yolo lost to noise.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="yolo_run", description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--ttft-p99", type=float, default=500)
    ap.add_argument("--itl-p99", type=float, default=250)
    ap.add_argument("--qps", type=float, default=None,
                    help="arrival rate; overrides the trace's arrival_ts")
    ap.add_argument("--dag", default="dag/llm.json")
    ap.add_argument("--run-dir", default="runs/yolo")
    ap.add_argument("--repeats", type=int, default=2,
                    help="LAUNCHES per cell. Separate launches, because "
                         "across-launch spread is what a single-contrast method "
                         "is exposed to.")
    ap.add_argument("--no-quality", action="store_true",
                    help="skip the accuracy benchmark on every config")
    ap.add_argument("--benchmark", default="math_500")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--port", type=int, default=8400)
    a = ap.parse_args()

    from methods import MethodRunner, setup
    from pb_screen import factors_from_dag
    from run import seed_config

    fp, slo, ctx = setup(a.model, a.trace, a.ttft_p99, a.itl_p99, a.qps)
    dag = json.loads(Path(a.dag).read_text())

    base = seed_config(fp)
    factors = factors_from_dag(dag, ctx)
    if not factors:
        print("  no applicable lossless factors for this workload")
        return 1

    # All-on is the union of every factor's ON state. Later factors win a key
    # collision, which is a real property of the method: a user switching
    # everything on has no principle for resolving two techniques that set the
    # same flag, and neither does this.
    all_on = dict(base)
    for f in factors:
        all_on.update(f["on"])

    print(f"\n  model     {fp.model.id}  {fp.model.n_params_b:.1f}B")
    print(f"  factors   {len(factors)}: {', '.join(f['id'] for f in factors)}")
    print(f"  budget    2 cells x {a.repeats} launch(es) = {2 * a.repeats} launches")
    print(f"  NOTE      yolo measures 2 configurations. It cannot attribute a "
          f"result\n            to any single factor -- that is the trade it makes.\n")

    runner = MethodRunner("yolo", fp, slo, a.trace, a.run_dir,
                          gpu=a.gpu, port=a.port,
                          benchmarks=[] if a.no_quality else [a.benchmark],
                          quality_every=not a.no_quality)

    cells = {"all_off": base, "all_on": all_on}
    got: dict[str, list] = {k: [] for k in cells}
    for rep in range(a.repeats):
        for name, cfg in cells.items():
            got[name].append(runner.measure(cfg, f"{name}-rep{rep + 1}"))

    # The cell's value is the MEAN of its launches, not the best of them.
    # Taking the max would let yolo keep whichever launch got the luckiest
    # draw, which is the failure mode the repeats exist to remove.
    means = {}
    for name, ts in got.items():
        ok = [t.goodput for t in ts if t.goodput]
        means[name] = statistics.fmean(ok) if ok else 0.0

    winner = max(means, key=means.get)
    # Ship the representative launch of the winning cell: the one closest to
    # that cell's mean, so the shipped record is typical rather than extreme.
    pool = [t for t in got[winner] if t.goodput] or got[winner]
    chosen = min(pool, key=lambda t: abs(t.goodput - means[winner]))

    print(f"\n{'=' * 72}")
    print(f"  {'cell':10s} {'mean tok/s':>11s} {'launches':>9s}  configs")
    for name in ("all_off", "all_on"):
        n_on = 0 if name == "all_off" else len(factors)
        print(f"  {name:10s} {means[name]:11.1f} {len(got[name]):9d}  "
              f"{n_on} factor(s) on")
    lift = (means["all_on"] / means["all_off"] - 1) if means["all_off"] else None
    print(f"\n  all-on vs all-off: {lift:+.1%}" if lift is not None else
          "\n  all-off produced no usable measurement")
    print(f"  yolo ships: {winner}")
    if winner == "all_off":
        print(f"  -- switching everything on made it WORSE, and yolo can say "
              f"nothing\n     about which of the {len(factors)} factors caused that.")

    runner.finish(chosen=chosen, extra={
        "cells": {k: {"mean_goodput": means[k],
                      "goodput": [t.goodput for t in got[k]],
                      "config": cells[k]} for k in cells},
        "factors": [f["id"] for f in factors],
        "winner": winner,
        "lift_all_on_vs_all_off": lift,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
