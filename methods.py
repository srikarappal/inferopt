"""One measurement contract, so search METHODS can be compared instead of
merely each producing a number.

    runner = MethodRunner("yolo", fp, slo, trace, run_dir, ...)
    t = runner.measure(cfg, "all_on")
    runner.finish(chosen="all_on")

Three methods now search the same space -- the sequential DAG walk, Plackett-
Burman screening, and yolo -- and until this existed they emitted three
different shapes of output. run.py wrote a rich trials.jsonl, pb_screen.py
wrote a bespoke rows.jsonl holding only goodput, and yolo did not exist at all.
Comparing them meant reading three formats and trusting that "goodput" meant
the same thing in each, which it did not: the walk sweeps concurrency and
reports the peak of the curve, while pb_screen measured at whatever the
evaluator picked.

So the contract is here, in one place, and the methods are thin:

  - every config a method tries is measured the SAME way, through one
    evaluator.measure call with the same probes, levels and benchmarks
  - every measurement lands in trials.jsonl in the Trial schema that run.py
    already writes, provenance stamp included
  - every method writes result.json with the same keys, including which config
    it would SHIP -- without that a method has no answer, only a history

WHY EVERY CONFIG GETS AN ACCURACY SCORE HERE

The traversal deliberately does not do this: a lossless node cannot move
quality, the equivalence probe is a stronger check than a 100-sample benchmark,
and re-scoring what cannot have changed costs minutes per node. All true, and
the right trade for a production search.

It is the wrong trade for a METHOD COMPARISON. The comparison's whole claim is
"method A ships a better config than method B", and a reader is entitled to see
the accuracy of every config each method considered rather than a baseline
score inherited by assumption. Measuring it also turns "lossless cannot move
quality" from a premise into a result. Set quality_every=False to get the
cheaper production behaviour.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from traverse import Trial


class MethodRunner:
    """Measures configs for one search method and records them comparably."""

    def __init__(self, method: str, fp, slo, trace: str, run_dir: str | Path, *,
                 gpu: str = "0", port: int = 8100,
                 benchmarks: list[str] | None = None,
                 quality_every: bool = True,
                 sweep: bool = True,
                 log=print):
        from evaluator import VllmEvaluator
        from provenance import trial_stamp
        from run import free_port

        self.method = method
        self.fp, self.slo = fp, slo
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.benchmarks = benchmarks if benchmarks is not None else ["math_500"]
        self.quality_every = quality_every
        self.sweep = sweep
        self.log = log
        self.stamp = trial_stamp(fp, trace, slo)
        self.ev = VllmEvaluator(fp, slo, trace, str(self.run_dir),
                                gpu=gpu, port=free_port(port))
        self.trials: list[Trial] = []
        self.journal = self.run_dir / "trials.jsonl"
        self.journal.write_text("")
        self.t0 = time.time()

    # -------------------------------------------------------------- measure

    def measure(self, config: dict, label: str, *,
                quality: bool | None = None,
                levels=None) -> Trial:
        """Measure one config. Never raises: a launch failure is a RESULT.

        A method that dies because one of its candidates would not start is
        being judged on the harness rather than on its search. The failure is
        recorded as a trial with goodput 0.0 and a launch_error diagnostic, so
        it is visible in the comparison as something the method spent a launch
        on -- which is exactly what happened.
        """
        from evaluator import SWEEP_LEVELS
        want_q = self.quality_every if quality is None else quality
        benches = self.benchmarks if want_q else []
        probes = ["goodput"] + (["quality"] if benches else [])
        if levels is None and self.sweep:
            levels = SWEEP_LEVELS

        n = len(self.trials) + 1
        self.log(f"  [{self.method}] {n:2d}. {label}")
        try:
            t = self.ev.measure(config, probes=probes, benchmarks=benches,
                                node_id=label, levels=levels)
        except Exception as e:
            self.log(f"        FAILED: {type(e).__name__}: {str(e)[:100]}")
            t = Trial(node_id=label, config=dict(config), goodput=0.0,
                      ttft_p99_ms=float("inf"), itl_p99_ms=float("inf"),
                      memory_gb=0.0, slo_ok=False,
                      diagnostics={"launch_error": f"{type(e).__name__}: {e}"})

        t.provenance = dict(self.stamp)
        d = t.diagnostics or {}
        if t.goodput:
            self.log(f"        {t.goodput:8.1f} tok/s  L={t.concurrency}  "
                     f"ttft {t.ttft_p99_ms:5.0f}ms  itl {t.itl_p99_ms:5.1f}ms  "
                     f"slo {d.get('slo_attainment', 0):.0%}"
                     + (f"  math_500 {t.quality.get('math_500'):.4f}"
                        if t.quality.get("math_500") is not None else ""))
        self.trials.append(t)
        with open(self.journal, "a") as fh:
            fh.write(json.dumps(t.__dict__, default=str) + "\n")
            fh.flush()
        return t

    # --------------------------------------------------------------- finish

    def finish(self, *, chosen: Trial | None, extra: dict | None = None) -> dict:
        """Write result.json. `chosen` is the config this method would SHIP.

        Recorded explicitly rather than inferred as max(goodput), because the
        methods disagree about what shipping means: the walk ships its
        incumbent, which is not always the best trial it saw, and yolo ships
        whichever of its two cells read higher.
        """
        ok = [t for t in self.trials if t.goodput]
        best = max(ok, key=lambda t: t.goodput) if ok else None
        res = {
            "method": self.method,
            "provenance": self.stamp,
            "launches": len(self.trials),
            "failed_launches": sum(1 for t in self.trials if not t.goodput),
            "minutes": (time.time() - self.t0) / 60,
            "chosen": (chosen.__dict__ if chosen else None),
            "best_seen": (best.__dict__ if best else None),
            "trials": [t.__dict__ for t in self.trials],
            **(extra or {}),
        }
        (self.run_dir / "result.json").write_text(
            json.dumps(res, indent=2, default=str))
        self.log(f"\n  [{self.method}] {len(self.trials)} launches, "
                 f"{res['minutes']:.0f} min, "
                 f"{res['failed_launches']} failed")
        if chosen:
            self.log(f"  [{self.method}] ships {chosen.node_id}: "
                     f"{chosen.goodput:.1f} tok/s at L={chosen.concurrency}")
        if best and chosen and best.goodput > (chosen.goodput or 0) * 1.001:
            # Worth saying out loud: a method can measure something better than
            # what it ships, and that gap is a property of the method.
            self.log(f"  [{self.method}] NOTE: best trial seen was "
                     f"{best.node_id} at {best.goodput:.1f}, above what it ships")
        return res


def setup(model: str, trace: str, ttft_p99: float, itl_p99: float,
          qps: float | None = None):
    """Fingerprint + SLO + Context, built exactly as run.py builds them.

    Shared so a method cannot accidentally search a different space than the
    one the walk searched -- which would make every comparison meaningless.
    """
    from fingerprint import Context
    from request import InferOptRequest, build_fingerprint
    fp, slo = build_fingerprint(InferOptRequest(
        model=model, trace=trace, ttft_p99_ms=ttft_p99, itl_p99_ms=itl_p99,
        **({"qps": qps} if qps else {})))
    return fp, slo, Context(fingerprint=fp, slo=slo)
