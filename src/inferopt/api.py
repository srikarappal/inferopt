"""optimize() -- the one call docs/api-design.md is written around.

    from inferopt import optimize, SLO

    result = optimize(
        model="Qwen/Qwen3-14B",
        trace="data/trace_shared.jsonl",
        slo=SLO(ttft_p99_ms=500, itl_p99_ms=250),
        qps=16,
        strategy="sequential",          # or "screen" or "yolo"
    )
    print(result.chosen, result.replicas(qps=16))

THE CALLER PICKS THE STRATEGY, and there is no auto-selection. It is tempting
and there is no evidence to base it on: on Qwen3-1.7B screening shipped 2333.6
tok/s against the walk's 2067.9, and on Qwen3-14B the three landed within 4% of
each other with yolo the best value by a wide margin. Which method wins is a
property of the space, and nothing here can read the space before searching it.

BASELINES ARE MEASURED, NOT ASSUMED. Stock and the aiconfigurator prediction are
run and reported alongside the search's answer rather than folded into it. That
is a direct consequence of a bug: run.py seeded the walk from aiconfigurator
while yolo and screen seeded from the conservative default, so the walk had a
head start no one could see -- the provenance stamps matched, because the seed
config was not in them. A prediction belongs in the comparison as a row, not
inside one competitor's starting point.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from inferopt._paths import default_dag, runs as _runs


@dataclass
class Result:
    """What comes back. Every number here was measured, or is absent."""

    model: str
    strategy: str
    trials: list[Any] = field(default_factory=list)
    chosen: Any = None
    best_seen: Any = None
    frontier: list[Any] = field(default_factory=list)
    baselines: dict[str, Any] = field(default_factory=dict)
    quality_changes: list[Any] = field(default_factory=list)
    launches: int = 0
    minutes: float = 0.0
    run_dir: str | None = None
    provenance: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)

    slo: Any = None
    """The SLO this was searched against, so eligibility can be reported."""

    def ships_within_slo(self, trial: Any = None) -> bool | None:
        """Whether the shipped config met the attainment floor, if one was set.

        None when no floor was set -- which is not the same as True, and the
        summary says so. Goodput already prices SLO misses in, so maximising it
        can ship a config at 76% attainment without comment; that is a defensible
        answer to "how many on-time tokens per second", and a surprising one to
        anybody who read the SLO as a promise.
        """
        t = trial if trial is not None else self.chosen
        if self.slo is None or getattr(self.slo, "min_slo_attainment", None) is None:
            return None
        att = ((getattr(t, "diagnostics", None) or {}).get("slo_attainment"))
        return self.slo.attainment_ok(att)

    @property
    def regressions(self) -> list[Any]:
        """Quality movements that are real AND in the wrong direction.

        Reported, never acted on. A config that costs accuracy still belongs on
        the frontier: "less accurate, far faster" is a trade someone may want,
        and dropping it removes the choice rather than making it.
        """
        return [c for c in self.quality_changes if c.is_regression]

    def replicas(self, qps: float, *, config: Any = None) -> int | None:
        """How many replicas serve `qps` at this config's measured goodput.

        `config` defaults to what the search shipped; pass a baseline's trial to
        ask the same question of stock. The demand is qps x mean output tokens,
        so it needs the workload -- absent that, this returns None rather than
        guessing.
        """
        import math
        t = config if config is not None else self.chosen
        gp = getattr(t, "goodput", None)
        mean_out = (self.provenance.get("workload") or {}).get("mean_output_tokens")
        if not gp or not mean_out:
            return None
        return math.ceil((qps * mean_out) / gp)

    def save(self, path: str | Path | None = None) -> Path:
        p = Path(path) if path else Path(self.run_dir or ".") / "result.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "model": self.model, "strategy": self.strategy,
            "provenance": self.provenance,
            "launches": self.launches, "minutes": self.minutes,
            "chosen": _d(self.chosen), "best_seen": _d(self.best_seen),
            "baselines": {k: _d(v) for k, v in self.baselines.items()},
            "frontier": [_d(t) for t in self.frontier],
            "trials": [_d(t) for t in self.trials],
            "quality_changes": [vars(c) for c in self.quality_changes],
            **self.extra,
        }, indent=2, default=str))
        return p

    def summary(self) -> str:
        lines = [f"  model      {self.model}",
                 f"  strategy   {self.strategy}  "
                 f"({self.launches} launches, {self.minutes:.0f} min)"]
        for name, t in self.baselines.items():
            lines.append(f"  baseline   {name:16s} {_gp(t)}")
        att = ((getattr(self.chosen, "diagnostics", None) or {}).get("slo_attainment"))
        lines.append(f"  ships      {_gp(self.chosen)}"
                     + (f"  at {att:.0%} SLO attainment" if att is not None else ""))
        ok = self.ships_within_slo()
        if ok is False:
            lines.append(f"  WARNING    the shipped config MISSES the attainment floor "
                         f"({self.slo.min_slo_attainment:.0%}); goodput counts only "
                         f"conforming requests, so a partial miss still maximises it")
        elif ok is None and att is not None and att < 0.95:
            lines.append(f"  NOTE       no attainment floor was set, and this config "
                         f"meets the latency targets for {att:.0%} of requests")
        if self.best_seen is not None and self.chosen is not None:
            b, c = getattr(self.best_seen, "goodput", 0), getattr(self.chosen, "goodput", 0)
            if b > (c or 0) * 1.001:
                lines.append(f"  NOTE       best MEASURED was {b:.1f}, above what it ships")
        for r in self.regressions:
            lines.append(f"  REGRESSION {r}")
        return "\n".join(lines)


def _d(t) -> dict | None:
    return None if t is None else (t if isinstance(t, dict) else dict(vars(t)))


def _gp(t) -> str:
    g = getattr(t, "goodput", None)
    if not g:
        return "-"
    L = getattr(t, "concurrency", None)
    return f"{g:8.1f} tok/s" + (f" at L={L}" if L else "")


def optimize(
    *,
    model: str,
    trace: str,
    slo=None,
    ttft_p99_ms: float | None = None,
    itl_p99_ms: float | None = None,
    qps: float | None = None,
    strategy: Literal["sequential", "screen", "yolo"] = "sequential",
    benchmarks: list[str] | None = None,
    quality_every_config: bool = True,
    lossless_only: bool = False,
    allow_loss: float | None = None,
    dag: str | None = None,
    seed_from_run: str | None = None,
    run_dir: str | None = None,
    gpu: str = "0",
    port: int = 8100,
    repeats: int = 1,
    survivors: int = 3,
    log=print,
) -> Result:
    """Search serving configurations and return measured operating points."""
    from inferopt.evaluator import hardware_defaults
    from inferopt.fingerprint import Context
    from inferopt.methods import MethodRunner
    from inferopt.request import InferOptRequest, build_fingerprint
    from inferopt.run import seed_config
    from inferopt.strategies import STRATEGIES

    if slo is not None:
        ttft_p99_ms = ttft_p99_ms or slo.ttft_p99_ms
        itl_p99_ms = itl_p99_ms or slo.itl_p99_ms
    fp, slo_ = build_fingerprint(InferOptRequest(
        model=model, trace=trace, ttft_p99_ms=ttft_p99_ms, itl_p99_ms=itl_p99_ms,
        allow_loss=allow_loss, **({"qps": qps} if qps else {})))
    ctx = Context(fingerprint=fp, slo=slo_)

    rd = run_dir or str(_runs(f"{model.split('/')[-1].lower()}-{strategy}"))
    bench = benchmarks if benchmarks is not None else ["math_500"]
    runner = MethodRunner(strategy, fp, slo_, trace, rd, gpu=gpu, port=port,
                          benchmarks=bench, quality_every=quality_every_config,
                          log=log)

    # The seed. Identical across strategies unless a previous run is continued,
    # which only the chaining strategy can use.
    seed = seed_config(fp)
    if seed_from_run:
        prev = json.loads((Path(seed_from_run) / "result.json").read_text())
        inc = (prev.get("incumbent") or prev.get("chosen") or {})
        inc = inc.get("config") or inc
        if inc:
            seed = {**inc, **{k: v for k, v in hardware_defaults(fp).items()
                              if k not in inc}}
            log(f"  seeding from {seed_from_run}")

    dag_json = json.loads(Path(dag or default_dag()).read_text())
    cls = STRATEGIES[strategy]
    if strategy == "sequential":
        strat = cls(dag_json, lossless_only=lossless_only,
                    force_benchmarks=bench if quality_every_config else None)
    else:
        from inferopt.pb_screen import factors_from_dag
        factors = factors_from_dag(dag_json, ctx)
        if not factors:
            raise ValueError(
                f"no applicable lossless factors for this workload, so "
                f"strategy={strategy!r} has nothing to vary. The sequential "
                f"walk would still run its lossy branch.")
        strat = (cls(factors, repeats=repeats) if strategy == "yolo"
                 else cls(factors, repeats=repeats, survivors=survivors))

    t0 = time.time()
    out = strat.search(ctx, runner, seed, log=log)
    res = Result(
        model=fp.model.id, strategy=strategy, trials=out.trials, slo=slo_,
        chosen=out.chosen, best_seen=out.best_seen, frontier=out.frontier,
        launches=out.launches or len(out.trials),
        minutes=out.minutes or (time.time() - t0) / 60,
        run_dir=rd, provenance={**runner.stamp,
                                "workload": fp.workload.model_dump()},
        extra=out.extra,
    )
    res.quality_changes = _changes(res, bench)
    res.save()
    return res


def _changes(res: Result, benchmarks: list[str]) -> list[Any]:
    """Every measured movement on a quality axis, against the first trial.

    Against the FIRST trial rather than the best: the question a reader asks is
    "did optimizing cost accuracy", and the answer is relative to where the
    search started, not to whichever config happened to score highest.
    """
    from inferopt.api_types import QualityChange
    from inferopt.quality import resolution
    out = []
    for b in benchmarks:
        base = next((t for t in res.trials
                     if (getattr(t, "quality", None) or {}).get(b) is not None), None)
        if base is None:
            continue
        before = base.quality[b]
        try:
            tol = resolution(b)
        except Exception:
            tol = 0.0
        for t in res.trials:
            q = (getattr(t, "quality", None) or {}).get(b)
            if q is None or t is base or getattr(t, "quality_inherited", False):
                continue
            c = QualityChange(benchmark=b, metric="exact_match", before=before,
                              after=q, resolution=tol,
                              node=str(getattr(t, "node_id", "")))
            if not c.within_noise:
                out.append(c)
    return out
