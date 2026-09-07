"""The user-facing quality types: what a caller writes, and what comes back.

    from inferopt import Benchmark, Metric, Sample, Verdict, LLMJudge

Phase 2 of docs/api-design.md. Deliberately a NEW module rather than an edit to
quality.py: a 14B lossy walk is in flight as this is written and imports
quality lazily -- evaluator.measure does `from quality import run_benchmark` on
every node that scores -- so touching that file would kill a running job at its
next quality probe rather than at startup.

WHY JUDGE AND METRIC ARE SEPARATE LAYERS

They answer different questions and they fail differently.

  judge   per SAMPLE: did this one output satisfy the task? Returns a Verdict
          per row, in row order, because the caller zips them against rows to
          find WHICH ticket regressed.
  metric  over the whole set: what number summarises those verdicts? pass@1 is
          a mean; recall on a subset is not; a latency percentile is neither.

quality.py's Benchmark already carries a `judge` with exactly this contract and
a `metric` as a bare string. The string is the gap: it names an aggregation
without stating its DIRECTION, so nothing downstream can tell whether a rise is
an improvement. Every metric here declares that.

WHY VERDICT IS NOT A BOOL

A bool cannot say why. The existing judges return list[bool], which is enough to
compute a score and useless the moment the score moves and someone asks what
changed -- the question RULER could never answer. A Verdict keeps the reason and
the parsed value alongside the pass/fail, at no measurement cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Sequence


# --------------------------------------------------------------------- sample

@dataclass(frozen=True)
class Sample:
    """One benchmark row and what the model said about it.

    `row` is the dataset record as loaded, untouched, because judges need
    fields that differ per benchmark -- MATH-500 carries `answer`, MBPP+ carries
    `entry_point` and its tests. Normalising it into a common shape here would
    force every judge through a lowest common denominator.
    """
    row: dict
    text: str
    prompt: str = ""
    index: int = 0
    n_output_tokens: int | None = None


@dataclass(frozen=True)
class Verdict:
    """Did this sample pass, and why.

    `ok` is the answer. `reason` is what makes a moved score diagnosable: a
    judge that returns False without saying whether the output was malformed,
    empty, truncated, or simply wrong leaves the next person guessing -- which
    is how a probe reading 0.05 on a 30B model stayed unexplained.

    `value` carries the parsed answer when there is one, so a caller can see
    what the model actually produced without re-parsing the text.
    """
    ok: bool
    reason: str = ""
    value: Any = None

    def __bool__(self) -> bool:          # so existing sum()/mean code still works
        return self.ok


Judge = Callable[[Sequence[Sample]], Sequence[Verdict]]


# --------------------------------------------------------------------- metric

# Built-in aggregations and their direction. A name not in here has no known
# direction, so Metric REQUIRES an explicit one -- see below.
_BUILTIN_METRICS: dict[str, tuple[str, str]] = {
    "pass@1":      ("max", "fraction of samples whose verdict is ok"),
    "exact_match": ("max", "fraction matching the reference exactly"),
    "accuracy":    ("max", "fraction of samples whose verdict is ok"),
    "recall":      ("max", "fraction of relevant samples recovered"),
    "precision":   ("max", "fraction of returned samples that were relevant"),
    "f1":          ("max", "harmonic mean of precision and recall"),
    "wer":         ("min", "word error rate"),
    "error_rate":  ("min", "fraction of samples whose verdict is not ok"),
}


@dataclass(frozen=True)
class Metric:
    """How to turn verdicts into one number, and which way is better.

    DIRECTION IS MANDATORY AND NOT INFERRED. Without it nothing downstream can
    read a movement: the Pareto frontier needs to know whether to maximise or
    minimise the axis, and a regression report needs to know whether -0.02 is a
    loss or a win. `wer` and `pass@1` move in opposite directions and no naming
    convention distinguishes them.

    AN UNKNOWN NAME RAISES rather than defaulting. `Metric("exact-match")` --
    a hyphen where the built-in has an underscore -- would otherwise become a
    silent custom metric with no direction and no function, and report 0.0 for
    every config, which reads as "quality unchanged".
    """
    name: str
    direction: Literal["max", "min"] | None = None
    fn: Callable[[Sequence[Sample], Sequence[Verdict]], float] | None = None
    description: str = ""

    def __post_init__(self):
        known = _BUILTIN_METRICS.get(self.name)
        if self.direction is None:
            if not known:
                raise ValueError(
                    f"Metric({self.name!r}) is not a built-in, so its direction "
                    f"cannot be inferred. Pass direction='max' or 'min'.\n"
                    f"  built-ins: {', '.join(sorted(_BUILTIN_METRICS))}\n"
                    f"  (a typo lands here: 'exact-match' is not 'exact_match')")
            object.__setattr__(self, "direction", known[0])
            if not self.description:
                object.__setattr__(self, "description", known[1])
        if self.fn is None and not known:
            raise ValueError(
                f"Metric({self.name!r}) has no fn and is not a built-in, so "
                f"there is nothing to compute. Pass fn=... .")

    def compute(self, samples: Sequence[Sample], verdicts: Sequence[Verdict]) -> float:
        if self.fn is not None:
            return float(self.fn(samples, verdicts))
        if not verdicts:
            raise ValueError(f"{self.name}: no verdicts to aggregate")
        # bool(v), NOT v.ok. Verdict implements __bool__ precisely so that a
        # judge returning plain bools keeps working -- and all three built-in
        # judges do return bools. Reaching for .ok defeated that and would have
        # raised AttributeError on every real benchmark, in production, after
        # the launch and the generation had already been paid for.
        ok = sum(1 for v in verdicts if bool(v))
        if self.name == "error_rate":
            return 1.0 - ok / len(verdicts)
        return ok / len(verdicts)

    @property
    def higher_is_better(self) -> bool:
        return self.direction == "max"


# ------------------------------------------------------------- quality change

@dataclass(frozen=True)
class QualityChange:
    """A measured movement on one quality axis, between two configs.

    Surfaced rather than acted on. Quality is an AXIS, not a gate: a config that
    costs accuracy still belongs on the frontier, because "less accurate, far
    faster" is a trade someone may want and deleting it removes the choice. So
    this reports; only an explicit require= or abandon_below= acts.

    `within_noise` is the honest qualifier and the reason the type exists at
    all. A delta smaller than the benchmark's own repeat-to-repeat spread is not
    a finding in either direction -- and this project has twice reported one as
    though it were.
    """
    benchmark: str
    metric: str
    before: float
    after: float
    resolution: float = 0.0
    node: str | None = None

    @property
    def delta(self) -> float:
        return self.after - self.before

    @property
    def within_noise(self) -> bool:
        return abs(self.delta) <= self.resolution

    @property
    def is_regression(self) -> bool:
        """Direction-aware, and False inside the noise."""
        if self.within_noise:
            return False
        d = _BUILTIN_METRICS.get(self.metric, ("max", ""))[0]
        return self.delta < 0 if d == "max" else self.delta > 0

    def __str__(self) -> str:
        tag = ("within noise" if self.within_noise
               else "REGRESSION" if self.is_regression else "improvement")
        at = f" at {self.node}" if self.node else ""
        return (f"{self.benchmark}.{self.metric} {self.before:.4f} -> "
                f"{self.after:.4f} ({self.delta:+.4f}, +/-{self.resolution:.4f}) "
                f"{tag}{at}")
