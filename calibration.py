"""Accumulating measurements that make the next run cheaper than the last.

    band = STORE.accept_band(fp)          # measured if we have seen this pair
    STORE.record_run(fp, accept_band=..., quality_tolerance=...)

Three things belong here, and they share one property: they are expensive to
measure, stable for a given (model, hardware) pair, and useless to re-derive on
every run.

  accept_band         how much a goodput improvement must exceed to be real
  quality_tolerance   how much a benchmark moves between identical configs
  predictor_error     stage 1.2's prediction vs stage 1.3's measurement

The defaults are deliberately LOOSE. A band that is too wide loses marginal
wins; a band that is too narrow accepts noise, and every false accept becomes
the parent of the next generation. Losing a 3% win is recoverable. Building six
nodes on top of a phantom one is not.

HISTORY -- why every threshold here is measured

  accept_band was guessed at 2%, and measurement said 3.8%. Two launches x three
  repeats put worst-case across-launch throughput spread at 1.91%, doubled for
  the band. The guessed value would have rejected real wins as noise. Note also
  that across-launch spread was 5x within-launch on one slice: within-launch
  variance alone badly understates what the keep/revert gate actually faces.

  Whole-output equivalence has a 19% false-positive floor. Measured per-token
  flip rate under greedy decoding is 0.44%, so comparing long outputs almost
  always finds a difference. Comparing the first 11 tokens gives 3.5%.

  accept_band is FROZEN for the duration of a run and evolves only across runs.
  A band that moved mid-traversal would leave early and late keep/revert
  decisions resting on different criteria, with a single frontier built from
  both.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from fingerprint import Fingerprint

STORE_PATH = Path(__file__).parent / "calibration.json"

# Used until a real measurement exists for this (model, hardware) pair.
DEFAULT_ACCEPT_BAND = 0.05

# Per-benchmark quality tolerance, as an absolute score delta. Scaled to how
# much room each has to diverge: a short extractive answer flips far less often
# than a long chain of thought. Measured values replace these on first contact.
DEFAULT_QUALITY_TOLERANCE = {
    "humaneval_plus": 0.03,      # ~150 tokens, and one bad token fails a test
    "mbpp_plus": 0.03,           # same shape: pass/fail on execution, no partial credit
    "math_500": 0.04,            # long CoT, most chances to derail
}


class Calibration(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_id: str
    gpu_name: str
    accept_band: float | None = Field(None, description="measured; 2 x worst across-launch throughput spread")
    quality_tolerance: dict[str, float] = Field(default_factory=dict, description="per benchmark, absolute score delta")
    per_token_flip_rate: float | None = Field(None,
        description="probability a greedy token differs between identical runs. Sets the "
                    "equivalence probe's comparison length: compare first K tokens where "
                    "1-(1-p)^K stays under the false-positive budget.")
    predictor_error: dict[str, float] = Field(default_factory=dict,
        description="stage 1.2 predicted vs stage 1.3 measured, per metric. Accumulates into "
                    "the evidence for calibrating a predictor to this hardware.")
    n_runs: int = 0
    note: str | None = None

    def equivalence_prefix_tokens(self, false_positive_budget: float = 0.05) -> int:
        """Longest prefix that keeps false positives under budget at this flip rate.

        Comparing whole outputs is what made the probe useless: at p=0.44%, a
        48-token comparison has a 19% false-positive floor.
        """
        p = self.per_token_flip_rate
        if not p:
            return 8
        k = 1
        while 1 - (1 - p) ** (k + 1) < false_positive_budget and k < 512:
            k += 1
        return k


class CalibrationStore:
    def __init__(self, path: Path = STORE_PATH):
        self.path = path
        self._data: dict[str, Calibration] = {}
        if path.exists():
            raw = json.loads(path.read_text())
            self._data = {k: Calibration(**v) for k, v in raw.items()}

    @staticmethod
    def key(fp: Fingerprint) -> str:
        return f"{fp.model.id}@{fp.hw.gpu_name}x{fp.hw.gpu_count}"

    def get(self, fp: Fingerprint) -> Calibration | None:
        return self._data.get(self.key(fp))

    def accept_band(self, fp: Fingerprint) -> tuple[float, str]:
        """Returns (band, provenance). Provenance goes in the run report so a
        result is never silently resting on a default."""
        c = self.get(fp)
        if c and c.accept_band is not None:
            return c.accept_band, f"measured over {c.n_runs} run(s)"
        return DEFAULT_ACCEPT_BAND, "default (this model/hardware pair has never been measured)"

    def quality_tolerance(self, fp: Fingerprint, benchmark: str) -> tuple[float, str]:
        c = self.get(fp)
        if c and benchmark in c.quality_tolerance:
            return c.quality_tolerance[benchmark], "measured"
        return DEFAULT_QUALITY_TOLERANCE.get(benchmark, 0.03), "default"

    def record_run(self, fp: Fingerprint, **fields) -> Calibration:
        """Merge a run's observations. Tolerances widen but never narrow: a
        single quiet run is not evidence that the noise went away."""
        k = self.key(fp)
        cur = self._data.get(k) or Calibration(model_id=fp.model.id, gpu_name=fp.hw.gpu_name)
        for name in ("accept_band", "per_token_flip_rate"):
            new = fields.get(name)
            if new is not None:
                old = getattr(cur, name)
                setattr(cur, name, new if old is None else max(old, new))
        for bench, tol in (fields.get("quality_tolerance") or {}).items():
            cur.quality_tolerance[bench] = max(cur.quality_tolerance.get(bench, 0.0), tol)
        cur.predictor_error.update(fields.get("predictor_error") or {})
        cur.n_runs += 1
        self._data[k] = cur
        self.path.write_text(json.dumps(
            {kk: vv.model_dump() for kk, vv in self._data.items()}, indent=2) + "\n")
        return cur


STORE = CalibrationStore()
