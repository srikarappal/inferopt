"""inferopt -- find a serving configuration for a model, and prove it.

    from inferopt import optimize, SLO, Benchmark

Given a model, a workload trace and an SLO, search the space of serving
configurations and return a Pareto frontier of measured operating points --
goodput against accuracy, latency and memory -- rather than a single number.

Three search strategies share one measurement contract, so their results are
comparable rather than merely each being a number:

    sequential  a greedy walk over the technique DAG, chaining an incumbent
    screen      Plackett-Burman screening, then a factorial over survivors
    yolo        everything on, against everything off

The public surface is re-exported here so callers never import a private
module path; everything else is an implementation detail and may move.
"""

__version__ = "0.1.0"

from inferopt.api import Result, optimize
from inferopt.api_types import (
    Metric,
    QualityChange,
    Sample,
    Verdict,
)
from inferopt.fingerprint import SLO
from inferopt.quality import Benchmark

__all__ = [
    "Benchmark",
    "Result",
    "Metric",
    "QualityChange",
    "SLO",
    "Sample",
    "Verdict",
    "__version__",
    "optimize",
]
