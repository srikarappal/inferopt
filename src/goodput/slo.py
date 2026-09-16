"""The two thresholds goodput is defined against.

A caller's own SLO type usually carries more than this -- inferopt's also holds
quality budgets and an attainment floor -- but only these two decide whether a
request counted. So this is a Protocol, not a base class: pass your own object
straight in and keep your own type.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class LatencyTarget(Protocol):
    """Anything with these two fields can price goodput."""
    ttft_p99_ms: float | None
    itl_p99_ms: float | None


@dataclass(frozen=True)
class Latency:
    """A standalone target, for callers with no SLO type of their own.

    Either field may be None, meaning that dimension is not constrained. Both
    None means every successful request counts, which makes goodput equal
    throughput -- occasionally what you want, and worth being explicit about.
    """
    ttft_p99_ms: float | None = None
    itl_p99_ms: float | None = None
