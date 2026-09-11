"""Resume a run from its own trials.jsonl, or refuse to.

    plan = resume.plan(run_dir, stamp)
    if plan.conflict:
        raise SystemExit(plan.reason)
    evaluator.replay = plan.cache          # measure() consults it first

THE DECISION IS NOT "DOES THE FILE EXIST". A run directory holding trials from a
different model, a different trace or a different SLO is not a run to resume, it
is a different experiment wearing the same path. Replaying those would pool
measurements that do not compare, which is the exact failure trial_stamp was
written to prevent: goodput counts only requests that met the SLO, so the same
config against a 500ms target and a 200ms one gives two numbers that are not
comparable and do not average.

So there are three outcomes, not two:

  fresh     no trials.jsonl, or it is empty. Start, truncate, proceed.
  resume    trials present and every stamp matches this run's. Replay the
            matching (node_id, config) pairs, spend launches only on the rest.
  conflict  trials present and a stamp differs. Refuse, name the field that
            differs, and make the caller choose a new directory or pass
            --restart. Today run.py truncates unconditionally, so pointing at an
            occupied directory destroys the record with no warning at all.

LAUNCHES ARE CREDITED, NOT FORGOTTEN. A resumed run reports the replayed trials
in its total, because "pb took 16 launches" has to stay true across an
interruption or the method comparison quietly becomes a comparison of who
crashed less.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

JOURNAL = "trials.jsonl"


def key(node_id: str, config: dict) -> tuple[str, str]:
    """What makes two measurements the same measurement.

    node_id alone is not enough: a sweep measures one node at several values and
    records each under the same id, so the config has to be part of the key or a
    resume would replay the first value in place of all of them.
    """
    return (str(node_id), json.dumps(config or {}, sort_keys=True, default=str))


@dataclass
class Plan:
    mode: str                                   # fresh | resume | conflict
    cache: dict[tuple[str, str], dict] = field(default_factory=dict)
    reason: str = ""
    n_trials: int = 0
    n_duplicates: int = 0

    @property
    def conflict(self) -> bool:
        return self.mode == "conflict"

    @property
    def resuming(self) -> bool:
        return self.mode == "resume"


def _stamp_diff(a: dict, b: dict) -> list[str]:
    """Which stamp fields disagree. Empty means the two runs are the same job."""
    return sorted({k for k in set(a) | set(b) if a.get(k) != b.get(k)})


def plan(run_dir: str | Path, stamp: dict | None) -> Plan:
    """Decide fresh, resume or conflict for this directory and this job."""
    path = Path(run_dir) / JOURNAL
    if not path.exists() or not path.read_text().strip():
        return Plan("fresh")

    rows: list[dict] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue                    # a torn last line is what a kill leaves
        if row.get("node_id"):
            rows.append(row)
    if not rows:
        return Plan("fresh")

    # Every trial carries its own stamp, so a directory that mixes jobs is
    # detectable even if the first trial happens to match.
    if stamp:
        for row in rows:
            got = row.get("provenance") or {}
            if not got:
                continue
            diff = _stamp_diff(got, stamp)
            if diff:
                return Plan("conflict", reason=(
                    f"{path} holds trials from a different job: "
                    f"{', '.join(diff)} differ. Use a new --run-dir, or pass "
                    f"--restart to discard {len(rows)} recorded trials."),
                    n_trials=len(rows))

    cache: dict[tuple[str, str], dict] = {}
    dupes = 0
    for row in rows:
        k = key(row["node_id"], row.get("config") or {})
        if k in cache:
            dupes += 1                  # keep the FIRST, so a replay is deterministic
            continue
        cache[k] = row
    return Plan("resume", cache=cache, n_trials=len(rows), n_duplicates=dupes,
                reason=f"resuming from {len(cache)} recorded measurements in {path}")


def to_trial(row: dict):
    """Rebuild a Trial from a journal row.

    Required fields get defaults rather than raising. A journal written by this
    code always carries them, but a hand-edited file, an older schema or a row
    truncated by a kill should cost one replay, not the whole resume.
    """
    from inferopt.traverse import Trial
    fields = Trial.__dataclass_fields__
    kw = {k: v for k, v in row.items() if k in fields}
    kw.setdefault("node_id", row.get("node_id") or "unknown")
    kw.setdefault("config", row.get("config") or {})
    for name, default in (("goodput", 0.0), ("ttft_p99_ms", float("inf")),
                          ("itl_p99_ms", float("inf")), ("memory_gb", 0.0)):
        if kw.get(name) is None:
            kw[name] = default
        kw.setdefault(name, default)
    return Trial(**kw)


def describe(p: Plan) -> str:
    """One line for the run banner."""
    if p.mode == "fresh":
        return "  resume    nothing to resume, starting fresh"
    if p.conflict:
        return f"  resume    CONFLICT: {p.reason}"
    extra = f", {p.n_duplicates} duplicate rows ignored" if p.n_duplicates else ""
    return (f"  resume    {len(p.cache)} measurements already on disk will be "
            f"replayed, not relaunched{extra}")
