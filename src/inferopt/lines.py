"""The walk's milestone lines: what a person watching a run reads, and what
Vertical Inference draws as a job's Flow tab.

Its console parses exactly these (console/job_flow.py) and its agent forwards
them unthrottled (agent/job_runner.py milestone_line). Its tests build their
lines with these functions, so changing a wording here fails there instead
of silently dropping a column from every job's diagram.
"""


def variants_word(count):
    return f"{count} variant{'' if count == 1 else 's'}"


def incumbent(goodput, source):
    """The baseline the walk measures every node against."""
    return f"incumbent   {goodput:.1f} goodput  [{source}]"


def predicted(goodput, chosen):
    """A predicted shape measured beside stock at the baseline, and whether the
    walk starts from it."""
    where = "the walk starts here" if chosen else "stock stays the start"
    return f"predicted   {goodput:.1f} goodput  [{where}]"


def start(node_id, count):
    """A node about to be measured: said before the first launch."""
    return f"  start {node_id:32s} {variants_word(count)}"


def skip(node_id, why):
    return f"  skip  {node_id:32s} {why[:70]}"


def verdict(keep, node_id, goodput, delta, count):
    """A measured node kept or reverted, its best variant against the baseline."""
    return (f"  {'KEEP' if keep else 'revert':>5} {node_id:32s} "
            f"{goodput:8.1f} goodput  {delta:+7.1%}  ({variants_word(count)})")


def refused(node_id, why):
    """A node reverted because no variant passed its gates."""
    return f"  revert {node_id:32s} no variant passed: {why}"


def stop(reason):
    return f"  STOP  {reason}"


def finalist_peak(node_id, goodput, concurrency, met, levels):
    """A finalist's dense re-measure: its peak and how many loads met the target."""
    return (f"  stage 2.1  {node_id}: peak {goodput:.1f} at L={concurrency}, "
            f"{met} of {levels} levels served within the target")


def resumed(count, extra=""):
    """A run picking up a journal: what replays instead of relaunching."""
    return f"  resume    {count} measurements already on disk will be replayed, not relaunched{extra}"
