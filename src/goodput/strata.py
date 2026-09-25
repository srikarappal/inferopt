"""A representative slice of a trace, for the measurement that has to speak
for the whole run before the run is spent.

The seed's window replays the trace in file order. At ten seconds a request
and concurrency one, that is whichever four prompts happen to come first, and
a calibration verdict drawn from four short prompts says nothing about the
long ones. So the calibration draws its requests across the trace's shape:
rows are bucketed by input length and by output length into a small grid,
and the order visits the cells round-robin, so the first handful of requests
already covers short and long of each. Deterministic, and it never invents a
request: every index is a row of the trace.
"""

from __future__ import annotations

BINS = 3


def _edges(values: list[int], bins: int) -> list[int]:
    """Bucket boundaries at the quantiles, deduplicated: a trace whose every
    row has the same length gets one bucket, not three empty ones."""
    ordered = sorted(values)
    edges = []
    for k in range(1, bins):
        edge = ordered[min(len(ordered) - 1, (k * len(ordered)) // bins)]
        if edge not in edges:
            edges.append(edge)
    return edges


def _bucket(value: int, edges: list[int]) -> int:
    return sum(1 for edge in edges if value >= edge)


def cells_of(in_tokens: list[int], out_tokens: list[int], bins: int = BINS) -> dict[tuple[int, int], list[int]]:
    """Row indices by (input bucket, output bucket), each list in trace order."""
    if not in_tokens:
        return {}
    outs = out_tokens if len(out_tokens) == len(in_tokens) else [0] * len(in_tokens)
    in_edges, out_edges = _edges(in_tokens, bins), _edges(outs, bins)
    cells: dict[tuple[int, int], list[int]] = {}
    for index, (i, o) in enumerate(zip(in_tokens, outs)):
        cells.setdefault((_bucket(i, in_edges), _bucket(o, out_edges)), []).append(index)
    return cells


def stratified_order(in_tokens: list[int], out_tokens: list[int], bins: int = BINS) -> list[int]:
    """Every row of the trace, ordered so that each pass over the cells takes
    one row from each: the first N requests cover the grid, the next N cover
    it again, until every row has been used once."""
    cells = cells_of(in_tokens, out_tokens, bins)
    queues = [list(rows) for _, rows in sorted(cells.items())]
    order: list[int] = []
    while any(queues):
        for queue in queues:
            if queue:
                order.append(queue.pop(0))
    return order


def label(cell: tuple[int, int], bins: int = BINS) -> str:
    names = {0: "short", bins - 1: "long"} if bins > 1 else {0: "all"}
    def name(bucket):
        return names.get(bucket, "mid")
    return f"{name(cell[0])} in / {name(cell[1])} out"


def describe(cells: dict[tuple[int, int], list[int]], bins: int = BINS) -> str:
    """One line: how many rows each cell holds."""
    return ", ".join(f"{label(cell, bins)} {len(rows)}" for cell, rows in sorted(cells.items()))
