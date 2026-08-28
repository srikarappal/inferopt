"""Render a finished run: the Pareto frontier as a table and a scatter plot.

    python plot.py runs/fourth/result.json

Terminal output only. The frontier is a trade-off surface, and a table alone
hides the shape of it: two configs three tokens/sec apart can sit 150ms apart on
latency, which is the whole reason both are on the frontier rather than one
dominating the other.

Axes are goodput (the objective, maximised) against TTFT p99 (minimised), the
two that actually trade against each other on this workload. Quality and memory
are in the table because on the lossless branch they barely move; when the lossy
branch runs they need their own view.

The BASELINE is plotted with everything else, deliberately. A frontier drawn
without it looks like a cluster of similar configs; drawn with it, the thing
that actually happened is obvious.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

W, H = 68, 19          # plot box, characters


def load(path: str) -> tuple[dict | None, list[dict], list[dict]]:
    r = json.loads(Path(path).read_text())
    base = r.get("baseline")
    if base is None:
        # Pre-dates the baseline fix. The seed trial may still be in the journal.
        j = Path(path).parent / "trials.jsonl"
        if j.exists():
            rows = [json.loads(l) for l in j.read_text().splitlines() if l.strip()]
            base = next((x for x in rows if x.get("node_id") == "stage_1_3"), None)
    return base, r["trials"], r["frontier"]


def table(base, trials, frontier) -> None:
    fkeys = {(t["node_id"], round(t["goodput"], 1)) for t in frontier}
    print(f"\n  {'':2s}{'node':28s} {'goodput':>9s} {'vs base':>9s} {'thru':>8s} "
          f"{'ttft p99':>9s} {'slo':>5s} {'quality':>8s}")
    print("  " + "-" * 84)

    rows = []
    if base:
        rows.append(("BASELINE", base, True))
    for t in trials:
        rows.append((t["node_id"], t, False))

    for name, t, is_base in rows:
        d = t.get("diagnostics") or {}
        g = t["goodput"]
        rel = "  baseline" if is_base else (
            f"{(g/base['goodput'] - 1)*100:+8.1f}%" if base and base["goodput"] else "       --")
        q = t.get("quality") or {}
        qs = f"{min(q.values()):.4f}" if q else "      --"
        mark = "* " if (t["node_id"], round(g, 1)) in fkeys else "  "
        if is_base:
            mark = "> "
        print(f"  {mark}{name:28s} {g:9.1f} {rel:>9s} "
              f"{d.get('throughput', float('nan')):8.1f} {t['ttft_p99_ms']:8.0f}ms "
              f"{d.get('slo_attainment', 0):5.0%} {qs:>8s}")
    print(f"\n  * on the Pareto frontier   > the baseline every % is against")


def scatter(base, trials, frontier) -> None:
    """Goodput (x, higher better) against TTFT p99 (y, lower better).

    Log-ish y is avoided on purpose: the baseline sits ~4x above everything else
    in latency and that gap is the result. Compressing it would hide it.
    """
    pts = [(t["goodput"], t["ttft_p99_ms"], t["node_id"], False) for t in trials]
    if base:
        pts.append((base["goodput"], base["ttft_p99_ms"], "BASELINE", True))
    fkeys = {(t["node_id"], round(t["goodput"], 1)) for t in frontier}

    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    x1 += max(1e-9, (x1 - x0) * 0.08); x0 -= max(1e-9, (x1 - x0) * 0.05)
    y1 += max(1e-9, (y1 - y0) * 0.08); y0 -= max(1e-9, (y1 - y0) * 0.05)

    grid = [[" "] * W for _ in range(H)]
    def cell(x, y):
        cx = int((x - x0) / (x1 - x0) * (W - 1))
        cy = int((1 - (y - y0) / (y1 - y0)) * (H - 1))
        return max(0, min(H - 1, cy)), max(0, min(W - 1, cx))

    for gx, gy, name, is_base in pts:
        r, c = cell(gx, gy)
        on_f = (name, round(gx, 1)) in fkeys
        grid[r][c] = "B" if is_base else ("#" if on_f else "o")

    print(f"\n  ttft p99 (ms)   lower is better ^")
    for i, row in enumerate(grid):
        yv = y1 - (y1 - y0) * i / (H - 1)
        axis = f"{yv:7.0f} |" if i % 3 == 0 else f"{'':7s} |"
        print(f"  {axis}{''.join(row)}")
    print(f"  {'':7s} +{'-' * W}")
    ticks = "".join(f"{x0 + (x1-x0)*k/4:<{W//4}.0f}" for k in range(5))[:W]
    print(f"  {'':8s}{ticks}")
    print(f"  {'':8s}goodput (tok/s)   higher is better ->")
    print(f"\n    B  baseline      #  on the frontier      o  measured, dominated")


def quality_scatter(base, trials, frontier) -> None:
    """Goodput against accuracy -- the axes a quality-first reader cares about.

    Latency is not the only thing traded away. Someone deploying a reasoning
    workload will give up throughput for accuracy without hesitating, and the
    goodput/TTFT view says nothing about that. Every point carries a quality
    score for this reason: measured on the lossy nodes, inherited from the
    baseline on the lossless ones (marked ~), where it cannot have changed.

    On a lossless-only run this plot is a horizontal line, and that is the
    correct picture: nothing gave up any accuracy. It becomes informative when
    the lossy branch runs and the line starts to bend.
    """
    pts = []
    for t in trials:
        q = t.get("quality") or {}
        if q:
            pts.append((t["goodput"], min(q.values()), t["node_id"],
                        t.get("quality_inherited", False), False))
    if base and (base.get("quality") or {}):
        bq = min(base["quality"].values())
        pts.append((base["goodput"], bq, "BASELINE", False, True))
    if not pts:
        print("\n  no quality scores recorded -- nothing to plot on the accuracy axis")
        return

    fkeys = {(t["node_id"], round(t["goodput"], 1)) for t in frontier}
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    x0, x1 = min(xs), max(xs); y0, y1 = min(ys), max(ys)
    if x1 - x0 < 1e-9: x1 = x0 + 1
    if y1 - y0 < 1e-9: y0, y1 = y0 - 0.01, y1 + 0.01
    pad = lambda a, b: (a - (b - a) * 0.08, b + (b - a) * 0.08)
    x0, x1 = pad(x0, x1); y0, y1 = pad(y0, y1)

    grid = [[" "] * W for _ in range(H)]
    for gx, gy, name, inherited, is_base in pts:
        c = max(0, min(W - 1, int((gx - x0) / (x1 - x0) * (W - 1))))
        r = max(0, min(H - 1, int((1 - (gy - y0) / (y1 - y0)) * (H - 1))))
        ch = "B" if is_base else ("#" if (name, round(gx, 1)) in fkeys else "o")
        grid[r][c] = ch

    print(f"\n  accuracy (worst benchmark)   higher is better ^")
    for i, row in enumerate(grid):
        yv = y1 - (y1 - y0) * i / (H - 1)
        axis = f"{yv:7.3f} |" if i % 3 == 0 else f"{'':7s} |"
        print(f"  {axis}{''.join(row)}")
    print(f"  {'':7s} +{'-' * W}")
    ticks = "".join(f"{x0 + (x1-x0)*k/4:<{W//4}.0f}" for k in range(5))[:W]
    print(f"  {'':8s}{ticks}")
    print(f"  {'':8s}goodput (tok/s)   higher is better ->")
    print(f"\n    B baseline   # frontier   o dominated"
          f"   (~ in the table = inherited, not re-measured)")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__.splitlines()[2].strip()); return 1
    base, trials, frontier = load(sys.argv[1])
    if base is None:
        print("\n  BASELINE NOT RECORDED in this run -- 'vs base' will be blank.\n"
              "  Runs after the baseline fix carry it in result.json.")
    table(base, trials, frontier)
    print(f"\n{'='*74}\n  LATENCY VIEW -- for a reader who cares about TTFT")
    scatter(base, trials, frontier)
    print(f"\n{'='*74}\n  ACCURACY VIEW -- for a reader who cares about quality")
    quality_scatter(base, trials, frontier)
    if base and base["goodput"]:
        best = max(trials, key=lambda t: t["goodput"])
        print(f"\n  best measured  {best['goodput']:.1f} tok/s at {best['node_id']}"
              f"  ({best['goodput']/base['goodput']:.2f}x baseline)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
