"""Three views of where a deployment's serving capacity goes.

    python -m inferopt.plot_flow runs/<dir> --out-dir docs/figs
    python -m inferopt.plot_flow runs/<a> runs/<b> --multiples

  sankey     four columns, links tinted by source -- the flow view
  waterfall  the attribution: baseline, each measured gain, the remainder
  multiples  the same composition across models, one panel each

WHY THREE. A single source and a flat list of sinks is part-to-whole, and the
honest form for that is a stacked bar, not a Sankey -- putting curves on it
produces a stacked bar with curves, which is what a first attempt at this looked
like. A Sankey earns the form only once flows CROSS between stages, which needs a
real middle column. The waterfall exists because the data was produced as an
ordered sequence of measured deltas, one launch each, and that provenance is
worth showing directly.

WORDING. "Delivered too late to count" was jargon dressed as a label: it means
tokens that were produced, paid for, and then missed the latency target, so they
score nothing. The nodes say that in words a reader who has never seen a goodput
definition can follow.
"""
from __future__ import annotations

import argparse
import json
import math
import textwrap
from pathlib import Path

INK = "#0b0b0b"
MUTED = "#44443f"
FAINT = "#77766f"
SURFACE = "#ffffff"
RULE = "#e4e3de"

# Node colours. Validated as a categorical set against the light surface
# (tools/validate_palette.py): worst CVD delta E 16.4 vs the target of 8, worst
# normal-vision 19.3 vs the floor of 15, all inside the 0.43-0.77 lightness band
# and above the 0.10 chroma floor. Links inherit their SOURCE node's colour at
# 0.4 alpha, which is the plotly energy chart's signature and the reason the
# picture reads as flow rather than as stacked blocks.
C_SOURCE = "#9a97b5"
C_USEFUL = "#8483e6"
C_LATE = "#ee9799"
C_LOSS = ["#e88f91", "#d67a7e", "#c4666c", "#b2525a", "#a03e48"]
C_WASTE = "#bf5e63"

LABELS = {
    "prefix_caching": "Prompt prefixes recomputed",
    "max_model_len_rightsize": "KV cache reserved but never touched",
    "graph_capture": "Per-step kernel launch overhead",
    "chunked_prefill": "Long prefills blocking decode",
    "kv_cache_fp8": "KV stored wider than it needs to be",
    "spec_decode_ngram": "One token per step, no speculation",
    "spec_decode_depth": "Speculation too shallow",
    "lossless_complete": "Remaining lossless headroom",
}
LATE = "Produced, then missed the deadline"


# ---------------------------------------------------------------- data
def decompose(run_dir: Path) -> dict:
    trials = [json.loads(l) for l in (run_dir / "trials.jsonl").read_text().splitlines() if l.strip()]
    scored = [t for t in trials if t.get("node_id") and t.get("goodput") is not None]
    if not scored:
        raise SystemExit(f"{run_dir}: no scored trials")
    seed = next((t for t in scored if t["node_id"] == "stage_1_3"), scored[0])
    best = max(scored, key=lambda t: t["goodput"])

    gains, prev, last = [], seed["goodput"], None
    for t in trials:
        if t.get("node_id") and t.get("goodput") is not None:
            last = t
        if t.get("kept") and t.get("goodput") is not None:
            d = t["goodput"] - prev
            if d > 0 and last is not None:
                gains.append((LABELS.get(last["node_id"],
                                         last["node_id"].replace("_", " ").capitalize()), d))
                prev = t["goodput"]

    def thr(t):
        v = ((t.get("diagnostics") or {}).get("throughput"))
        if v is None:
            raise SystemExit(f"{t['node_id']}: no throughput")
        return v

    # THE SLO FILTER IS APPLIED ONCE, NOT TWICE. goodput ALREADY counts only
    # tokens from requests that met the target; multiplying it by
    # slo_attainment filters the same thing again. It produced a "too slow"
    # band of 52% at the tuned config when the measured figure is 28%, and it
    # inverted the trend -- the chart showed late delivery getting WORSE with
    # tuning (41% -> 52%) when it actually improves (53% -> 28%).
    #
    #     tokens produced   = throughput
    #     inside the target = goodput
    #     too slow          = throughput - goodput

    cap = best["goodput"]
    resid = cap - seed["goodput"] - sum(v for _, v in gains)
    if resid > cap * 0.005:
        gains.append(("Other causes not isolated", resid))
    gains.sort(key=lambda g: -g[1])
    if len(gains) > 5:
        gains = gains[:4] + [(f"Other ({len(gains) - 4} smaller causes)",
                              sum(v for _, v in gains[4:]))]
    return {
        "capacity": cap,
        "b_goodput": seed["goodput"],
        "b_thr": thr(seed),
        "a_thr": thr(best),
        "b_useful": seed["goodput"],
        "b_late": thr(seed) - seed["goodput"],
        "a_useful": best["goodput"],
        "a_late": thr(best) - best["goodput"],
        "gains": gains,
        "diag_b": seed.get("diagnostics") or {},
        "diag_a": best.get("diagnostics") or {},
    }


# ---------------------------------------------------------------- sankey
def _ribbon(ax, x0, x1, ya0, ya1, yb0, yb1, color):
    import numpy as np
    t = np.linspace(0, 1, 160)
    e = 3 * t ** 2 - 2 * t ** 3
    ax.fill_between(x0 + (x1 - x0) * t, yb0 + (yb1 - yb0) * e, ya0 + (ya1 - ya0) * e,
                    color=color, alpha=0.40, linewidth=0, zorder=1)


def sankey(d: dict, out: Path, title: str, subtitle: str) -> Path:
    """Four columns: capacity -> realised? -> what happened -> useful or wasted.

    The fourth column funnels everything back to two nodes, which is what makes
    the reference energy chart legible: however many intermediate paths there
    are, the eye lands on one comparison at the end.

    'Never realised' bypasses the middle columns. Those tokens do not exist --
    they have no phase and no outcome -- so any attempt to route them through a
    stage would be inventing structure. The reference chart does the same with
    flows that skip electricity generation.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    cap = d["capacity"]
    never = sum(v for _, v in d["gains"])
    fig = plt.figure(figsize=(14.5, 7.8), dpi=230)
    fig.patch.set_facecolor(SURFACE)
    ax = fig.add_axes([0.035, 0.055, 0.93, 0.72])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    X = [0.035, 0.30, 0.60, 0.925]
    W, PAD = 0.011, 0.022

    def col(items, x):
        """Stack a column, return {key: (y_top, height, colour)}."""
        tot = sum(v for _, v, _ in items)
        span = 1.0 - PAD * (len(items) - 1)
        out, y = {}, 1.0
        for key, val, c in items:
            h = val / cap * span * (cap / tot if tot > cap else 1.0)
            h = val / cap * span
            out[key] = (y, h, c)
            ax.add_patch(Rectangle((x, y - h), W, h, facecolor=c, edgecolor="black",
                                   linewidth=0.5, zorder=4))
            y -= h + PAD
        return out

    c1 = col([("cap", cap, C_SOURCE)], X[0])
    c2 = col([("real", d["b_useful"] + d["b_late"], C_USEFUL),
              ("never", never, C_WASTE)], X[1])
    mid = [("useful", d["b_useful"], C_USEFUL), ("late", d["b_late"], C_LATE)]
    mid += [(f"g{i}", v, C_LOSS[min(i, len(C_LOSS) - 1)]) for i, (_, v) in enumerate(d["gains"])]
    c3 = col(mid, X[2])
    c4 = col([("good", d["b_useful"], C_USEFUL),
              ("bad", cap - d["b_useful"], C_WASTE)], X[3])

    def link(a, akey, b, bkey, val, cur, colour):
        ya, ha, _ = a[akey]; yb, hb, _ = b[bkey]
        fa, fb = val / cap, val / cap
        y0 = ya - cur[akey]; y1 = yb - cur[bkey]
        _ribbon(ax, X[cur["xa"]] + W, X[cur["xb"]], y0, y1, y0 - fa * cur["sa"],
                y1 - fb * cur["sb"], colour)
        cur[akey] += fa * cur["sa"]; cur[bkey] += fb * cur["sb"]

    s2 = 1.0 - PAD * 1
    s3 = 1.0 - PAD * (len(mid) - 1)
    s4 = 1.0 - PAD * 1

    cur = {"cap": 0.0, "real": 0.0, "never": 0.0, "xa": 0, "xb": 1, "sa": 1.0, "sb": s2}
    link(c1, "cap", c2, "real", d["b_useful"] + d["b_late"], cur, C_SOURCE)
    link(c1, "cap", c2, "never", never, cur, C_SOURCE)

    cur = {**{k: 0.0 for k in list(c2) + list(c3)}, "xa": 1, "xb": 2, "sa": s2, "sb": s3}
    link(c2, "real", c3, "useful", d["b_useful"], cur, C_USEFUL)
    link(c2, "real", c3, "late", d["b_late"], cur, C_USEFUL)
    for i, (_, v) in enumerate(d["gains"]):
        link(c2, "never", c3, f"g{i}", v, cur, C_WASTE)

    cur = {**{k: 0.0 for k in list(c3) + list(c4)}, "xa": 2, "xb": 3, "sa": s3, "sb": s4}
    link(c3, "useful", c4, "good", d["b_useful"], cur, C_USEFUL)
    link(c3, "late", c4, "bad", d["b_late"], cur, C_LATE)
    for i, (_, v) in enumerate(d["gains"]):
        link(c3, f"g{i}", c4, "bad", v, cur, C_LOSS[min(i, len(C_LOSS) - 1)])

    def lab(nodes, key, text, x, side="right", bold=False, val=None):
        y, h, _ = nodes[key]
        tx = x + W + 0.008 if side == "right" else x - 0.008
        ax.text(tx, y - h / 2, text, fontsize=9.2, color=INK if bold else MUTED,
                va="center", ha="left" if side == "right" else "right",
                fontweight="bold" if bold else "normal")
        if val is not None:
            ax.text(tx, y - h / 2 - 0.036, f"{val:,.0f} tok/s   {val / cap:.0%}",
                    fontsize=8.4, color=FAINT,
                    ha="left" if side == "right" else "right", va="center")

    lab(c1, "cap", "Serving capacity", X[0], "left", True, cap)
    lab(c2, "real", "Tokens the server produced", X[1], "right", False,
        d["b_useful"] + d["b_late"])
    lab(c2, "never", "Throughput never reached", X[1], "right", False, never)
    lab(c3, "useful", "Answered inside the latency target", X[2], "right", True, d["b_useful"])
    lab(c3, "late", LATE, X[2], "right", False, d["b_late"])
    for i, (name, v) in enumerate(d["gains"]):
        lab(c3, f"g{i}", name, X[2], "right", False, v)
    lab(c4, "good", "Useful work", X[3], "left", True, d["b_useful"])
    lab(c4, "bad", "Wasted capacity", X[3], "left", False, cap - d["b_useful"])

    fig.text(0.035, 0.945, title, fontsize=19.5, fontweight="bold", color=INK)
    fig.text(0.035, 0.905, subtitle, fontsize=10.6, color=MUTED, va="top")
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------- waterfall
def waterfall(d: dict, out: Path, title: str, subtitle: str) -> Path:
    """Baseline, each measured gain, the remainder. The provenance view.

    This is the form that matches how the numbers were made: one launch per step,
    each bar a measured delta against the incumbent before it. A stacked bar can
    show the composition but not the sequence, and the sequence is the argument.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    # ONE QUANTITY THROUGHOUT. The first version started the run at
    # useful-work-delivered and then added THROUGHPUT deltas to it, so the steps
    # summed to 176 while the final bar read 95 -- two different measures of the
    # same axis, and the chart contradicted itself in plain sight. The bars are
    # now all throughput reached, which sums to capacity by construction, and
    # only the final column splits into what lands inside the target and what
    # does not.
    cap = d["capacity"]
    steps = [("As deployed,\nat defaults", d["b_goodput"], "base")]
    for name, v in sorted(d["gains"], key=lambda g: -g[1]):
        steps.append((textwrap.fill(name, 18), v, "gain"))
    tuned = d["a_useful"]
    steps.append(("After\nreconfiguration", tuned, "total"))

    fig = plt.figure(figsize=(13.2, 7.0), dpi=230)
    fig.patch.set_facecolor(SURFACE)
    ax = fig.add_axes([0.075, 0.17, 0.90, 0.60])
    top = cap * 1.06
    ax.set_ylim(0, top); ax.set_xlim(-0.7, len(steps) - 0.3)
    for sp in ("top", "right", "bottom", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=0, labelsize=9.2, colors=MUTED)
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([s[0] for s in steps], fontsize=9.0, color=MUTED)
    for g in ax.get_yticks():
        ax.axhline(g, color=RULE, lw=0.8, zorder=0)
    # The bars are THROUGHPUT reached, not throughput delivered on time -- only
    # the last column splits those. Labelling the axis with the narrower quantity
    # is how the earlier version came to contradict itself.
    ax.set_ylabel("tok/s of throughput the server sustains", fontsize=9.4, color=MUTED)

    run = 0.0
    for i, (name, val, kind) in enumerate(steps):
        if kind == "base":
            ax.add_patch(Rectangle((i - 0.34, 0), 0.68, val, facecolor=C_USEFUL,
                                   edgecolor="none", zorder=3))
            ax.text(i, val + top * 0.02, f"{val:,.0f}", ha="center", fontsize=9.2,
                    color=INK, fontweight="bold")
            run = val
        elif kind == "gain":
            ax.add_patch(Rectangle((i - 0.34, run), 0.68, val,
                                   facecolor=C_LOSS[min(i - 1, len(C_LOSS) - 1)],
                                   edgecolor="none", zorder=3))
            ax.plot([i - 0.34 - 0.32, i - 0.34], [run, run], color=FAINT, lw=0.8,
                    zorder=2)
            ax.text(i, run + val + top * 0.02, f"+{val:,.0f}", ha="center",
                    fontsize=9.0, color=MUTED)
            run += val
        else:
            # The run lands on capacity by construction; this column splits it
            # into what actually reaches a user in time and what does not.
            ax.add_patch(Rectangle((i - 0.34, 0), 0.68, val, facecolor=C_USEFUL,
                                   edgecolor="none", zorder=3))
            ax.add_patch(Rectangle((i - 0.34, val), 0.68, cap - val,
                                   facecolor=C_LATE, alpha=0.55, edgecolor="none",
                                   zorder=3))
            ax.text(i, cap + top * 0.02, f"{cap:,.0f}", ha="center", fontsize=9.2,
                    color=INK, fontweight="bold")
            ax.text(i, val / 2, f"{val:,.0f}\ninside\nthe target", ha="center",
                    va="center", fontsize=8.6, color="#ffffff", fontweight="bold")
            if cap - val > cap * 0.06:
                ax.text(i, val + (cap - val) / 2,
                        f"{cap - val:,.0f}\ntoo slow\n{(cap - val) / cap:.0%}",
                        ha="center", va="center", fontsize=8.4, color=MUTED)

    fig.text(0.035, 0.945, title, fontsize=19.5, fontweight="bold", color=INK)
    fig.text(0.035, 0.905, subtitle, fontsize=10.6, color=MUTED, va="top")
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------- multiples
def multiples(runs: list, out: Path, names: list) -> Path:
    """Same composition, one panel per model, shared percentage axis.

    Model size is a FACET, not a stage: a token does not flow through "14B", and
    two models' capacities are not the same quantity, so they cannot share a
    Sankey trunk. Faceting is what makes the trend the headline.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    ds = [decompose(Path(r)) for r in runs]
    # Panel geometry in ABSOLUTE figure fractions with the figure grown per
    # panel, not a height divided by the panel count -- the latter put the first
    # panel's top at 0.92 and ran it straight through the subtitle.
    n = len(ds)
    fig = plt.figure(figsize=(12.4, 3.0 + 2.0 * n), dpi=230)
    fig.patch.set_facecolor(SURFACE)
    TOP, H, STEP = 0.80, 0.17, 0.27
    for k, (d, nm) in enumerate(zip(ds, names)):
        cap = d["capacity"]
        ax = fig.add_axes([0.10, TOP - H - k * STEP, 0.72, H])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        for row, (lbl, useful, late) in enumerate(
                (("after", d["a_useful"], d["a_late"]), ("before", d["b_useful"], d["b_late"]))):
            y = 0.08 + row * 0.48
            x = 0.0
            for val, c in ((useful, C_USEFUL), (late, C_LATE), (cap - useful - late, C_WASTE)):
                w = val / cap
                if w <= 0:
                    continue
                ax.add_patch(Rectangle((x, y), w, 0.34, facecolor=c,
                                       edgecolor=SURFACE, linewidth=1.4, zorder=3))
                x += w
            ax.text(-0.012, y + 0.17, "tuned" if lbl == "after" else "defaults",
                    ha="right", va="center", fontsize=9.0, color=MUTED)
            ax.text(1.012, y + 0.17, f"{useful / cap:.0%}", ha="left", va="center",
                    fontsize=10.4, color=INK, fontweight="bold")
        ax.text(0.0, 1.02, nm, fontsize=11.2, fontweight="bold", color=INK, va="bottom")
        ax.text(1.012, 1.02, f"{cap:,.0f} tok/s capacity", fontsize=8.8, color=FAINT,
                va="bottom", ha="left")

    fig.text(0.035, 0.945, "Bigger models waste more of what they could deliver",
             fontsize=18, fontweight="bold", color=INK)
    fig.text(0.035, 0.895,
             "Share of demonstrated serving capacity answered inside the latency"
             " target. Periwinkle is useful work; rose is capacity produced too\n"
             "slowly to count or never produced at all. Same workload and same"
             " target throughout.",
             fontsize=10.0, color=MUTED, va="top")
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("plot_flow")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--names", default="")
    ap.add_argument("--multiples", action="store_true")
    a = ap.parse_args(argv)
    outd = Path(a.out_dir); outd.mkdir(parents=True, exist_ok=True)
    names = a.names.split(",") if a.names else [Path(r).name for r in a.runs]

    if a.multiples:
        p = multiples(a.runs, outd / "capacity-multiples.png", names)
        print(f"  wrote {p}")
        return 0
    for r, nm in zip(a.runs, names):
        d = decompose(Path(r))
        tag = nm.lower().replace(" ", "-")
        sub = (f"Of {d['capacity']:,.0f} tok/s this accelerator was measured to sustain,"
               f" {d['b_useful'] / d['capacity']:.0%} reaches a user inside the latency"
               f" target at stock settings.\nEverything else is capacity the deployment"
               f" paid for and did not get.")
        print("  wrote", sankey(d, outd / f"capacity-sankey-{tag}.png",
                                "Where serving capacity goes", sub))
        sub2 = (f"Each bar is a separate measured launch: the configuration change, and"
                f" what it recovered.\nThe last column is the same hardware after every"
                f" change that survived its own re-measurement.")
        print("  wrote", waterfall(d, outd / f"capacity-waterfall-{tag}.png",
                                   "What the tuning actually recovered", sub2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
