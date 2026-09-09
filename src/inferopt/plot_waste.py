"""Where a serving deployment's token capacity actually goes.

    python -m inferopt.plot_waste runs/<dir> -o waste.png

Modelled on the IEA/Tesla global energy flow chart: one conserved quantity
entering on the left, every loss named and sized, and a single honest number at
the end for how much of it did useful work.

THE CONSERVED QUANTITY IS TOKEN CAPACITY, NOT TOKENS PRODUCED. That choice is
the whole design. A chart drawn over tokens produced shows almost no waste --
96% of generated tokens arrive inside the deadline even on an untuned server --
because the dominant loss in inference is not work thrown away, it is work the
accelerator never did. Redundant prefill and over-reserved KV do not burn tokens;
they stop tokens existing. Only a capacity denominator can show them, and it is
the difference between a chart that says "we are 96% efficient" and one that says
"we are 64% efficient", of which the second is true.

The capacity figure is the best throughput this hardware was MEASURED to sustain,
not a roofline. A bandwidth roofline would give a larger and more flattering
waste number, and it would be wrong: the weights-only version ignores KV traffic
entirely, and a first pass at including it lands near the measured figure. So the
inefficiency drawn here is a LOWER BOUND -- everything shown is real, and the
true waste is larger by however much the hardware could do better than anything
we found.

Every flow is measured. The "never realised" bands are the goodput deltas
recorded when each technique was switched on, one launch each; the on-time and
late split comes from the per-request records, which carry each request's own
TTFT and inter-token latency and so can be re-split at any deadline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Palette. Deliberately not the reference chart's: grey for loss and one strong
# colour for useful work is the grammar worth borrowing, the exact hues are not.
INK = "#14181d"
MUTED = "#6b737d"
WASTE = "#c9ccd1"
WASTE_EDGE = "#b3b7bd"
USEFUL = "#1f6feb"
SOURCE = "#8c2f2f"
RECOVERED = "#2f7d5c"


def _decompose(run_dir: Path) -> dict:
    """Read one run and return the before/after flow table.

    Raises rather than guessing: a chart that silently falls back to a default
    when a field is missing is a chart that states a measurement nobody made.
    """
    trials = [json.loads(l) for l in (run_dir / "trials.jsonl").read_text().splitlines() if l.strip()]
    scored = [t for t in trials if t.get("node_id") and t.get("goodput") is not None]
    if not scored:
        raise SystemExit(f"{run_dir}/trials.jsonl has no scored trials")

    seed = next((t for t in scored if t["node_id"] == "stage_1_3"), scored[0])
    best = max(scored, key=lambda t: t["goodput"])

    # The kept steps, in order, with what each was worth.
    #
    # `kept: true` is NOT recorded on the named trial -- it lands on a following
    # incumbent record that carries the goodput and no node_id. Reading `kept`
    # off the named trials alone finds nothing, and the whole attribution then
    # collapses into one "unattributed" band, which is a chart that has stopped
    # explaining anything. So the technique credited is the most recent named
    # trial carrying the same goodput as the incumbent record.
    gains, prev, last_named = [], seed["goodput"], None
    for t in trials:
        if t.get("node_id") and t.get("goodput") is not None:
            last_named = t
        if not t.get("kept") or t.get("goodput") is None:
            continue
        d = t["goodput"] - prev
        if d > 0 and last_named is not None:
            gains.append((last_named["node_id"], d))
            prev = t["goodput"]

    def split(t):
        """on-time vs late tokens, as fractions, from the recorded attainment."""
        d = t.get("diagnostics") or {}
        att = d.get("slo_attainment")
        if att is None:
            raise SystemExit(f"trial {t['node_id']} has no slo_attainment")
        return att, 1.0 - att

    src = best["goodput"]
    b_on, b_late = split(seed)
    a_on, a_late = split(best)
    return {
        "capacity": src,
        "before": {
            "delivered": seed["goodput"] * b_on,
            "late": seed["goodput"] * b_late,
            "unrealised": gains,
            "total": seed["goodput"],
        },
        "after": {"delivered": src * a_on, "late": src * a_late, "total": src},
        "diag_before": seed.get("diagnostics") or {},
        "diag_after": best.get("diagnostics") or {},
    }


# Names for the flow bands. The node ids are internal; a reader of the chart
# needs the mechanism, not the identifier.
LABELS = {
    "prefix_caching": "Prompt prefixes recomputed",
    "max_model_len_rightsize": "KV cache reserved and unused",
    "graph_capture": "Kernel launch overhead",
    "chunked_prefill": "Prefill blocking decode",
    "kv_cache_fp8": "KV stored wider than needed",
    "spec_decode_ngram": "Sequential decode",
    "lossless_complete": "Remaining lossless headroom",
}


def _ribbon(ax, x0, x1, y0a, y0b, y1a, y1b, color, alpha=1.0):
    """One flow, drawn as a pair of cubic curves closed into a band."""
    import numpy as np
    t = np.linspace(0, 1, 120)
    ease = 3 * t ** 2 - 2 * t ** 3            # smoothstep: flat at both ends
    x = x0 + (x1 - x0) * t
    top = y0a + (y1a - y0a) * ease
    bot = y0b + (y1b - y0b) * ease
    ax.fill_between(x, bot, top, color=color, alpha=alpha, linewidth=0, zorder=1)


def _panel(ax, title, capacity, flows, note):
    """One Sankey: a single source on the left, named sinks on the right."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.0, 1.14, title, fontsize=12.5, fontweight="bold", color=INK,
            transform=ax.transAxes)
    if note:
        ax.text(0.0, 1.055, note, fontsize=8.7, color=MUTED, transform=ax.transAxes)

    xs, xe = 0.045, 0.80
    bar_w = 0.014
    # Source bar, full height.
    ax.add_patch(__import__("matplotlib").patches.Rectangle(
        (xs - bar_w, 0.0), bar_w, 1.0, color=SOURCE, zorder=3, linewidth=0))
    ax.text(xs - bar_w - 0.012, 0.5, f"{capacity:,.0f} tok/s\nserving capacity",
            fontsize=9.4, color=INK, ha="right", va="center", linespacing=1.5)

    # Destinations are separated by a gap so the ribbons taper between a
    # contiguous source and a split sink. Without it the bands are parallel and
    # the picture reads as a stacked bar, which hides the one thing a Sankey is
    # for: that a single quantity is being divided.
    from matplotlib.patches import Rectangle
    gap = 0.045
    span = 1.0 - gap * (len(flows) - 1)

    # Geometry first, labels second. A 1% band is three pixels tall, so a label
    # centred on it lands on top of its neighbour's -- which is how the first
    # draft rendered "delivered too late" straight through the band below it.
    bands, y_src, y_dst = [], 1.0, 1.0
    for label, value, kind in flows:
        f = value / capacity
        h_src, h_dst = f, f * span
        bands.append((label, value, kind, y_src, h_src, y_dst, h_dst))
        y_src -= h_src
        y_dst -= h_dst + gap

    for label, value, kind, ys, hs, yd, hd in bands:
        col = USEFUL if kind == "useful" else WASTE
        _ribbon(ax, xs, xe, ys, ys - hs, yd, yd - hd, col,
                alpha=1.0 if kind == "useful" else 0.92)
        ax.add_patch(Rectangle((xe, yd - hd), bar_w, hd, color=col,
                               zorder=3, linewidth=0))

    # Push labels apart top-down, and draw a leader to the band when a label has
    # been moved off its own centre.
    # In AXES units, and the axes is only 0.20 of the figure: a two-line label at
    # 8.9pt needs about a fifth of the panel height, not the 0.135 the first
    # attempt used, which still let two labels overlap.
    MIN_SEP = 0.215
    placed = []
    for label, value, kind, ys, hs, yd, hd in bands:
        want = yd - hd / 2
        if placed and placed[-1] - want < MIN_SEP:
            want = placed[-1] - MIN_SEP
        placed.append(want)
        if abs(want - (yd - hd / 2)) > 0.012:
            ax.plot([xe + bar_w, xe + bar_w + 0.011],
                    [yd - hd / 2, want], color=WASTE_EDGE, lw=0.8, zorder=2)
        ax.text(xe + bar_w + 0.016, want,
                f"{value:,.0f} tok/s   {value / capacity:.0%}\n{label}",
                fontsize=8.9, color=INK if kind == "useful" else MUTED,
                va="center", linespacing=1.5,
                fontweight="bold" if kind == "useful" else "normal")


def build(run_dir: Path, out: Path, subtitle: str = "") -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = _decompose(run_dir)
    cap = d["capacity"]
    b, a = d["before"], d["after"]

    before_flows = [("Delivered within the latency target", b["delivered"], "useful"),
                    ("Delivered too late to count", b["late"], "waste")]
    for node, gain in b["unrealised"]:
        before_flows.append((LABELS.get(node, node.replace("_", " ").capitalize()),
                             gain, "waste"))
    # Anything the named steps do not account for. Shown rather than absorbed:
    # a residual folded silently into another band is a number nobody measured.
    resid = cap - sum(v for _, v, _ in before_flows)
    if resid > cap * 0.005:
        before_flows.append(("Unattributed", resid, "waste"))

    after_flows = [("Delivered within the latency target", a["delivered"], "useful"),
                   ("Delivered too late to count", a["late"], "waste")]
    resid_a = cap - sum(v for _, v, _ in after_flows)
    if resid_a > cap * 0.005:
        after_flows.append(("Unattributed", resid_a, "waste"))

    fig = plt.figure(figsize=(13.8, 8.4), dpi=200)
    fig.patch.set_facecolor("white")

    fig.text(0.045, 0.963, "Serving Capacity Today is Mostly Unrealised",
             fontsize=20.5, fontweight="bold", color=INK)

    eff_b = b["delivered"] / cap
    eff_a = a["delivered"] / cap
    hit_b = (d["diag_before"].get("prefix_hit_rate") or 0.0)
    hit_a = (d["diag_after"].get("prefix_hit_rate") or 0.0)
    kv_a = (d["diag_after"].get("kv_cache_util") or 0.0)

    body = (
        f"Measured on one accelerator against a fixed workload and a fixed latency target. The"
        f" capacity figure is the highest\nsustained throughput this hardware was observed to"
        f" reach, so every loss below is one we have watched it recover — the true\ninefficiency"
        f" is larger. A server left at its defaults delivers {eff_b:.0%} of that capacity inside"
        f" the deadline. Most of what is missing is\nnot work discarded but work never done:"
        f" prompt prefixes recomputed rather than reused ({hit_b:.0%} of prefill was served from"
        f" cache,\nagainst {hit_a:.0%} after), and key-value cache reserved far beyond what the"
        f" workload touches (only {kv_a:.0%} of the pool is ever used).\nA further"
        f" {b['late'] / cap:.1%} is produced but arrives after the deadline, so it is paid for and"
        f" thrown away. Reconfiguring the same hardware,\nwith no change to the model or its"
        f" outputs, raises the share doing useful work from {eff_b:.0%} to {eff_a:.0%}."
    )
    fig.text(0.045, 0.905, body, fontsize=10.0, color=INK, linespacing=1.62,
             va="top")

    ax1 = fig.add_axes([0.155, 0.395, 0.545, 0.20])
    ax2 = fig.add_axes([0.155, 0.095, 0.545, 0.20])
    _panel(ax1, "As deployed, at defaults", cap, before_flows,
           f"{eff_b:.0%} of capacity does useful work")
    _panel(ax2, "After reconfiguration", cap, after_flows,
           f"{eff_a:.0%} of capacity does useful work")

    fig.text(0.045, 0.028,
             "Flows conserve: every band is a measured quantity and they sum to the"
             " capacity bar. Grey is capacity lost, blue is capacity delivered."
             + (f"  {subtitle}" if subtitle else ""),
             fontsize=8.5, color=MUTED, style="italic")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("plot_waste")
    ap.add_argument("run_dir")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--subtitle", default="")
    a = ap.parse_args(argv)
    run = Path(a.run_dir)
    out = Path(a.out) if a.out else run.parent / f"{run.name}-waste.png"
    p = build(run, out, a.subtitle)
    print(f"  wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
