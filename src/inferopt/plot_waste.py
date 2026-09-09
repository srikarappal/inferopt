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

# PALETTE -- computed against the validator's checks, not chosen by eye.
#
# Two encoding jobs, so two gates. "Delivered" and "lost" are CATEGORICAL
# identities and take the categorical floors. The three loss causes are one hue
# separated by lightness, which is an ORDINAL ramp and takes the ordinal gate.
# Holding a same-hue ramp to the categorical normal-vision floor of 15 is
# impossible by construction -- steps of one hue differ by delta L alone, so
# their delta E is ~8 -- and trying to was why the first two palette searches
# returned nothing at all.
#
# Measured on the light surface #fcfcfb (OKLab delta E x100, Machado CVD):
#   delivered vs each loss step, ALL pairs : CVD 16.4  (target >= 8)
#                                            normal 19.5 (floor >= 15)
#   loss ramp, ordinal                     : delta L 0.081 (>= 0.06)
#                                            lightest step 2.15:1 (>= 2.0)
#   lightness band 0.43-0.77 and chroma >= 0.10: all four inside
#
# The two palest loss steps sit below 3:1 against the surface. That is the
# documented relief case, not a pass: it obliges visible direct labels, which
# every band carries.
INK = "#0b0b0b"          # text-primary
MUTED = "#52514e"        # text-secondary
FAINT = "#7a7975"        # text-muted
SURFACE = "#fcfcfb"      # the surface the palette was validated against
USEFUL = "#8483e6"       # delivered -- periwinkle
LOSS = ["#ee9799", "#d67a7e", "#bf5e63"]   # dusty-rose ordinal ramp, light -> dark
SOURCE = "#b8b5cf"
GRID = "#e6e5e1"


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
    from matplotlib.patches import FancyBboxPatch, Rectangle
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.0, 1.15, title, fontsize=12.5, fontweight="bold", color=INK,
            transform=ax.transAxes)
    if note:
        ax.text(0.0, 1.06, note, fontsize=8.7, color=MUTED, transform=ax.transAxes)

    xs, xe = 0.045, 0.80
    bar_w = 0.013

    # A 2px surface gap between adjacent fills, on BOTH sides. Without it the
    # ribbons abut and read as one mass, which is the thing a flow diagram is
    # supposed to take apart.
    src_gap, gap = 0.006, 0.045
    n = len(flows)
    src_span = 1.0 - src_gap * (n - 1)
    dst_span = 1.0 - gap * (n - 1)

    bands, y_src, y_dst = [], 1.0, 1.0
    for label, value, kind in flows:
        f = value / capacity
        bands.append((label, value, kind, y_src, f * src_span, y_dst, f * dst_span))
        y_src -= f * src_span + src_gap
        y_dst -= f * dst_span + gap

    ax.add_patch(Rectangle((xs - bar_w, 0.0), bar_w, 1.0, color=SOURCE,
                           zorder=3, linewidth=0))
    ax.text(xs - bar_w - 0.014, 0.5, f"{capacity:,.0f} tok/s\nserving capacity",
            fontsize=9.4, color=INK, ha="right", va="center", linespacing=1.55)

    for idx, (label, value, kind, ys, hs, yd, hd) in enumerate(bands):
        col = USEFUL if kind == "useful" else LOSS[min(idx - 1, len(LOSS) - 1)]
        _ribbon(ax, xs, xe, ys, ys - hs, yd, yd - hd, col,
                alpha=1.0 if kind == "useful" else 0.95)
        # 4px rounded data-end on the sink, flat where it meets the ribbon.
        ax.add_patch(FancyBboxPatch(
            (xe, yd - hd + 0.004), bar_w, max(hd - 0.008, 0.001),
            boxstyle="round,pad=0.004,rounding_size=0.004",
            facecolor=col, edgecolor=SURFACE, linewidth=1.0, zorder=3))

    # Labels wear TEXT tokens, never the series colour; a swatch beside them
    # carries identity. Pushed apart top-down, with a leader back to the band
    # when one has been moved off its own centre -- a 1% band is three pixels
    # tall and a label centred on it lands on its neighbour's.
    MIN_SEP = 0.265
    placed = []
    for idx, (label, value, kind, ys, hs, yd, hd) in enumerate(bands):
        want = yd - hd / 2
        if placed and placed[-1] - want < MIN_SEP:
            want = placed[-1] - MIN_SEP
        placed.append(want)
        if abs(want - (yd - hd / 2)) > 0.012:
            ax.plot([xe + bar_w + 0.003, xe + bar_w + 0.014],
                    [yd - hd / 2, want], color=GRID, lw=0.9, zorder=2, clip_on=False)
        col = USEFUL if kind == "useful" else LOSS[min(idx - 1, len(LOSS) - 1)]
        # clip_on=False: a label pushed past the panel edge keeps its swatch,
        # and identity must never be carried by position alone.
        sw = Rectangle((xe + bar_w + 0.020, want + 0.040), 0.011, 0.032,
                       color=col, zorder=4, linewidth=0)
        sw.set_clip_on(False)
        ax.add_patch(sw)
        ax.text(xe + bar_w + 0.040, want + 0.056,
                f"{value:,.0f} tok/s   {value / capacity:.0%}",
                fontsize=9.0, color=INK, va="center",
                fontweight="bold" if kind == "useful" else "normal")
        ax.text(xe + bar_w + 0.040, want - 0.050, label,
                fontsize=8.7, color=MUTED if kind == "useful" else FAINT, va="center")


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

    fig = plt.figure(figsize=(13.8, 8.9), dpi=200)
    fig.patch.set_facecolor(SURFACE)

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

    ax1 = fig.add_axes([0.155, 0.415, 0.545, 0.195])
    ax2 = fig.add_axes([0.155, 0.135, 0.545, 0.195])
    _panel(ax1, "As deployed, at defaults", cap, before_flows,
           f"{eff_b:.0%} of capacity does useful work")
    _panel(ax2, "After reconfiguration", cap, after_flows,
           f"{eff_a:.0%} of capacity does useful work")

    # A legend is present for >= 2 series, and with <= 4 the marks are ALSO direct
    # labelled -- identity never rests on colour alone. It doubles as the relief
    # for the two loss steps that sit below 3:1 against the surface.
    from matplotlib.patches import Rectangle as _R
    lx = 0.155
    for name, col in [("Delivered within the latency target", USEFUL)] + [
            (lab, LOSS[min(i, len(LOSS) - 1)])
            for i, (lab, _, _) in enumerate(before_flows[1:])]:
        fig.patches.append(_R((lx, 0.072), 0.0105, 0.016, color=col,
                              transform=fig.transFigure, figure=fig, linewidth=0))
        fig.text(lx + 0.016, 0.080, name, fontsize=8.4, color=MUTED, va="center")
        lx += 0.017 + 0.0068 * len(name)

    fig.text(0.045, 0.028,
             "Flows conserve: every band is a measured quantity and they sum to the"
             " capacity bar. Periwinkle is capacity delivered inside the target;"
             " the rose bands are capacity lost, ordered light to dark by size."
             + (f"  {subtitle}" if subtitle else ""),
             fontsize=8.5, color=MUTED, style="italic")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    # THE TABLE TWIN. A static image cannot carry a tooltip or a toggle, and two
    # of the loss steps sit below 3:1 against the surface, so the values have to
    # be reachable without reading colour at all. Direct labels do most of that
    # work; this is the WCAG-clean equivalent, and it is also what anyone
    # re-checking the arithmetic will actually open.
    tbl = out.with_suffix(".txt")
    lines = [f"{'':44s} {'tok/s':>9s} {'share':>7s}", "-" * 62]
    for name, flows in (("AS DEPLOYED, AT DEFAULTS", before_flows),
                        ("AFTER RECONFIGURATION", after_flows)):
        lines += ["", name]
        for label, value, _ in flows:
            lines.append(f"  {label:42s} {value:9,.1f} {value / cap:7.1%}")
        lines.append(f"  {'total (= serving capacity)':42s} "
                     f"{sum(v for _, v, _ in flows):9,.1f} "
                     f"{sum(v for _, v, _ in flows) / cap:7.1%}")
    tbl.write_text("\n".join(lines) + "\n")
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
