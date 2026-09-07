"""Draw a Plackett-Burman run: the design matrix, the response, the effects.

    python plot_pb.py runs/rerun-1.7b-pb
    python plot_pb.py runs/rerun-1.7b-pb --response ttft_p99_ms -o /tmp/x.png

A screen's output is a table of numbers whose STRUCTURE is the argument, and the
table hides it. The design is what makes the arithmetic valid -- every factor on
in half the runs, every pair balanced -- and reading twelve rows of plus and
minus signs does not show that. Drawn, it is immediate: the column for a factor
that helps is visibly hotter than its blanks.

Reads effects.json and needs no GPU, so it can be run against a finished screen
at any time. Separate from pb_screen.py on purpose -- a plotting bug should not
be able to take down a run that costs three GPU-hours.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(prog="plot_pb", description=__doc__.split("\n\n")[0])
    ap.add_argument("run_dir")
    ap.add_argument("--response", default="goodput",
                    help="which measured response to colour and rank by. The "
                         "screen records several: goodput, ttft_p99_ms, "
                         "itl_p99_ms, slo_attainment, concurrency, memory_gb "
                         "and the accuracy benchmark.")
    ap.add_argument("-o", "--out", default=None)
    a = ap.parse_args()

    d = Path(a.run_dir)
    e = json.loads((d / "effects.json").read_text())
    design = e["design"]
    ids = e.get("factor_ids") or [f["id"] for f in e["factors"]]
    resp = (e.get("responses") or {}).get(a.response)
    if resp is None:
        resp = e.get("results") if a.response == "goodput" else None
    if resp is None:
        print(f"  {a.response!r} not recorded. Have: "
              f"{', '.join(sorted(e.get('responses') or {}))}")
        return 1
    eff = ((e.get("effects_by_response") or {}).get(a.response)) or e["factors"]
    lenth = e.get("lenth") or {}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    n, k = len(design), len(ids)
    ok = [v for v in resp if v is not None]
    if not ok:
        print("  every row failed; nothing to draw")
        return 1
    # Colour scale spans only the SUCCESSFUL rows. Including failures as 0 would
    # compress every real difference into the top of the ramp.
    norm = Normalize(vmin=min(ok), vmax=max(ok))
    cmap = plt.cm.viridis
    lower_better = a.response in ("ttft_p99_ms", "itl_p99_ms")

    fig, (axg, axe) = plt.subplots(
        1, 2, figsize=(13.5, 0.52 * n + 2.4),
        gridspec_kw={"width_ratios": [2.0, 1.0]})

    # ---- design matrix. ON cells carry the row's response, off cells are
    # blank: the eye should follow the factor that is on, not the whole grid.
    for r in range(n):
        v = resp[r]
        for c in range(k):
            on = design[r][c]
            axg.add_patch(plt.Rectangle(
                (c, n - 1 - r), 1, 1,
                facecolor=(cmap(norm(v)) if (on and v is not None)
                           else ("0.93" if not on else "0.75")),
                edgecolor="white", lw=1.6))
            if on:
                axg.text(c + 0.5, n - 1 - r + 0.5, "on", ha="center",
                         va="center", fontsize=7,
                         color="white" if v is not None and norm(v) < 0.6 else "0.15")
        axg.text(k + 0.15, n - 1 - r + 0.5,
                 f"{v:,.0f}" if v is not None else "FAILED",
                 va="center", fontsize=8.5,
                 color="0.15" if v is not None else "firebrick",
                 fontweight="bold" if v is not None else "normal")
    axg.set_xlim(0, k + 1.4); axg.set_ylim(0, n)
    axg.set_xticks([c + 0.5 for c in range(k)])
    axg.set_xticklabels([i.replace("_", "\n") for i in ids], fontsize=7.5)
    axg.set_yticks([n - 1 - r + 0.5 for r in range(n)])
    axg.set_yticklabels([f"run {r+1}" for r in range(n)], fontsize=8)
    axg.set_title(f"design matrix -- each factor ON in {n//2} of {n} runs, "
                  f"every PAIR balanced\ncell colour = that run's {a.response}",
                  fontsize=9.5)
    for s in axg.spines.values():
        s.set_visible(False)
    axg.tick_params(length=0)

    # ---- effects
    vals = [(x["id"], x.get("effect")) for x in eff if x.get("effect") is not None]
    vals.sort(key=lambda t: t[1])
    ypos = range(len(vals))
    me = lenth.get("me") if a.response == "goodput" else None
    colours = ["firebrick" if v < 0 else "seagreen" for _, v in vals]
    if me:
        # Grey out anything Lenth cannot separate from noise, so the plot does
        # not assert a direction the data will not support.
        colours = [c if abs(v) > me else "0.72" for c, (_, v) in zip(colours, vals)]
    axe.barh(list(ypos), [v for _, v in vals], color=colours, height=0.62)
    axe.set_yticks(list(ypos))
    axe.set_yticklabels([i for i, _ in vals], fontsize=8)
    axe.axvline(0, color="0.3", lw=1)
    if me:
        for x in (me, -me):
            axe.axvline(x, color="0.45", ls="--", lw=1)
        axe.text(me, len(vals) - 0.4, "  Lenth margin", fontsize=7, color="0.45")
    axe.set_xlabel(f"effect on {a.response}  "
                   f"(mean with factor ON minus mean with it off)", fontsize=8.5)
    axe.set_title("effects" + ("  -- grey = inside the noise" if me else ""),
                  fontsize=9.5)
    axe.grid(axis="x", alpha=0.25, lw=0.6)
    if lower_better:
        axe.invert_xaxis()
        axe.set_xlabel(axe.get_xlabel() + "   (left is better)", fontsize=8.5)

    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axg,
                 fraction=0.03, pad=0.13).set_label(a.response, fontsize=8)
    fig.suptitle(f"Plackett-Burman screen -- {d.name}", fontsize=11)
    note = ("Effects are confounded with two-way interactions (resolution III). "
            "A large bar says 'measure this in stage 2', not 'this factor causes it'.")
    if any(v is None for v in resp):
        note += f"  {sum(1 for v in resp if v is None)} run(s) failed and are "
        note += "excluded from every mean."
    fig.text(0.5, 0.005, note, ha="center", fontsize=7.5, color="0.35")
    fig.tight_layout(rect=(0, 0.035, 1, 0.955))

    out = a.out or str(d / f"pb-{a.response}.png")
    fig.savefig(out, dpi=150)
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
