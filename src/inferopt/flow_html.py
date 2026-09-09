"""A self-contained interactive waterfall of what tuning recovered.

    python -m inferopt.flow_html runs/rerun-14b-seqdag --name "14B" -o out.html
    python -m inferopt.flow_html runs/a runs/b --name "14B,Small" -o out.html

BOTH MEASURES, RELATED CORRECTLY. Bar height is throughput -- every token the
server produced. The solid portion of the first and last bars is goodput, the
part that reached a user inside the latency target, and the pale remainder is
what arrived too slowly to be useful.

    produced          = throughput
    inside the target = goodput           <- ALREADY SLO-filtered
    too slow          = throughput - goodput

Multiplying goodput by slo_attainment applies that filter a second time. An
earlier version did, and it reported a "too slow" share of 52% at the tuned
configuration where the measured figure is 28% -- and inverted the trend, showing
late delivery getting worse with tuning when it more than halves.

The numbers are read from trials.jsonl at render time, never typed in. The one
thing this makes visible that the goodput view hid: the walk optimised goodput,
so in throughput terms two of its accepted steps are NEGATIVE. They are drawn
that way. A step that trades throughput for latency is a real decision, and a
chart that quietly drops the down-bars would be arguing rather than reporting.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

# Bright pastel, validated against the light surface with
# tools/validate_palette.py: worst adjacent CVD delta E 9.1 (target >= 8), worst
# normal-vision 18.1 (floor >= 15), all six inside the 0.43-0.77 lightness band
# and above the 0.10 chroma floor. "Bright pastel" is L 0.75 with chroma pushed
# to 0.17 -- as light as the band allows with as much colour as CVD separation
# survives. All six sit below 3:1 against the surface, which is the documented
# relief case: every bar carries a visible value and there is a table view.
PALETTE = ["#a59bff", "#ff7d76", "#00c4f7", "#d8a700", "#33cc7d", "#f37fcf"]

LABELS = {
    "prefix_caching": "Reuse prompt prefixes",
    "max_model_len_rightsize": "Right-size the KV cache",
    "graph_capture": "Capture CUDA graphs",
    "chunked_prefill": "Chunk long prefills",
    "kv_cache_fp8": "Narrow the KV cache",
    "spec_decode_ngram": "Speculate several tokens",
    "spec_decode_depth": "Speculate deeper",
    "lossless_complete": "Remaining lossless headroom",
}


def walk(run_dir: Path) -> dict:
    """Baseline, each accepted step's THROUGHPUT delta, and the final."""
    rows = [json.loads(l) for l in (run_dir / "trials.jsonl").read_text().splitlines() if l.strip()]
    named = [t for t in rows if t.get("node_id") and t.get("goodput") is not None]
    if not named:
        raise SystemExit(f"{run_dir}: no scored trials")
    seed = next((t for t in named if t["node_id"] == "stage_1_3"), named[0])
    best = max(named, key=lambda t: t["goodput"])

    def thr(t):
        return (t.get("diagnostics") or {}).get("throughput")

    base = thr(seed)
    if base is None:
        raise SystemExit(f"{run_dir}: seed has no throughput")

    steps, prev, last = [], base, None
    for t in rows:
        if t.get("node_id") and thr(t) is not None:
            last = t
        if t.get("kept") and last is not None and thr(last) is not None:
            v = thr(last)
            changed = {k: val for k, val in (last.get("config") or {}).items()
                       if k != "model" and (seed.get("config") or {}).get(k) != val}
            steps.append({
                "label": LABELS.get(last["node_id"],
                                    last["node_id"].replace("_", " ").capitalize()),
                "delta": round(v - prev, 1),
                "after": round(v, 1),
                "concurrency": last.get("concurrency"),
                "config": ", ".join(f"{k} = {val}" for k, val in sorted(changed.items())) or "—",
            })
            prev = v

    # `final` and `final_good` must describe the SAME configuration. Taking the
    # walk's last accepted step for one and the best-by-goodput trial for the
    # other put 267.1 tok/s beside a goodput of 197.3 that belongs to a
    # different launch.
    final_thr = thr(best)
    # BOTH measures, related the only way they honestly can be:
    #   produced          = throughput
    #   inside the target = goodput   (already SLO-filtered -- do NOT multiply
    #                                  it by slo_attainment, that applies the
    #                                  same filter twice)
    #   too slow          = throughput - goodput
    return {
        "base": round(base, 1),
        "final": round(final_thr, 1),
        "base_good": round(seed["goodput"], 1),
        "final_good": round(best["goodput"], 1),
        "steps": steps,
        "base_L": seed.get("concurrency"),
    }


TEMPLATE = """<meta charset="utf-8"><title>Tailored inference</title>
<style>
:root {
  color-scheme: light;
  --bg:#ffffff; --panel:#fbfbfa; --ink:#0b0b0b; --muted:#4a4a44; --faint:#7d7c74;
  --rule:#e7e6e1; --base:#a59bff; --final:#a59bff; --up:#33cc7d; --down:#ff7d76;
  --shadow:0 1px 2px rgba(0,0,0,.05),0 8px 24px rgba(0,0,0,.05);
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --bg:#131316; --panel:#1a1a1e; --ink:#f4f4f2; --muted:#b9b8b0; --faint:#8b8a82;
    --rule:#2b2b30; --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px rgba(0,0,0,.35);
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --bg:#131316; --panel:#1a1a1e; --ink:#f4f4f2; --muted:#b9b8b0; --faint:#8b8a82;
  --rule:#2b2b30; --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px rgba(0,0,0,.35);
}
* { box-sizing:border-box }
body { margin:0; background:var(--bg); color:var(--ink);
  font:15px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Inter,Helvetica,Arial,sans-serif;
  -webkit-font-smoothing:antialiased }
.wrap { max-width:1120px; margin:0 auto; padding:46px 28px 64px }
h1 { font-size:30px; line-height:1.15; letter-spacing:-.02em; margin:0 0 12px }
.lede { color:var(--muted); max-width:70ch; margin:0 0 26px; font-size:15.5px }
.controls { display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin:0 0 22px }
button { font:inherit; font-size:13.5px; color:var(--ink); background:var(--panel);
  border:1px solid var(--rule); border-radius:999px; padding:7px 15px; cursor:pointer;
  transition:background .15s,border-color .15s }
button:hover { border-color:var(--faint) }
button[aria-pressed="true"] { background:var(--ink); color:var(--bg); border-color:var(--ink) }
button:focus-visible { outline:2px solid var(--base); outline-offset:2px }
.card { background:var(--panel); border:1px solid var(--rule); border-radius:16px;
  padding:26px 26px 12px; box-shadow:var(--shadow) }
.headline { display:flex; gap:34px; flex-wrap:wrap; margin:0 0 6px }
.stat b { display:block; font-size:30px; letter-spacing:-.02em; line-height:1.1 }
.stat span { color:var(--faint); font-size:12.5px }
svg { display:block; width:100%; height:auto; overflow:visible }
.bar { cursor:pointer; transition:opacity .12s }
.bar:focus-visible { outline:2px solid var(--ink); outline-offset:2px }
.dim .bar { opacity:.32 }
.dim .bar.on { opacity:1 }
.tick { fill:var(--faint); font-size:11.5px }
.grid { stroke:var(--rule); stroke-width:1 }
.conn { stroke:var(--faint); stroke-width:1; stroke-dasharray:2 3 }
.vlab { fill:var(--ink); font-size:12.5px; font-weight:600 }
.xlab { fill:var(--muted); font-size:12px }
#tip { position:fixed; pointer-events:none; opacity:0; transition:opacity .1s;
  background:var(--ink); color:var(--bg); padding:9px 12px; border-radius:9px;
  font-size:12.5px; max-width:290px; box-shadow:var(--shadow); z-index:9 }
#tip b { display:block; margin-bottom:3px; font-size:13px }
#tip code { font:12px ui-monospace,SFMono-Regular,Menlo,monospace; opacity:.85 }
table { border-collapse:collapse; width:100%; margin-top:8px; font-size:13.5px }
th,td { text-align:right; padding:7px 10px; border-bottom:1px solid var(--rule) }
th:first-child,td:first-child { text-align:left }
th { color:var(--faint); font-weight:600; font-size:12px; text-transform:uppercase;
  letter-spacing:.04em }
td.num { font-variant-numeric:tabular-nums }
[hidden] { display:none !important }
.foot { color:var(--faint); font-size:12.5px; margin-top:22px; max-width:80ch }
</style>
<div class="wrap">
  <h1>Same model, same machine, __MULT__&times; the throughput</h1>
  <p class="lede">Every bar is a separate measured launch on one accelerator against a
  fixed workload. The first is the server as it ships; each step after it is one
  configuration change and what it was worth; the last is the same hardware with every
  change that survived re-measurement. Hover any bar for the change it represents.</p>

  <div class="controls" id="picker"></div>

  <div class="card">
    <div class="headline">
      <div class="stat"><b id="s-base"></b><span>tok/s as it ships</span></div>
      <div class="stat"><b id="s-final"></b><span>tok/s tailored</span></div>
      <div class="stat"><b id="s-mult"></b><span>improvement</span></div>
    </div>
    <svg id="chart" viewBox="0 0 1000 420" role="img" aria-labelledby="cap"></svg>
    <p id="cap" class="sr" hidden></p>
    <table id="tbl" hidden>
      <thead><tr><th>Step</th><th>Change</th><th>Delta</th><th>Throughput</th></tr></thead>
      <tbody></tbody>
    </table>
  </div>

  <div class="controls" style="margin-top:16px">
    <button id="toggle-table" aria-pressed="false">Show the numbers</button>
    <button id="toggle-theme" aria-pressed="false">Dark</button>
  </div>

  <p class="foot">Bar height is every token the server produced per second; the solid
  portion of the first and last bars is what reached a user inside a 500&thinsp;ms
  first-token / 250&thinsp;ms inter-token target, and the pale remainder arrived too
  slowly to be useful. Bars that fall are steps that traded raw throughput for
  latency &mdash; shown as measured rather than hidden.</p>
</div>
<div id="tip" role="status"></div>
<script>
const DATA = __DATA__, PAL = __PAL__;
let cur = 0, dim = null;
const $ = s => document.querySelector(s);
const fmt = n => n.toLocaleString(undefined,{maximumFractionDigits:0});

const picker = $("#picker");
if (DATA.length > 1) DATA.forEach((d,i) => {
  const b = document.createElement("button");
  b.textContent = d.name; b.setAttribute("aria-pressed", i===0);
  b.onclick = () => { cur = i; draw(); };
  picker.appendChild(b);
});

function draw() {
  const d = DATA[cur];
  [...picker.children].forEach((b,i) => b.setAttribute("aria-pressed", i===cur));
  const mult = d.final / d.base;
  $("#s-base").textContent = fmt(d.base);
  $("#s-final").textContent = fmt(d.final);
  $("#s-mult").textContent = mult.toFixed(1) + "\\u00d7";
  document.querySelector("h1").textContent =
    `Same model, same machine, ${mult.toFixed(1)}\\u00d7 the throughput`;

  // Bars: baseline, one per accepted step, then the tailored total.
  const bars = [{label:"As it ships", sub:"stock settings", from:0, to:d.base,
                 kind:"base", good:d.base_good}];
  let run = d.base;
  d.steps.forEach(s => {
    bars.push({label:s.label, sub:s.config, from:Math.min(run, run+s.delta),
               to:Math.max(run, run+s.delta), delta:s.delta, kind:"step", after:s.after});
    run += s.delta;
  });
  bars.push({label:"Personalized\\ninference", sub:"every change that held", from:0,
             to:d.final, kind:"final", good:d.final_good});

  const W=1000, H=420, L=64, R=18, T=26, B=74;
  const max = Math.max(d.final, ...bars.map(b=>b.to)) * 1.12;
  const y = v => H - B - (v/max)*(H-T-B);
  const n = bars.length, slot = (W-L-R)/n, bw = Math.min(78, slot*0.62);

  let s = "";
  const ticks = 5;
  for (let i=0;i<=ticks;i++) {
    const v = max*i/ticks;
    s += `<line class="grid" x1="${L}" x2="${W-R}" y1="${y(v)}" y2="${y(v)}"/>`;
    s += `<text class="tick" x="${L-10}" y="${y(v)+4}" text-anchor="end">${fmt(v)}</text>`;
  }
  s += `<text class="tick" x="${L-10}" y="${T-8}" text-anchor="end">tok/s</text>`;

  bars.forEach((b,i) => {
    const cx = L + slot*i + slot/2, x = cx - bw/2;
    const col = b.kind==="base" || b.kind==="final" ? PAL[0]
              : (b.delta >= 0 ? PAL[4] : PAL[1]);
    const h = Math.max(2, y(b.from) - y(b.to));
    if (b.kind==="step" && i>0) {
      const py = y(bars[i-1].kind==="base" ? bars[i-1].to : (bars[i-1].after ?? bars[i-1].to));
      s += `<line class="conn" x1="${L+slot*(i-1)+slot/2+bw/2}" x2="${x}" y1="${py}" y2="${py}"/>`;
    }
    s += `<rect class="bar" tabindex="0" data-i="${i}" x="${x}" y="${y(b.to)}" `
       + `width="${bw}" height="${h}" rx="5" fill="${col}" fill-opacity="${b.good?0.35:1}"/>`;
    if (b.good) {                       // the portion that arrived in time
      s += `<rect class="bar" tabindex="0" data-i="${i}" x="${x}" y="${y(b.good)}" `
         + `width="${bw}" height="${Math.max(2,y(0)-y(b.good))}" rx="5" fill="${col}"/>`;
      s += `<text class="tick" x="${cx}" y="${y(b.good)-7}" text-anchor="middle">`
         + `${fmt(b.good)} in time</text>`;
    }
    const val = b.kind==="step" ? (b.delta>=0?"+":"\\u2212") + fmt(Math.abs(b.delta)) : fmt(b.to);
    s += `<text class="vlab" x="${cx}" y="${y(b.to)-9}" text-anchor="middle">${val}</text>`;
    b.label.split("\\n").forEach((ln,k) => {
      s += `<text class="xlab" x="${cx}" y="${H-B+22+k*15}" text-anchor="middle">${ln}</text>`;
    });
  });
  $("#chart").innerHTML = s;

  const tb = $("#tbl tbody"); tb.innerHTML = "";
  bars.forEach(b => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${b.label.replace("\\n"," ")}</td><td>${b.sub}</td>`
      + `<td class="num">${b.kind==="step" ? (b.delta>=0?"+":"\\u2212")+fmt(Math.abs(b.delta)) : "\\u2014"}</td>`
      + `<td class="num">${fmt(b.kind==="step" ? b.after : b.to)}</td>`;
    tb.appendChild(tr);
  });
  $("#cap").textContent = `Throughput rises from ${fmt(d.base)} to ${fmt(d.final)} tok/s `
    + `across ${d.steps.length} configuration changes.`;

  const tip = $("#tip"), chart = $("#chart");
  chart.querySelectorAll(".bar").forEach(r => {
    const b = bars[+r.dataset.i];
    const show = e => {
      chart.parentElement.classList.add("dim"); r.classList.add("on");
      tip.innerHTML = `<b>${b.label.replace("\\n"," ")}</b>`
        + (b.kind==="step"
            ? `${b.delta>=0?"+":"\\u2212"}${fmt(Math.abs(b.delta))} tok/s \\u2192 `
              + `${fmt(b.after)} tok/s<br><code>${b.sub}</code>`
            : `${fmt(b.to)} tok/s produced, ${fmt(b.good)} of it inside the `
              + `latency target (${Math.round(100*(b.to-b.good)/b.to)}% too slow)`
              + `<br><code>${b.sub}</code>`);
      tip.style.opacity = 1;
      const p = r.getBoundingClientRect();
      tip.style.left = Math.min(innerWidth-306, Math.max(8, p.left+p.width/2-140)) + "px";
      tip.style.top  = Math.max(8, p.top - tip.offsetHeight - 12) + "px";
    };
    const hide = () => { chart.parentElement.classList.remove("dim");
                         r.classList.remove("on"); tip.style.opacity = 0; };
    r.addEventListener("mouseenter", show);
    r.addEventListener("focus", show);
    r.addEventListener("mouseleave", hide);
    r.addEventListener("blur", hide);
  });
}

$("#toggle-table").onclick = e => {
  const on = $("#tbl").hasAttribute("hidden");
  $("#tbl").toggleAttribute("hidden", !on);
  e.target.setAttribute("aria-pressed", on);
  e.target.textContent = on ? "Hide the numbers" : "Show the numbers";
};
$("#toggle-theme").onclick = e => {
  const dark = document.documentElement.getAttribute("data-theme") !== "dark";
  document.documentElement.setAttribute("data-theme", dark ? "dark" : "light");
  e.target.setAttribute("aria-pressed", dark);
  e.target.textContent = dark ? "Light" : "Dark";
};
draw();
</script>
"""


def build(runs: list, names: list, out: Path) -> Path:
    data = []
    for r, nm in zip(runs, names):
        w = walk(Path(r))
        w["name"] = nm
        data.append(w)
    mult = data[0]["final"] / data[0]["base"]
    page = (TEMPLATE
            .replace("__DATA__", json.dumps(data))
            .replace("__PAL__", json.dumps(PALETTE))
            .replace("__MULT__", f"{mult:.1f}"))
    out.write_text(page)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("flow_html")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--name", default="")
    ap.add_argument("-o", "--out", default="tailored-inference.html")
    a = ap.parse_args(argv)
    names = a.name.split(",") if a.name else [Path(r).name for r in a.runs]
    p = build(a.runs, names, Path(a.out))
    print(f"  wrote {p}")
    for r, nm in zip(a.runs, names):
        w = walk(Path(r))
        print(f"    {nm:12s} {w['base']:7.1f} -> {w['final']:7.1f} tok/s "
              f"({w['final'] / w['base']:.2f}x, {len(w['steps'])} steps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
