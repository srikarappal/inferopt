"""Write a run out in the schema AIPerf uses, so the two are comparable.

WHY THIS SCHEMA AND NOT OUR OWN. A measurement nobody else can read is a
measurement people are entitled to doubt. NVIDIA's AIPerf is the closest thing
to a default in this space, so emitting its field names and units means anyone
holding an AIPerf run can diff ours against it directly, with their tools and
without trusting our arithmetic.

WHAT WE DO NOT CLAIM. These files carry `producer: goodput/<version>` and no
`aiperf_version`. They are AIPerf-SHAPED, not AIPerf-PRODUCED, and the one
thing worse than an unreadable measurement is a file that misrepresents who
measured it. Fields AIPerf reports that we do not measure (GPU telemetry, HTTP
transport timings) are absent rather than zero-filled, because a zero reads as
"measured, and it was zero".

WHAT WE ADD. Our own objective is tokens/sec of SLO-conforming output, which
AIPerf does not have a field for -- its `goodput` is requests/sec. Both are
emitted: `goodput` in requests/sec under AIPerf's name and unit, and our
tokens/sec figure under `goodput_tokens_per_sec`, in a clearly separate block
so nobody reads one as the other. That confusion is exactly why summarize()
already reports both.

Files written, mirroring an AIPerf artifact directory:

    profile_export_aiperf.json    aggregate, with percentiles and run config
    profile_export.jsonl          one record per request, timestamps included
    server_metrics_export.json    whatever was scraped from /metrics
    inputs.json                   the prompts and output lengths actually sent
    profile_export_console.txt    the same numbers, for a human
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

from goodput.driver import Req
from goodput.metrics import summarize
from goodput.slo import LatencyTarget as SLO

SCHEMA_VERSION = "1.4"
PRODUCER = "goodput/0.1.0"

# Same convention summarize() uses, so a percentile here and a percentile there
# are the same number. Nearest-rank on the sorted sample, no interpolation: with
# ~380 requests in a window, p99 is a max over the slowest four either way, and
# interpolating between two of them invents precision the sample does not have.
_QUANTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99)


def _pct(xs: list[float], q: float) -> float:
    return xs[min(len(xs) - 1, int(q * (len(xs) - 1)))] if xs else float("nan")


def stat(values: list[float], unit: str) -> dict:
    """One metric in AIPerf's shape: summary, spread, and every percentile."""
    xs = sorted(v for v in values if v is not None and not math.isnan(v))
    if not xs:
        return {"unit": unit, "count": 0}
    n = len(xs)
    avg = sum(xs) / n
    var = sum((x - avg) ** 2 for x in xs) / n if n > 1 else 0.0
    out = {"unit": unit, "count": n, "avg": avg, "min": xs[0], "max": xs[-1],
           "std": math.sqrt(var), "sum": sum(xs)}
    out.update({f"p{q}": _pct(xs, q / 100) for q in _QUANTILES})
    return out


def scalar(value: float, unit: str) -> dict:
    return {"unit": unit, "avg": value}


def _epoch_offset_ns() -> int:
    """Convert perf_counter seconds to epoch nanoseconds.

    Req.start is a perf_counter reading, which is monotonic and has no defined
    zero, while every other tool timestamps in epoch time. Reading both clocks
    back to back gives the offset to within their sampling gap -- sub-millisecond
    -- which is far below anything these records are used to reason about.
    """
    return time.time_ns() - int(time.perf_counter() * 1e9)


def _itl_ms(r: Req) -> float | None:
    if r.ttft is None or r.n_out <= 1:
        return None
    return (r.latency - r.ttft) / (r.n_out - 1) * 1e3


def aggregate(reqs: list[Req], t0: float, t1: float, slo: SLO, *,
              config: dict | None = None, run_info: dict | None = None) -> dict:
    """The aggregate export, AIPerf field names, our measurements."""
    m = summarize(reqs, t0, t1, slo)
    started = [r for r in reqs if t0 <= r.start < t1]
    done = [r for r in started if r.ok]
    win = max(1e-9, t1 - t0)
    off = _epoch_offset_ns()

    ttft_ms = [r.ttft * 1e3 for r in done if r.ttft is not None]
    itls = [v for v in (_itl_ms(r) for r in done) if v is not None]
    lat_ms = [r.latency * 1e3 for r in done]
    osl = [float(r.n_out) for r in done]
    isl = [float(r.n_in) for r in done if r.n_in]
    per_user = [r.n_out / r.latency for r in done if r.latency > 0]

    out = {
        "schema_version": SCHEMA_VERSION,
        "producer": PRODUCER,
        "was_cancelled": False,
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%S",
                                    time.localtime((t0 * 1e9 + off) / 1e9)),
        "end_time": time.strftime("%Y-%m-%dT%H:%M:%S",
                                  time.localtime((t1 * 1e9 + off) / 1e9)),
        "benchmark_duration": scalar(win, "sec"),
        "request_count": scalar(float(len(done)), "requests"),
        "request_throughput": scalar(len(done) / win, "requests/sec"),
        "request_latency": stat(lat_ms, "ms"),
        "time_to_first_token": stat(ttft_ms, "ms"),
        "inter_token_latency": stat(itls, "ms"),
        "output_sequence_length": stat(osl, "tokens"),
        "input_sequence_length": stat(isl, "tokens"),
        "output_token_throughput": scalar(m["throughput"], "tokens/sec"),
        "output_token_throughput_per_user": stat(per_user, "tokens/sec/user"),
        "goodput": scalar(m["goodput_req_s"], "requests/sec"),
        "good_request_count": scalar(float(round(m["goodput_req_s"] * win)), "requests"),
        "total_output_tokens": scalar(float(sum(r.n_out for r in done)), "tokens"),
        "total_osl": scalar(float(sum(osl)), "tokens"),
        "total_isl": scalar(float(sum(isl)), "tokens"),
        # OURS, kept separate and named so it cannot be read as an AIPerf field.
        # goodput above is requests/sec because that is what AIPerf's goodput
        # means; the objective this project optimises is the tokens/sec one.
        "goodput_extensions": {
            "goodput_tokens_per_sec": m["goodput"],
            "slo_attainment": m["slo_attainment"],
            "slo_attainment_denominator": "requests STARTED in window, failures included",
            "ttft_p95_ms": m["ttft_p95_ms"],
            "itl_p95_ms": m["itl_p95_ms"],
            "ttft_n": m["ttft_n"],
            "failed": m["failed"],
            "failure_reasons": m["failure_reasons"],
            "slo_targets": {"ttft_p99_ms": getattr(slo, "ttft_p99_ms", None),
                            "itl_p99_ms": getattr(slo, "itl_p99_ms", None)},
        },
    }
    if config is not None:
        out["input_config"] = config
    if run_info is not None:
        out["run_info"] = run_info
    return out


def records(reqs: list[Req], t0: float, t1: float, slo: SLO) -> list[dict]:
    """Per-request rows, AIPerf's metadata/metrics split.

    Every request that STARTED in the window is emitted, failures included, and
    a failure carries its reason. Dropping them would make the file disagree
    with slo_attainment, whose denominator is deliberately `started`.
    """
    off = _epoch_offset_ns()
    rows = []
    for r in reqs:
        if not (t0 <= r.start < t1):
            continue
        md = {
            "request_start_ns": int(r.start * 1e9 + off),
            "request_end_ns": int((r.start + r.latency) * 1e9 + off),
            "benchmark_phase": "profiling",
            "phase_kind": "profiling",
            "was_cancelled": False,
            "ok": r.ok,
        }
        if r.error:
            md["error"] = r.error
        mx: dict = {"output_sequence_length": {"value": r.n_out, "unit": "tokens"},
                    "request_latency": {"value": r.latency * 1e3, "unit": "ms"},
                    "good_request_count": {"value": int(r.meets(slo)), "unit": "requests"}}
        if r.n_in:
            mx["input_sequence_length"] = {"value": r.n_in, "unit": "tokens"}
        if r.ttft is not None:
            mx["time_to_first_token"] = {"value": r.ttft * 1e3, "unit": "ms"}
        itl = _itl_ms(r)
        if itl is not None:
            mx["inter_token_latency"] = {"value": itl, "unit": "ms"}
        rows.append({"metadata": md, "metrics": mx})
    return rows


def console(agg: dict) -> str:
    """The same numbers a human can read without a JSON viewer."""
    g = agg["goodput_extensions"]
    w = [f"{'metric':32s} {'value':>14s}  unit", "-" * 56]
    def row(name, d):
        v = d.get("avg")
        w.append(f"{name:32s} {v:14.2f}  {d.get('unit','')}" if v is not None
                 else f"{name:32s} {'-':>14s}  {d.get('unit','')}")
    for name in ("request_throughput", "output_token_throughput", "goodput",
                 "request_count", "benchmark_duration"):
        row(name, agg[name])
    w.append("")
    w.append(f"{'metric':32s} {'avg':>10s} {'p50':>10s} {'p95':>10s} {'p99':>10s}")
    w.append("-" * 76)
    for name in ("time_to_first_token", "inter_token_latency", "request_latency",
                 "input_sequence_length", "output_sequence_length"):
        d = agg[name]
        if not d.get("count"):
            continue
        w.append(f"{name:32s} {d['avg']:10.2f} {d['p50']:10.2f} "
                 f"{d['p95']:10.2f} {d['p99']:10.2f}   {d['unit']} (n={d['count']})")
    w += ["", f"{'goodput (tokens/sec)':32s} {g['goodput_tokens_per_sec']:14.2f}",
          f"{'slo attainment':32s} {g['slo_attainment']:14.3f}",
          f"{'failed':32s} {g['failed']:14d}"]
    if g["failure_reasons"]:
        w.append(f"  failure reasons: {g['failure_reasons']}")
    w.append(f"  slo targets: {g['slo_targets']}")
    w.append(f"  produced by {agg['producer']} (schema {agg['schema_version']}, not AIPerf)")
    return "\n".join(w) + "\n"


def write_run(out_dir, reqs: list[Req], t0: float, t1: float, slo: SLO, *,
              config: dict | None = None, run_info: dict | None = None,
              server_metrics: dict | None = None,
              prompts: list[str] | None = None,
              output_lengths=None) -> Path:
    """Write the whole artifact directory. Returns the directory."""
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)

    agg = aggregate(reqs, t0, t1, slo, config=config, run_info=run_info)
    (d / "profile_export_aiperf.json").write_text(json.dumps(agg, indent=2, default=str))
    with open(d / "profile_export.jsonl", "w") as fh:
        for row in records(reqs, t0, t1, slo):
            fh.write(json.dumps(row, default=str) + "\n")
    (d / "profile_export_console.txt").write_text(console(agg))
    if server_metrics is not None:
        (d / "server_metrics_export.json").write_text(
            json.dumps({"metrics": server_metrics}, indent=2, default=str))
    if prompts is not None:
        n = len(prompts)
        lens = list(output_lengths) if output_lengths is not None else []
        (d / "inputs.json").write_text(json.dumps({
            "count": n,
            "entries": [{"text": p,
                         "output_length": (lens[i % len(lens)] if lens else None)}
                        for i, p in enumerate(prompts)],
        }, indent=2, default=str))
    return d
