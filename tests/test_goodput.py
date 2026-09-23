"""Unit tests for the standalone `goodput` package.

Deliberately imports NOTHING from inferopt. If this file ever needs an inferopt
import, the package has stopped being reusable and that is the bug, not this
test.

The properties worth testing here are the ones that were wrong at some point, or
that a reader could plausibly get wrong when reusing the package:

  the attainment denominator      failures belong in it, and excluding them once
                                  reported 100% attainment for a config losing a
                                  fifth of its requests to HTTP 400
  the two goodput units           AIPerf's goodput is requests/sec, ours is
                                  tokens/sec. Conflating them silently rescales
                                  every number by mean_output_tokens
  window edges                    a request straddling the boundary contributes
                                  exactly the tokens it produced inside it
  export honesty                  files must not claim to be AIPerf's, and must
                                  not zero-fill what was never measured
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from goodput import export
from goodput.driver import Req, _mt
from goodput.metrics import summarize
from goodput.slo import Latency

FAIL: list[str] = []
N = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global N
    N += 1
    if not cond:
        FAIL.append(f"{name}: {detail}")
        print(f"  FAIL  {name}" + (f"  -- {detail}" if detail else ""))


def section(t: str) -> None:
    print(f"\n=== {t} ===")


def req(start=0.0, ttft=0.05, n_out=10, ok=True, itl=0.01, error="", n_in=100):
    """A request whose token times are consistent with its ttft and itl."""
    r = Req(start=start, ttft=ttft, n_out=n_out, ok=ok, error=error, n_in=n_in)
    r.latency = ttft + max(0, n_out - 1) * itl
    r.token_times = [start + ttft + i * itl for i in range(n_out)]
    return r


# --------------------------------------------------------------------------
def test_a_chunk_is_credited_with_the_tokens_it_carried():
    """SGLang streams a diffusion LM one block per chunk. Counting chunks read
    a 34 tokens/s server as 1.3; usage per chunk says what each one held."""
    import asyncio
    from goodput import driver

    class _Resp:
        status_code = 200
        def __init__(self, lines): self._lines = lines
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def aiter_lines(self):
            for line in self._lines:
                yield line

    class _Client:
        def __init__(self, lines): self._lines = lines
        def stream(self, *a, **k): return _Resp(self._lines)

    def chunk(text, done):
        return "data: " + json.dumps({"choices": [{"text": text}],
                                      "usage": {"completion_tokens": done, "prompt_tokens": 7}})
    block_stream = [chunk("a" * 32, 32), chunk("b" * 32, 64), chunk("c" * 8, 72),
                    "data: " + json.dumps({"choices": [], "usage": {"completion_tokens": 72, "prompt_tokens": 7}}),
                    "data: [DONE]"]
    r = asyncio.run(driver._one(_Client(block_stream), "http://x", "m", "p", 72))
    assert r.ok and r.n_out == 72 and r.n_in == 7
    assert len(r.token_times) == 72, "three chunks carried 72 tokens"

    # A server that reports usage only at the end: one token per chunk, exact
    # for an autoregressive stream.
    ar_stream = [chunk_no_usage for chunk_no_usage in (
        "data: " + json.dumps({"choices": [{"text": "x"}]}),
        "data: " + json.dumps({"choices": [{"text": "y"}]}),
        "data: " + json.dumps({"choices": [], "usage": {"completion_tokens": 2, "prompt_tokens": 3}}),
        "data: [DONE]")]
    r = asyncio.run(driver._one(_Client(ar_stream), "http://x", "m", "p", 2))
    assert r.n_out == 2 and len(r.token_times) == 2


def test_meets():
    section("Req.meets: the per-request predicate goodput is built on")
    unconstrained = Latency()
    check("no thresholds means every ok request counts",
          req().meets(unconstrained))
    check("a failed request never counts", not req(ok=False).meets(unconstrained))
    check("ttft=None never counts (nothing was ever produced)",
          not Req(start=0, ttft=None, ok=True).meets(unconstrained))

    slo = Latency(ttft_p99_ms=100, itl_p99_ms=20)
    check("ttft under the target passes", req(ttft=0.09).meets(slo))
    check("ttft over the target fails", not req(ttft=0.11).meets(slo))
    check("itl under the target passes", req(itl=0.019).meets(slo))
    check("itl over the target fails", not req(itl=0.021).meets(slo))

    # n_out-1 intervals, not n_out. Dividing by n_out understates ITL by
    # 1/n_out, which at n_out=2 is a 50% error and would pass a config that
    # misses the target.
    r = req(ttft=0.01, n_out=2, itl=0.03)
    check("itl divides by n_out-1, so a 2-token response is one interval",
          not r.meets(Latency(itl_p99_ms=20)),
          f"latency={r.latency:.3f} implies itl {(r.latency - r.ttft) * 1e3:.0f}ms")
    check("a 1-token response is exempt from itl (no interval exists)",
          req(n_out=1, ttft=0.01).meets(Latency(itl_p99_ms=1)))

    check("ttft and itl are independent: failing either fails the request",
          not req(ttft=0.5, itl=0.001).meets(slo)
          and not req(ttft=0.001, itl=0.5).meets(slo))


def test_mt():
    section("_mt: resolving a per-request output length")
    check("scalar passes through", _mt(128, 7) == 128)
    check("sequence indexes by request", _mt([1, 2, 3], 1) == 2)
    check("sequence wraps", _mt([1, 2, 3], 4) == 2)
    check("empty sequence degrades to 1, never 0", _mt([], 3) == 1)


def test_attainment_denominator():
    section("slo_attainment counts failures against you")
    slo = Latency(ttft_p99_ms=1000)
    # 8 fast successes, 2 hard failures. Excluding failures from the
    # denominator reports 100% for a server dropping a fifth of its traffic.
    reqs = [req(start=i * 0.1, ttft=0.01) for i in range(8)]
    reqs += [req(start=0.9 + i * 0.1, ok=False, error="HTTP 400") for i in range(2)]
    m = summarize(reqs, 0.0, 10.0, slo)
    check("failures are in the denominator", abs(m["slo_attainment"] - 0.8) < 1e-9,
          f"got {m['slo_attainment']}")
    check("failed is counted", m["failed"] == 2, f"got {m['failed']}")
    check("completed excludes failures", m["completed"] == 8, f"got {m['completed']}")
    check("failure reasons are named, not just tallied",
          m["failure_reasons"] == {"HTTP 400": 2}, f"got {m['failure_reasons']}")


def test_goodput_vs_throughput():
    section("goodput is throughput minus the late tokens")
    fast = [req(start=i * 0.1, ttft=0.01, n_out=10) for i in range(10)]
    slow = [req(start=1.0 + i * 0.1, ttft=5.0, n_out=10) for i in range(10)]
    m = summarize(fast + slow, 0.0, 30.0, Latency(ttft_p99_ms=100))

    check("throughput counts every token", abs(m["throughput"] * 30.0 - 200) < 1e-6,
          f"got {m['throughput'] * 30.0}")
    check("goodput counts only the conforming half",
          abs(m["goodput"] * 30.0 - 100) < 1e-6, f"got {m['goodput'] * 30.0}")
    check("goodput never exceeds throughput", m["goodput"] <= m["throughput"])

    same = summarize(fast + slow, 0.0, 30.0, Latency())
    check("with no thresholds goodput equals throughput",
          abs(same["goodput"] - same["throughput"]) < 1e-9)

    # req/s and tok/s are different numbers and must not be interchangeable.
    check("req/s is tok/s divided by tokens per request, not equal to it",
          abs(m["goodput_req_s"] * 10 - m["goodput"]) < 1e-6,
          f"tok/s={m['goodput']} req/s={m['goodput_req_s']}")


def test_window_edges():
    section("a request straddling the window contributes only its inside tokens")
    # Starts inside at t=9.5, emits one token every 0.5s for 10 tokens, so
    # tokens land at 9.6 .. 14.1 and the window closes at 12.0.
    r = req(start=9.5, ttft=0.1, n_out=10, itl=0.5)
    m = summarize([r], 0.0, 12.0, Latency())
    inside = sum(1 for tt in r.token_times if 0.0 <= tt < 12.0)
    check("only in-window tokens are counted",
          abs(m["throughput"] * 12.0 - inside) < 1e-6,
          f"counted {m['throughput'] * 12.0}, expected {inside}")
    check("the test is meaningful: some tokens really did fall outside",
          inside < r.n_out, f"inside={inside} of {r.n_out}")

    before = summarize([req(start=-5.0)], 0.0, 10.0, Latency())
    check("a request that started before the window is not in the denominator",
          before["slo_attainment"] == 0.0 and before["completed"] == 0)


def test_summarize_empty():
    section("summarize on nothing does not raise")
    m = summarize([], 0.0, 10.0, Latency())
    check("attainment of no requests is 0, not a crash", m["slo_attainment"] == 0.0)
    check("goodput of no requests is 0", m["goodput"] == 0.0)
    check("percentiles of no samples are nan, not 0",
          m["ttft_p99_ms"] != m["ttft_p99_ms"], f"got {m['ttft_p99_ms']}")


# --------------------------------------------------------------------------
def test_stat():
    section("export.stat: the shape every metric is reported in")
    s = export.stat([1.0, 2.0, 3.0, 4.0], "ms")
    check("count is the sample size", s["count"] == 4)
    check("avg", abs(s["avg"] - 2.5) < 1e-9)
    check("min/max", s["min"] == 1.0 and s["max"] == 4.0)
    check("sum", s["sum"] == 10.0)
    check("std of a known sample", abs(s["std"] - 1.118033988749895) < 1e-9,
          f"got {s['std']}")
    check("single sample has zero spread",
          export.stat([7.0], "ms")["std"] == 0.0)

    empty = export.stat([], "ms")
    check("an empty metric reports count 0 and claims no percentiles",
          empty["count"] == 0 and "p99" not in empty and "avg" not in empty,
          f"got {sorted(empty)}")
    check("nan samples are dropped rather than poisoning the stats",
          export.stat([1.0, float('nan'), 3.0], "ms")["count"] == 2)


def test_percentile_convention_matches_summarize():
    section("export and summarize must agree on what p99 means")
    # 100 samples where nearest-rank and linear interpolation differ, so the
    # test can actually fail if the conventions drift apart.
    reqs = [req(start=i * 0.01, ttft=0.001 * (i + 1), n_out=5, itl=0.001)
            for i in range(100)]
    slo = Latency(ttft_p99_ms=10_000)
    m = summarize(reqs, 0.0, 10.0, slo)
    agg = export.aggregate(reqs, 0.0, 10.0, slo)
    for name, ours in (("ttft_p99_ms", agg["time_to_first_token"]["p99"]),
                       ("itl_p99_ms", agg["inter_token_latency"]["p99"])):
        check(f"{name} identical in both", abs(m[name] - ours) < 1e-9,
              f"summarize={m[name]} export={ours}")
    check("p95 also agrees",
          abs(m["ttft_p95_ms"] - agg["time_to_first_token"]["p95"]) < 1e-9)


def test_aggregate_round_trip():
    section("export.aggregate reports the same numbers summarize does")
    reqs = [req(start=i * 0.05, ttft=0.02 + i * 0.001, n_out=20) for i in range(60)]
    reqs += [req(start=3.5, ok=False, error="timeout")]
    slo = Latency(ttft_p99_ms=60, itl_p99_ms=50)
    m = summarize(reqs, 0.0, 10.0, slo)
    a = export.aggregate(reqs, 0.0, 10.0, slo)
    g = a["goodput_extensions"]

    pairs = [("output_token_throughput", a["output_token_throughput"]["avg"], m["throughput"]),
             ("goodput req/s", a["goodput"]["avg"], m["goodput_req_s"]),
             ("goodput tok/s", g["goodput_tokens_per_sec"], m["goodput"]),
             ("slo_attainment", g["slo_attainment"], m["slo_attainment"]),
             ("ttft p99", a["time_to_first_token"]["p99"], m["ttft_p99_ms"]),
             ("itl p99", a["inter_token_latency"]["p99"], m["itl_p99_ms"]),
             ("failed", g["failed"], m["failed"]),
             ("request_count", a["request_count"]["avg"], float(m["completed"]))]
    for name, x, y in pairs:
        check(f"{name} round-trips", abs(x - y) < 1e-6, f"export={x} summarize={y}")

    check("the two goodput units are actually different numbers here",
          abs(a["goodput"]["avg"] - g["goodput_tokens_per_sec"]) > 1.0,
          "if these are equal the test cannot catch a unit mixup")
    check("goodput carries requests/sec as its unit",
          a["goodput"]["unit"] == "requests/sec", a["goodput"]["unit"])
    check("the slo actually measured against is recorded",
          g["slo_targets"] == {"ttft_p99_ms": 60, "itl_p99_ms": 50},
          str(g["slo_targets"]))


def test_records():
    section("export.records: one row per request that started in the window")
    reqs = [req(start=1.0, ttft=0.01), req(start=2.0, ok=False, error="HTTP 400"),
            req(start=-1.0), req(start=99.0)]
    slo = Latency(ttft_p99_ms=100)
    rows = export.records(reqs, 0.0, 10.0, slo)
    m = summarize(reqs, 0.0, 10.0, slo)
    check("row count equals summarize's `started`",
          len(rows) == 2, f"rows={len(rows)}")
    check("summarize agrees on how many started", m["completed"] + m["failed"] == len(rows),
          f"summarize counted {m['completed']}+{m['failed']}, rows={len(rows)}")
    check("failures are emitted, not dropped",
          sum(1 for r in rows if not r["metadata"]["ok"]) == 1)
    check("a failure carries its reason",
          any(r["metadata"].get("error") == "HTTP 400" for r in rows))
    check("good_request_count is per-request 0/1",
          sorted(r["metrics"]["good_request_count"]["value"] for r in rows) == [0, 1])
    check("timestamps are epoch nanoseconds, ordered like the starts",
          rows[0]["metadata"]["request_start_ns"] < rows[1]["metadata"]["request_start_ns"])
    check("epoch conversion lands in a plausible era (year > 2020)",
          rows[0]["metadata"]["request_start_ns"] > 1_600_000_000 * 10**9,
          str(rows[0]["metadata"]["request_start_ns"]))
    check("input_sequence_length is reported when the server gave one",
          rows[0]["metrics"]["input_sequence_length"]["value"] == 100)
    check("input_sequence_length is ABSENT, not 0, when it was never reported",
          "input_sequence_length" not in
          export.records([req(start=1.0, n_in=0)], 0.0, 10.0, slo)[0]["metrics"])


def test_export_honesty():
    section("the files must not claim to be something they are not")
    reqs = [req(start=i * 0.1) for i in range(20)]
    a = export.aggregate(reqs, 0.0, 10.0, Latency())
    check("no aiperf_version is claimed", "aiperf_version" not in a)
    check("producer names us", a["producer"].startswith("goodput/"), a.get("producer"))
    check("schema_version is declared", a["schema_version"] == export.SCHEMA_VERSION)
    for absent in ("telemetry_data", "gpu_telemetry", "http_req_duration",
                   "decode_duration"):
        check(f"{absent} is absent rather than zero-filled", absent not in a)
    check("our extensions are namespaced, not mixed into AIPerf's fields",
          "goodput_extensions" in a and "goodput_tokens_per_sec" not in a)


# Keys AIPerf 0.12.0 puts on every percentile metric, and the core field names
# it reports. Recorded here as a golden so a schema drift on our side shows up
# as a test failure rather than as a file someone's tooling silently misreads.
AIPERF_STAT_KEYS = {"unit", "count", "avg", "min", "max", "std", "sum",
                    "p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99"}
AIPERF_CORE_FIELDS = {
    "request_throughput", "request_latency", "request_count",
    "time_to_first_token", "inter_token_latency", "output_token_throughput",
    "output_token_throughput_per_user", "output_sequence_length",
    "input_sequence_length", "goodput", "good_request_count",
    "total_output_tokens", "benchmark_duration", "total_isl", "total_osl",
    "schema_version", "start_time", "end_time", "was_cancelled",
}


def test_schema_parity():
    section("schema parity with AIPerf 0.12.0")
    reqs = [req(start=i * 0.1, ttft=0.01 + i * 0.001, n_out=10 + i) for i in range(30)]
    a = export.aggregate(reqs, 0.0, 10.0, Latency(ttft_p99_ms=500),
                         config={"max_num_seqs": 256})
    missing = sorted(AIPERF_CORE_FIELDS - set(a))
    check("every core AIPerf field is emitted", not missing, f"missing {missing}")
    check("input_config travels with the numbers", a.get("input_config") == {"max_num_seqs": 256})
    for metric in ("time_to_first_token", "inter_token_latency",
                   "output_sequence_length", "request_latency"):
        extra = AIPERF_STAT_KEYS - set(a[metric])
        check(f"{metric} has AIPerf's full stat-dict", not extra, f"missing {sorted(extra)}")


def test_write_run():
    section("write_run: the artifact directory")
    reqs = [req(start=i * 0.1, ttft=0.01 + i * 0.001) for i in range(25)]
    reqs.append(req(start=1.0, ok=False, error="boom"))
    slo = Latency(ttft_p99_ms=500)
    with tempfile.TemporaryDirectory() as td:
        d = export.write_run(Path(td) / "L8", reqs, 0.0, 10.0, slo,
                             config={"a": 1}, run_info={"node_id": "n"},
                             server_metrics={"vllm:kv_cache_usage_perc": 0.18},
                             prompts=["p0", "p1"], output_lengths=[10, 20])
        names = {p.name for p in d.iterdir()}
        expected = {"profile_export_aiperf.json", "profile_export.jsonl",
                    "profile_export_console.txt", "server_metrics_export.json",
                    "inputs.json"}
        check("all artifacts written", names == expected, f"got {sorted(names)}")

        agg = json.loads((d / "profile_export_aiperf.json").read_text())
        lines = (d / "profile_export.jsonl").read_text().splitlines()
        check("jsonl has one line per started request",
              len(lines) == len(export.records(reqs, 0.0, 10.0, slo)),
              f"{len(lines)} lines")
        check("every jsonl line is valid json", all(json.loads(l) for l in lines))
        check("run_info is carried through", agg["run_info"]["node_id"] == "n")
        sm = json.loads((d / "server_metrics_export.json").read_text())
        check("server metrics are nested under `metrics` like AIPerf's",
              sm["metrics"]["vllm:kv_cache_usage_perc"] == 0.18)
        inp = json.loads((d / "inputs.json").read_text())
        check("inputs record what was sent",
              inp["count"] == 2 and inp["entries"][1]["output_length"] == 20,
              str(inp)[:120])
        txt = (d / "profile_export_console.txt").read_text()
        check("console names the producer so a reader is not misled",
              "not AIPerf" in txt)
        check("console reports the tokens/sec goodput", "goodput (tokens/sec)" in txt)


def test_package_is_standalone():
    section("the package must not drag inferopt in")
    mods = [m for m in sys.modules if m.split(".")[0] == "inferopt"]
    check("importing goodput pulled in no inferopt module", not mods, f"{mods}")


# ==========================================================================
def main() -> int:
    for fn in (test_meets, test_mt, test_attainment_denominator,
               test_goodput_vs_throughput, test_window_edges, test_summarize_empty,
               test_stat, test_percentile_convention_matches_summarize,
               test_aggregate_round_trip, test_records, test_export_honesty,
               test_schema_parity, test_write_run, test_package_is_standalone):
        try:
            fn()
        except Exception as e:
            import traceback
            print(f"  ERROR in {fn.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
            FAIL.append(f"{fn.__name__} raised: {e}")
    print(f"\n  {N - len(FAIL)}/{N} checks passed")
    if FAIL:
        print(f"\n  {len(FAIL)} FAILURE(S):")
        for f in FAIL:
            print(f"    - {f}")
        return 1
    print("  all goodput unit checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
