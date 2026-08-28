"""Exercise the measurement path against a fake server. No GPU, seconds to run.

    python selftest.py

Exists because three separate wasted runs were caused by defects that a single
in-process call would have caught:

  run 3   crashed at the quality node after nine launches -- a KeyError in a
          code path no test ever entered
  run 5   measured every node at L=4 and produced three identical numbers,
          because the operating point was pinned to the seed's peak
  (caught) `med` was referenced nine times and assigned zero times after an
          edit deleted the block that computed it -- a guaranteed NameError on
          the first node, ~9 minutes in

The common shape: the code parsed, imported, and looked right. Nothing executed
`measure()` end to end without a GPU, so nothing found out.

This runs the real `measure()` and the real `traverse()` against a synthetic
server whose latency degrades with concurrency, so goodput genuinely peaks.
It asserts on structure and invariants, not on exact numbers.
"""

from __future__ import annotations

import asyncio
import sys
import time

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  -- {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(f"{name}: {detail}")


# --------------------------------------------------------------------------
# a synthetic server with a real peak
# --------------------------------------------------------------------------

PEAK_L = 32
STATE = {"inflight": 0, "max_seen": 0}


def install_fake_server(ev, per_token_s: float = 0.004):
    """Latency rises once in-flight exceeds PEAK_L, so goodput peaks there."""
    async def fake_one(c, base_url, model, prompt, max_tokens, stream=True):
        STATE["inflight"] += 1
        STATE["max_seen"] = max(STATE["max_seen"], STATE["inflight"])
        L = STATE["inflight"]
        # Steep enough that TTFT actually crosses the 500ms SLO above
        # PEAK_L. With a gentle penalty goodput rises monotonically and
        # there is no interior maximum for the bracket to find -- the
        # test then asserts something the model cannot exhibit.
        penalty = 0.02 * max(0, L - PEAK_L)
        r = ev.Req(start=time.perf_counter())
        await asyncio.sleep(0.01 + penalty)
        r.ttft = 0.01 + penalty
        n = 5
        for i in range(n):
            await asyncio.sleep(per_token_s)
            r.token_times.append(time.perf_counter())
        r.n_out = n
        r.latency = time.perf_counter() - r.start
        r.ok = True
        STATE["inflight"] -= 1
        return r
    ev._one = fake_one


def main() -> int:
    import importlib.util

    spec = importlib.util.spec_from_file_location("ev", "evaluator.py")
    ev = importlib.util.module_from_spec(spec)
    sys.modules["ev"] = ev
    spec.loader.exec_module(ev)
    install_fake_server(ev)

    # Short windows so the whole file runs in seconds.
    ev.SETTLE_S, ev.SWEEP_WINDOW_S, ev.WARMUP_S = 0.15, 0.6, 0.1

    from fingerprint import SLO
    from request import InferOptRequest, build_fingerprint

    fp, slo = build_fingerprint(InferOptRequest(
        model="Qwen/Qwen3-14B", trace="data/trace.jsonl",
        ttft_p99_ms=500, itl_p99_ms=250, allow_loss=0.03))

    print("\n=== _closed_loop: window excludes settle and drain ===")
    reqs, t0, t1 = asyncio.run(ev._closed_loop(
        "http://x", "m", ["p"] * 64, 8, 8, settle_s=0.3, window_s=0.6))
    check("window is the nominal duration, not settle+window+drain",
          abs((t1 - t0) - 0.6) < 0.02, f"got {t1-t0:.3f}s")
    check("requests exist from before the window opened (pipeline was full)",
          any(r.start < t0 for r in reqs),
          "no request started during settle -- the window opens cold")
    m = ev.summarize(reqs, t0, t1, slo)
    for k in ("goodput", "goodput_req_s", "throughput", "throughput_req_s",
              "ttft_p99_ms", "itl_p99_ms", "slo_attainment", "completed"):
        check(f"summarize emits {k}", k in m, f"keys={sorted(m)}")

    print("\n=== measure(): the full path, the one nothing exercised ===")

    class FakeEval(ev.VllmEvaluator):
        """Real measure(), real bracket, fake server and no subprocess."""
        def __init__(self):
            self.fp, self.slo, self.log = fp, slo, lambda *a: None
            self.gpu, self.port = "0", 8000
            from pathlib import Path
            import tempfile
            self.run_dir = Path(tempfile.mkdtemp())
            self.trace_path = "data/trace.jsonl"
            self.prompts = ["prompt " + str(i) for i in range(64)]
            self.max_tokens, self.qps, self.conc = 8, 16.0, 16
            self.equiv_k, self.equiv_ref = 8, None
            self.base_url = "http://fake"

        from contextlib import contextmanager

        @contextmanager
        def _serve(self, config, tag):
            yield "fake-model"

        def _metrics(self):
            return {}

        def _gpu_memory_gb(self):
            return 1.0

        def _equivalence(self, model):
            return 0.0

    e = FakeEval()
    t = e.measure({"max_num_seqs": 64}, probes=["goodput"], benchmarks=[],
                  node_id="selftest", concurrency=16)

    check("measure() returns without NameError", t is not None)
    check("Trial.concurrency is ASSIGNED, not left null", t.concurrency is not None,
          "this is the field that would have made run five's failure obvious")
    check("Trial.curve carries the bracket", len(t.curve) >= 3,
          f"got {len(t.curve)} points")
    check("curve points carry their concurrency",
          all("concurrency" in c for c in t.curve))
    check("goodput <= throughput (goodput is a subset of tokens)",
          t.goodput <= t.diagnostics["throughput"] + 1e-6,
          f"goodput {t.goodput} > thru {t.diagnostics['throughput']}")
    check("diagnostics carry req/s in both flavours",
          "goodput_req_s" in t.diagnostics and "throughput_req_s" in t.diagnostics,
          f"keys={sorted(t.diagnostics)}")
    check("scored concurrency is the curve's argmax",
          t.concurrency == max(t.curve, key=lambda c: c["goodput"])["concurrency"])

    bracket = sorted(c["concurrency"] for c in t.curve)
    check("bracket spans an octave either side of the anchor",
          bracket[0] <= 8 and bracket[-1] >= 32, f"levels={bracket}")

    print("\n=== the run-five regression: does a better config score higher? ===")
    # Two configs, identical except one sustains more concurrency. Under the old
    # fixed-L code both scored the same; the bracket must separate them.
    global PEAK_L
    PEAK_L = 8
    STATE["max_seen"] = 0
    low = e.measure({"cfg": "narrow"}, probes=["goodput"], benchmarks=[],
                    node_id="narrow", concurrency=16)
    PEAK_L = 64
    STATE["max_seen"] = 0
    high = e.measure({"cfg": "wide"}, probes=["goodput"], benchmarks=[],
                     node_id="wide", concurrency=16)
    check("a config that sustains more concurrency scores higher",
          high.goodput > low.goodput,
          f"wide {high.goodput} vs narrow {low.goodput} -- the bracket is not "
          f"separating them, which is exactly the run-five failure")
    check("the scored point is an interior maximum, not a boundary",
          all(t.concurrency != max(c["concurrency"] for c in t.curve)
              or len(t.curve) >= 5 for t in (low, high)),
          f"narrow L={low.concurrency} of {sorted(c['concurrency'] for c in low.curve)}, "
          f"wide L={high.concurrency} of {sorted(c['concurrency'] for c in high.curve)} "
          f"-- a peak at the top edge means the search stopped before finding it")

    print("\n=== traverse(): full walk, operating point must follow the incumbent ===")
    PEAK_L = 32
    import json
    from pathlib import Path
    from fingerprint import Context
    from traverse import traverse

    ctx = Context(fingerprint=fp, slo=slo, incumbent={"max_num_seqs": 64},
                  quality_baseline={"math_500": 0.74})
    ctx.incumbent_metrics = None
    dag = json.loads(Path("dag/llm.json").read_text())
    res = traverse(dag, ctx, e, log=lambda *a: None, baseline=t, concurrency=16)

    check("traverse completes without error", res is not None)
    check("every trial records its concurrency",
          all(x.concurrency is not None for x in res.trials),
          f"{sum(1 for x in res.trials if x.concurrency is None)} of "
          f"{len(res.trials)} are null")
    missing = [x.node_id for x in res.trials if not x.quality and x.slo_ok]
    check("every successful trial carries a quality coordinate",
          not missing,
          f"{missing} would plot with no accuracy value and vanish from the "
          f"quality view of the frontier")
    check("frontier is non-empty when trials succeeded",
          len(res.frontier()) > 0 or not res.trials)
    check("Result carries the baseline", res.baseline is not None)
    check("Result carries the operating point", res.concurrency is not None)

    print("\n=== report() and the frontier plot render ===")
    from traverse import report
    lines: list[str] = []
    report(res, log=lines.append, demand_tok_s=3988.0)
    blob = "\n".join(lines)
    check("report prints the BASELINE block", "BASELINE" in blob)
    check("report prints replicas", "replicas" in blob)
    check("report prints the concurrency column", " L " in blob or "L*" in blob)

    print()
    if FAILURES:
        print(f"  {len(FAILURES)} FAILURE(S)")
        for f in FAILURES:
            print(f"    - {f}")
        return 1
    print("  all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
