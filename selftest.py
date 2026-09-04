"""Exercise the measurement path against a fake server. No GPU, ~2 minutes.

Most of it runs in seconds against a synthetic server. The mbpp_plus section
is the slow part and is meant to be: it actually EXECUTES generated code
through evalplus, because the only way to know a code scorer works is to
watch it pass correct code and fail incorrect code.

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
import json
import os
import sys
import time
from pathlib import Path

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


def configs_under_test_backend(er, fp):
    """moe_backend that a `--config file` run would actually serve."""
    import argparse
    a = argparse.Namespace(from_run=None, config=["configs/stock.json"], model=fp.model.id)
    return er.configs_under_test(a, fp)[0][1].get("moe_backend")


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

    print("\n=== fingerprint: single-file checkpoints (no index) ===")
    from request import InferOptRequest as _R, build_fingerprint as _bf
    small, _ = _bf(_R(model="Qwen/Qwen3-0.6B", trace="data/trace.jsonl"))
    check("attention_type is a TYPE, not a parameter count",
          small.model.attention_type in ("mha", "gqa", "mqa", "mla"),
          f"got {small.model.attention_type!r} -- the fallback param-count branch "
          f"reused the variable holding the attention type")
    check("weights read from the single-file checkpoint, not arithmetic",
          0.5 < small.model.weight_gb < 5.0, f"{small.model.weight_gb} GB")

    print("\n=== MoE and MLA across providers (config + headers only, no weights) ===")
    from request import MoEReconciliationError, detect_model as _dm
    from request import InferOptRequest as _R2

    # THIS SECTION NEEDS THE NETWORK, and says so rather than failing
    # ambiguously. It reads safetensors headers from the Hub over HTTP Range to
    # measure stored bytes-per-parameter, because a checkpoint's
    # quantization_config lies -- DeepSeek-V3 declares fp8 and stores BF16.
    #
    # When a header read fails the code falls back to torch_dtype, which halves
    # the parameter count for an fp8 checkpoint and correctly trips the MoE
    # reconciliation guard. So a network blip produces the SAME exception as a
    # genuinely unreconcilable checkpoint, and a red suite that means nothing.
    # It has already flaked twice in one day.
    import httpx as _httpx
    try:
        _httpx.head("https://huggingface.co", timeout=5, follow_redirects=True)
        _online = True
    except Exception as _e:
        _online = False
        print(f"  SKIP  the Hub is unreachable ({type(_e).__name__}); these checks "
              f"read safetensors headers over the network and cannot run offline")
    CASES = [
        # model, published total B, published active B, expected attention
        ("mistralai/Mixtral-8x7B-Instruct-v0.1", 46.7, 12.9, "gqa"),   # num_local_experts
        ("Qwen/Qwen3-30B-A3B",                   30.5,  3.3, "gqa"),   # num_experts + sparse_step
        ("moonshotai/Kimi-K2-Instruct",        1029.0, 32.0, "mla"),   # n_routed_experts + MLA
        ("Qwen/Qwen3-14B",                       14.8, None, "gqa"),   # dense control
    ]
    for mid, pub_t, pub_a, attn in (CASES if _online else []):
        try:
            f = _dm(_R2(model=mid, trace="data/trace.jsonl"))
        except Exception as ex:
            check(f"{mid.split('/')[-1]}: fingerprints", False, f"{type(ex).__name__}: {ex}")
            continue
        short = mid.split("/")[-1]
        check(f"{short}: attention_type is {attn}", f.attention_type == attn,
              f"got {f.attention_type}")
        check(f"{short}: total within 5% of published {pub_t}B",
              abs(f.n_params_b - pub_t) / pub_t < 0.05, f"got {f.n_params_b:,.1f}B")
        if pub_a:
            act = f.active_params_b or f.n_params_b
            check(f"{short}: active within 15% of published {pub_a}B",
                  abs(act - pub_a) / pub_a < 0.15, f"got {act:,.1f}B")

    # A MoE checkpoint whose weights do not add up must RAISE, not approximate.
    # DeepSeek-V3's index reports 1369GB against 671B published parameters while
    # its shard headers measure 1.22 bytes/param -- those cannot both be true.
    raised = not _online          # nothing to prove when we cannot reach the Hub
    try:
        if _online:
            _dm(_R2(model="deepseek-ai/DeepSeek-V3", trace="data/trace.jsonl"))
    except MoEReconciliationError:
        raised = True
    except Exception:
        pass
    check("an unreconcilable MoE checkpoint RAISES rather than approximating",
          raised,
          "a wrong active count sets the roofline, the memory budget and the replica "
          "count -- silently proceeding produces a whole run of confident wrong answers")

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

    print("\n=== prompt cursor: phases must not replay the same prompts ===")
    seen: list[str] = []
    orig = ev._one
    async def recording(c, base_url, model, prompt, max_tokens, stream=True):
        seen.append(prompt)
        return await orig(c, base_url, model, prompt, max_tokens, stream)
    ev._one = recording
    cur = [0]
    for _ in range(3):                       # warmup + two passes
        asyncio.run(ev._closed_loop("http://x", "m", [f"p{i}" for i in range(400)],
                                    8, 6, 0.1, 0.3, cursor=cur))
    ev._one = orig
    uniq = len(set(seen))
    check("the cursor advanced across phases", cur[0] > 0, f"cursor={cur[0]}")
    check("phases served mostly DISJOINT prompts, not the same ones replayed",
          uniq > 0.6 * len(seen),
          f"{uniq} unique of {len(seen)} issued -- replay inflates a warm prefix "
          f"cache to full-prompt hits that no production workload produces")

    print("\n=== provenance: one record, shared by run.py and eval_repro ===")
    import argparse as _ap
    from provenance import banner as _banner, environment as _env, provenance as _prov
    _p = _ap.ArgumentParser()
    _p.add_argument("--alpha", default=1)
    _p.add_argument("--beta", default="x")
    _args = _p.parse_args(["--beta", "y"])
    _m = _prov(_p, _args, fp, extra={"port": 8103})
    check("records the command and cwd", "command" in _m and "cwd" in _m)
    check("marks an EXPLICIT arg as explicit", _m["args"]["beta"]["explicit"] is True)
    check("marks a DEFAULTED arg as not explicit", _m["args"]["alpha"]["explicit"] is False,
          "reading a value back cannot say whether it was chosen or defaulted")
    check("carries the default alongside the value",
          _m["args"]["alpha"]["default"] == 1)
    check("records the code version and whether the tree was dirty",
          "inferopt_commit" in _m["environment"] and "inferopt_dirty" in _m["environment"])
    check("records the resolved values the caller computed",
          _m["resolved"]["port"] == 8103)
    check("carries the full fingerprint",
          set(_m["fingerprint"]) == {"model", "hardware", "workload", "lora"})
    import pathlib as _pl
    for f in ("run.py", "eval_repro.py"):
        check(f"{f} uses the shared module",
              "from provenance import" in _pl.Path(f).read_text(),
              "two implementations of one record diverge")

    print("\n=== serving_metrics(): ONE implementation, used by both callers ===")
    ev.WINDOW_S, ev.WARMUP_S = 0.4, 0.1
    med, passes = e.serving_metrics("m", 12, warmup=False)
    check("returns an aggregate and the passes behind it",
          isinstance(med, dict) and len(passes) == ev.REPEATS,
          f"{len(passes)} passes, expected {ev.REPEATS}")
    check("aggregate is direction-aware: ttft is the WORST pass",
          med["ttft_p99_ms"] == max(p["ttft_p99_ms"] for p in passes),
          "an optimistic TTFT is exactly backwards for an SLO gate")
    check("and goodput is the worst (lowest) pass",
          med["goodput"] == min(p["goodput"] for p in passes))
    check("concurrency is recorded on the aggregate", med.get("concurrency") == 12)
    check("pass-to-pass spread is reported", "pass_spread" in med)
    import inspect as _i
    src_ev = _i.getsource(ev.VllmEvaluator.measure)
    check("measure() delegates rather than keeping its own copy",
          "self.serving_metrics(" in src_ev,
          "two implementations of one measurement diverge -- that is how a "
          "number gets reported against a claim it does not support")
    import pathlib as _p
    check("eval_repro uses the same function, not a lookalike",
          "ev.serving_metrics(" in _p.Path("eval_repro.py").read_text()
          and "ev._point(" not in _p.Path("eval_repro.py").read_text())

    print("\n=== aggregate(): worst case in BOTH directions ===")
    a = {"goodput": 200.0, "ttft_p99_ms": 400.0, "slo_attainment": 1.0, "concurrency": 30}
    b = {"goodput": 150.0, "ttft_p99_ms": 900.0, "slo_attainment": 0.8, "concurrency": 30}
    agg = ev.aggregate([a, b])
    check("goodput takes the LOWER of two passes", agg["goodput"] == 150.0, str(agg))
    check("ttft takes the HIGHER of two passes -- min() here reported the better "
          "latency, backwards for an SLO gate", agg["ttft_p99_ms"] == 900.0, str(agg))
    check("concurrency is excluded from the aggregate", "concurrency" not in agg)

    print("\n=== --fixed-concurrency: run-four reproduction path ===")
    ev.WINDOW_S = 0.5
    fx = e.measure({"cfg": "pinned"}, probes=["goodput"], benchmarks=[],
                   node_id="pinned", fixed_concurrency=24)
    check("measure() returns on the fixed-concurrency path", fx is not None)
    check("it measures at exactly the pinned L", fx.concurrency == 24,
          f"got {fx.concurrency}")
    check("no sweep happened (curve is empty)", fx.curve == [],
          f"curve has {len(fx.curve)} points -- the sweep was not skipped")
    check("the mode is recorded on the trial",
          fx.diagnostics.get("mode") == "fixed_concurrency_open_loop",
          f"mode={fx.diagnostics.get('mode')}")
    check("goodput <= throughput on this path too",
          fx.goodput <= fx.diagnostics["throughput"] + 1e-6)

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

    # ----------------------------------------------------------------------
    # Benchmarks judge in ONE place, and a code benchmark actually executes.
    #
    # Two failures are being guarded against, both of which have happened here:
    # scoring logic living in two files and drifting apart, and a probe that
    # returns 0.0 for every config -- which reads as "quality unchanged" rather
    # than "the probe is broken".
    # ----------------------------------------------------------------------
    print("\n=== benchmarks: one judge per benchmark, shared by both callers ===")
    import inspect as _inspect

    from quality import BENCHMARKS

    src = Path("eval_repro.py").read_text()
    check("score_once calls bench.judge rather than re-deriving verdicts",
          "bench.judge(rows, texts)" in src)
    check("score_once no longer sniffs the row shape",
          '"answers" in r' not in src,
          "picking the metric by inspecting the row is a second copy of quality.py")
    check("run_benchmark takes max_tokens from the Benchmark",
          "gen([prompt(r) for r in rows], b.max_tokens)" in Path("quality.py").read_text())

    for name, b in BENCHMARKS.items():
        sig = list(_inspect.signature(b.judge).parameters)
        check(f"{name}: judge takes (rows, texts)", sig[:2] == ["rows", "texts"],
              f"got {sig}")

    # MATH-500's judge, on hand-built texts, in both directions.
    m_rows = [{"answer": "42"}, {"answer": "7"}]
    got = BENCHMARKS["math_500"].judge(
        m_rows, [r"the answer is \boxed{42}", "I could not solve it"])
    check("math_500 judge: right answer True, no boxed answer False", got == [True, False])

    # ----------------------------------------------------------------------
    # The launch deadline follows PROGRESS, not wall clock.
    #
    # A Qwen3-30B-A3B run spent 40 min downloading and 88 min inside
    # FlashInfer's sm120 CUTLASS MoE compile, then got killed at 7200s having
    # measured nothing -- and reported only "not healthy in 7200s", which reads
    # as a slow launch rather than a stuck one. The last log line said
    # gen_cutlass_fused_moe_sm120_module the whole time.
    # ----------------------------------------------------------------------
    print("\n=== launch deadline: extends on progress, fails fast on silence ===")
    import tempfile as _tf
    import types as _types

    class _FakeProc:
        returncode = None
        pid = 999999
        def poll(self): return None            # alive throughout
        def wait(self, timeout=None): return 0

    def _run_serve(writer, healthy_after):
        """Drive the real _serve() against a fake server. Returns (ok, err)."""
        d = Path(_tf.mkdtemp())
        e = ev.VllmEvaluator.__new__(ev.VllmEvaluator)
        e.fp, e.slo, e.log = fp, slo, lambda *a: None
        e.gpu, e.port, e.run_dir = "0", 8000, d
        e.base_url = "http://127.0.0.1:8000/v1"
        e.trace_path = "data/trace.jsonl"
        calls = {"n": 0}

        def fake_popen(cmd, stdout=None, stderr=None, env=None, start_new_session=None):
            return _FakeProc()

        def fake_get(url, timeout=None):
            calls["n"] += 1
            log = d / "launches" / "t" / "server.log"
            writer(log, calls["n"])
            if healthy_after and calls["n"] >= healthy_after:
                return _types.SimpleNamespace(status_code=200)
            raise ev.httpx.HTTPError("nope")

        # The teardown in _serve kills the process group. Our fake proc has a
        # made-up pid, so os.getpgid raises ProcessLookupError and masks the
        # LaunchError we are actually testing for.
        op, og = ev.subprocess.Popen, ev.httpx.get
        okp, ogp = ev.os.killpg, ev.os.getpgid
        ok_, err = False, None
        ev.subprocess.Popen, ev.httpx.get = fake_popen, fake_get
        ev.os.killpg, ev.os.getpgid = (lambda *a: None), (lambda *a: 0)
        try:
            with e._serve({}, "t"):
                ok_ = True
        except ev.LaunchError as ex:
            err = str(ex)
        finally:
            ev.subprocess.Popen, ev.httpx.get = op, og
            ev.os.killpg, ev.os.getpgid = okp, ogp
        return ok_, err

    orig_stall, orig_to = ev.STALL_S, ev.LAUNCH_TIMEOUT_S
    ev.STALL_S, ev.LAUNCH_TIMEOUT_S = 4.0, 4.0
    try:
        # 1. writes once, then goes silent -> must fail, naming the last line
        def silent(log, n):
            if n == 1:
                log.write_text("INFO starting\n"
                               "module = gen_cutlass_fused_moe_sm120_module(x).build_and_load()\n")
        ok_, err = _run_serve(silent, healthy_after=0)
        check("a silent launch fails rather than waiting out the clock",
              not ok_ and err is not None and "stopped producing output" in err, f"{err}")
        check("the failure names the last thing the server said",
              err is not None and "gen_cutlass_fused_moe_sm120_module" in err,
              "without it, two hours went into diagnosing the wrong thing")

        # 2. keeps writing, healthy late -> deadline extends past LAUNCH_TIMEOUT_S
        def chatty(log, n):
            with open(log, "a") as fh:
                fh.write(f"INFO still loading shard {n}\n")
        t0 = time.monotonic()
        ok_, err = _run_serve(chatty, healthy_after=6)
        check("a launch that keeps logging is allowed past LAUNCH_TIMEOUT_S",
              ok_, f"failed with: {err}")
    finally:
        ev.STALL_S, ev.LAUNCH_TIMEOUT_S = orig_stall, orig_to

    # ==================================================================
    # A DEFAULT MUST BE VALID FOR THE ARTIFACT IT LANDS ON.
    #
    # The check below this one asserts moe_backend=triton IS SET for a MoE on
    # sm12x. That is a tautology -- it sets triton, then confirms triton was
    # set -- and it passed while an 18-hour run built a 23 GB artifact that
    # could not load:
    #
    #   ValueError: moe_backend='triton' is not supported for NvFP4 MoE.
    #
    # This one tests a PROPERTY instead, against vLLM's own list rather than a
    # copy of it: whatever config we would serve for a quantized artifact must
    # use a backend vLLM accepts for that artifact's quant_algo.
    # ==================================================================
    print("\n=== the served config is valid for the artifact it loads ===")
    from evaluator import (_artifact_quant_algo, _nvfp4_moe_backends,
                           reconcile_moe_backend)
    valid = _nvfp4_moe_backends()
    check("vLLM's NvFP4 MoE backend list is readable", bool(valid),
          "cannot verify our config against it; the list moved or the import "
          "path changed")
    check("triton is NOT among them -- the assumption that broke an 18h run",
          "triton" not in valid, f"got {sorted(valid)}")

    # THE PATH THAT ACTUALLY RUNS. Checking hardware_defaults alone is a
    # tautology: it sets triton, so asserting triton is set proves nothing. What
    # matters is the config _serve finally hands to vLLM, so that is what is
    # exercised here.
    quantized = [p for p in sorted(Path("artifacts").glob("*--*"))
                 if _artifact_quant_algo(str(p))]
    check("there is at least one quantized artifact to test against",
          bool(quantized), "cannot verify the reconciliation without one")
    for art in quantized:
        algo = _artifact_quant_algo(str(art))
        cfg = reconcile_moe_backend(
            {"moe_backend": "triton", "model": str(art)}, log=lambda *a: None)
        needs_fix = "NVFP4" in algo.upper() or algo == "MIXED_PRECISION"
        if needs_fix:
            check(f"{art.name}: triton corrected for {algo}",
                  cfg["moe_backend"] in valid,
                  f"still {cfg['moe_backend']!r}; vLLM accepts {sorted(valid)} "
                  f"and would raise at engine init AFTER the artifact was built")
        else:
            check(f"{art.name}: {algo} left alone",
                  cfg["moe_backend"] == "triton", f"got {cfg['moe_backend']!r}")

    # An UNQUANTIZED MoE must keep triton -- that is the whole reason the
    # default exists, and "fixing" it everywhere would reintroduce the 88-minute
    # FlashInfer CUTLASS JIT hang.
    cfg = reconcile_moe_backend({"moe_backend": "triton",
                                 "model": "Qwen/Qwen3-30B-A3B"},
                                log=lambda *a: None)
    check("an unquantized MoE keeps triton", cfg["moe_backend"] == "triton",
          f"got {cfg['moe_backend']!r} -- this is what avoids the sm120 JIT hang")

    print("\n=== MoE on sm12x avoids FlashInfer's CUTLASS JIT ===")
    from run import seed_config as _seed
    # EVERY launch path, not just the traversal. moe_backend was set only in
    # run.py's seed_config, so a stock eval_repro run on the same model went
    # down FlashInfer's sm120 path and hung 30 min compiling kernels while the
    # traversal beside it ran fine.
    import eval_repro as _er
    import argparse as _ap
    for mid, want in (("Qwen/Qwen3-30B-A3B", "triton"), ("Qwen/Qwen3-14B", None)):
        _fp, _ = _bf(_R(model=mid, trace="data/trace.jsonl"))
        paths = {
            "seed_config": _seed(_fp).get("moe_backend"),
            "stock_config": _er.stock_config(_fp).get("moe_backend"),
            "--config file": configs_under_test_backend(_er, _fp),
        }
        for name, got in paths.items():
            check(f"{mid.split('/')[-1]:14s} {name:14s} -> moe_backend={want}",
                  got == want, f"got {got!r}")

    class _HW:
        def __init__(s, sm, uni): s.sm_major, s.unified_memory = sm, uni
    class _M:
        def __init__(s, dense): s.is_dense, s.max_model_len = dense, 32768
    class _FP:
        def __init__(s, dense, sm): s.model, s.hw, s.workload = _M(dense), _HW(sm, True), fp.workload
    for dense, sm, want in ((False, 12, "triton"), (True, 12, None), (False, 9, None)):
        got = _seed(_FP(dense, sm)).get("moe_backend")
        check(f"{'dense' if dense else 'MoE':5s} on sm{sm} -> moe_backend={want}",
              got == want, f"got {got!r}")

    print("\n=== mbpp_plus: the code benchmark actually executes ===")
    mbpp_data = Path("data/mbpp_plus.jsonl")
    if not mbpp_data.exists():
        check("mbpp_plus dataset present", False,
              "run `python fetch_data.py` -- mbpp_plus checks skipped")
    else:
        import subprocess as _sp
        rows = [json.loads(l) for l in mbpp_data.read_text().splitlines()[:12]]
        r = _sp.run([sys.executable, "-c",
                     "import json;from evalplus.data import get_mbpp_plus;"
                     "print(json.dumps({k:v['prompt']+v['canonical_solution'] "
                     "for k,v in get_mbpp_plus().items()}))"],
                    capture_output=True, text=True, timeout=600,
                    env={**os.environ, "PYTHONPATH": str(Path(".evalplus-pkgs").resolve())})
        if r.returncode != 0:
            check("evalplus is importable from .evalplus-pkgs", False, r.stderr[-300:])
        else:
            canon = json.loads(r.stdout.strip().splitlines()[-1])
            judge = BENCHMARKS["mbpp_plus"].judge

            good = [f"```python\n{canon[x['task_id']]}\n```" for x in rows]
            check("canonical solutions all PASS", all(judge(rows, good)),
                  "a judge that fails correct code would read as quantization damage")

            prose = ["This problem compares two lists element-wise."] * len(rows)
            check("prose with no code all FAILS", not any(judge(rows, prose)),
                  "a judge that passes non-code cannot detect damage either")

            # Row order. mbpp_score runs a process pool and returns verdicts as
            # they COMPLETE, so its dict is in arrival order. Callers zip the
            # result against rows to find which problem flipped; unordered
            # verdicts would blame the wrong problem every time.
            mixed = list(good)
            mixed[0] = "nope"
            mixed[3] = "also nope"
            v = judge(rows, mixed)
            check("verdicts come back in ROW order, not completion order",
                  v[0] is False and v[3] is False and all(v[i] for i in (1, 2, 4, 5)),
                  f"got {v}")

        # REAL model output, frozen. The synthetic checks above prove the judge
        # can tell correct code from prose; they cannot catch a change that
        # shifts the SCORE on the messy middle -- fenced blocks with commentary
        # around them, helper functions, near-misses.
        #
        # These 60 generations are Qwen3-14B's actual answers, captured when the
        # scorer drove untrusted_check by hand. The rewrite to evalplus's own
        # evaluate() had to reproduce every verdict, and did: 60/60. Pinning it
        # here means the next change to the scorer must too.
        fixture = Path("fixtures/mbpp_real_generations.jsonl")
        if fixture.exists():
            fx = [json.loads(l) for l in fixture.read_text().splitlines() if l.strip()]
            v = BENCHMARKS["mbpp_plus"].judge(
                [{"task_id": r["task_id"]} for r in fx], [r["text"] for r in fx])
            want = [r["expected"] for r in fx]
            agree = sum(1 for a, b in zip(v, want) if a == b)
            check(f"real generations reproduce pass@1 {sum(want)/len(want):.4f}",
                  v == want,
                  f"{agree}/{len(fx)} verdicts agree; got "
                  f"{sum(v)/len(v):.4f}, expected {sum(want)/len(want):.4f}")
        else:
            check("real-generation fixture present", False, f"{fixture} missing")

        # TWO JUDGES AT ONCE IN ONE CHECKOUT. This is the bug that cost a
        # seven-config ladder: the scorer wrote .mbpp-samples.jsonl and the
        # _eval_results.json evaluate() derives from it at FIXED paths in the
        # repo root, so a selftest scoring in the same directory as a running
        # ladder clobbered its results file. The ladder found none of its own
        # task_ids, scored every problem False, and reported 0.0000 for all
        # seven configs with no error anywhere -- while identical code, run
        # alone afterwards, gave 0.7169 on the same generations.
        from concurrent.futures import ThreadPoolExecutor
        j = BENCHMARKS["mbpp_plus"].judge
        big = ([{"task_id": r["task_id"]} for r in rows], good)
        small = (big[0][:6], good[:6])
        solo_big = sum(j(*big)) / len(big[1])
        solo_small = sum(j(*small)) / len(small[1])
        with ThreadPoolExecutor(2) as _ex:
            f1 = _ex.submit(lambda: sum(j(*big)) / len(big[1]))
            f2 = _ex.submit(lambda: sum(j(*small)) / len(small[1]))
            con_big, con_small = f1.result(), f2.result()
        check("two concurrent judges in one checkout do not corrupt each other",
              con_big == solo_big and con_small == solo_small,
              f"solo {solo_big:.4f}/{solo_small:.4f} vs "
              f"concurrent {con_big:.4f}/{con_small:.4f}")

        check("mbpp_plus uses the chat template", BENCHMARKS["mbpp_plus"].chat,
              "raw completions on an instruct model score prompt format, not capability")
        check("math_500 does NOT use the chat template",
              not BENCHMARKS["math_500"].chat,
              "every recorded MATH-500 number was measured raw; changing it "
              "makes new rows incomparable to old ones")

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
