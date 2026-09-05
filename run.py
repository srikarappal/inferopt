"""inferopt CLI.

    python run.py trace    --out data/trace.jsonl            # build a workload trace
    python run.py optimize --model Qwen/Qwen3.5-9B \\
                           --trace data/trace.jsonl \\
                           --ttft-p99 500 --itl-p99 30 --allow-loss 0.03

Stage 1.2 runs AIConfigurator against the nearest supported member of the GPU's
architecture family, corrects the result with a roofline for the real hardware,
and hands the DAG a seed. It also reports when an SLO is arithmetically
unreachable -- which no amount of measurement can determine faster.

HISTORY

  gpu_memory_utilization is 0.75 on unified-memory parts, not the usual 0.90.
  There the fraction is of SYSTEM memory, and the CPU, the benchmark client and
  the OS all compete for the remainder. 0.90 of 122GB left ~1.6GB of headroom
  and ran the box to the edge of the OOM killer.

  enforce_eager is on in the seed, deliberately. torch.compile costs ~30-40 min
  on this hardware and is keyed on shapes, so changing max_num_seqs or
  max_model_len -- which the traversal does constantly -- invalidates it. Paying
  it per node would cost more than the entire search. Compilation is tested ONCE
  as its own node, against the accumulated config.

  free_port exists because production is on the same box. The OCR server sits on
  8813 and vLLM defaults to 8000.

  Stage 1.3 stops the run if the seed cannot meet the SLO. Everything downstream
  is a ratio against that measurement, so a starting point at goodput ~0 makes
  every later comparison a comparison of noise.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from calibration import STORE
from fingerprint import NodeMeasurement
from request import InferOptRequest, build_fingerprint
from traverse import report, traverse

# Benchmarks run once on the seed so the frontier has a quality axis. Lossless
# nodes inherit these; lossy nodes re-measure.
BASELINE_BENCHMARKS = ["math_500"]


def cmd_trace(args) -> int:
    """Build a replayable trace from a ShareGPT dump.

    The benchmark needs real prompt TEXT, not just token counts -- the
    fingerprint can be derived from counts, but you cannot replay a workload you
    only have statistics about.
    """
    src = Path(args.sharegpt)
    if not src.exists():
        print(f"  {src} not found. Point --sharegpt at a ShareGPT V3 dump.")
        return 1
    rng = random.Random(args.seed)
    convs = json.loads(src.read_text())
    prompts = [t["value"] for c in convs for t in c.get("conversations", [])[:1]
               if 64 <= len(t.get("value") or "") <= 16000]
    rng.shuffle(prompts)
    prompts = prompts[: args.n]

    # REAL shared prefixes, not just labels.
    #
    # This used to tag 30% of rows with a prefix_id and change nothing about the
    # prompt text. The fingerprint then reported prefix_overlap=32% while the
    # actual prompts shared 1.2% of their leading characters, so the
    # prefix_caching node was gated on a number that described nothing. Worse,
    # the load generator replays the same prompt indices in every phase, so what
    # the 4x prefix-caching win actually measured was DUPLICATE-REQUEST caching
    # -- identical full prompts served again from a warm cache -- not prefix
    # sharing. Production traffic does not do that; it shares system prompts.
    rows, t = [], 0.0
    shared = {
        f"sys{i}": (
            f"You are a careful assistant operating under policy set {i}. "
            f"Answer accurately and concisely. If you are unsure, say so rather "
            f"than guessing. Do not fabricate citations, figures or quotations. "
            f"Prefer concrete examples over abstract description. When a question "
            f"has several reasonable readings, state which one you are answering. "
            f"Keep responses focused on what was asked.\n\n" * args.prefix_tokens_mult
        ) for i in range(5)
    }
    for p in prompts:
        t += rng.expovariate(args.qps)
        pid = rng.choice(list(shared)) if rng.random() < args.prefix_share else None
        rows.append({
            "prompt": (shared[pid] + p) if pid else p,
            "input_tokens": (len(shared[pid]) if pid else 0) // 4 + len(p) // 4,
            "output_tokens": int(rng.lognormvariate(5.4, 0.6)),
            "arrival_ts": round(t, 4),
            "prefix_id": pid,
            "adapter_id": None,
            "temperature": 0.0,
        })
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"  wrote {args.out}  ({len(rows)} requests, ~{args.qps} qps, "
          f"{args.prefix_share:.0%} prefix sharing)")
    return 0


def seed_config(fp) -> dict:
    """Conservative starting point, used until a predictor covers this hardware.

    gpu_memory_utilization is 0.75 on unified-memory parts rather than the usual
    0.90: there the fraction is of SYSTEM memory, and the CPU, the benchmark
    client and the OS are all competing for the remainder. 0.90 of 122GB left
    ~1.6GB of headroom on this box and ran the machine to the edge of the OOM
    killer.
    """
    need = fp.workload.p999_input_tokens + fp.workload.p99_output_tokens
    cfg = {
        "max_num_seqs": 256,
        "max_model_len": min(fp.model.max_model_len, max(4096, ((need // 1024) + 2) * 1024)),
        "enable_prefix_caching": False,
        "enable_chunked_prefill": False,
        # Eager on purpose. torch.compile costs ~30 min on this hardware and is
        # keyed on shapes, so changing max_num_seqs or max_model_len -- which
        # the traversal does constantly -- invalidates it. Paying that per node
        # would cost more than the entire search. Compilation is tested ONCE, as
        # its own node, against the accumulated config.
        "enforce_eager": True,
    }

    # Hardware-required flags come from ONE place, so every launch path gets
    # them -- see evaluator.hardware_defaults. They go UNDER the seed, so
    # anything set above wins.
    from evaluator import hardware_defaults
    cfg = {**hardware_defaults(fp), **cfg}
    return cfg


def prefetch_weights(model: str, log=print) -> None:
    """Download the checkpoint BEFORE the traversal, outside any launch timeout.

    A launch timeout is meant to answer "did this config start?". It was also
    answering "did 61 GB arrive over the network?", which is not a property of
    the config and not something a launch budget should cover. The first
    Qwen3-30B-A3B run spent 40 of its 120 allotted minutes downloading, inside
    the window, before it had measured anything at all.

    Doing it here also makes the wait legible: huggingface_hub prints progress,
    where a download inside vLLM's startup is invisible -- the server logs
    "Starting to load model" and then says nothing for forty minutes, which is
    indistinguishable from a hang.

    A local path is left alone. Failure is NOT fatal: vLLM will fetch it itself,
    and refusing to start a traversal because a pre-fetch failed would be a
    worse outcome than a slow first launch.
    """
    if Path(model).exists():
        log(f"  weights   local path, nothing to fetch")
        return
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        log(f"  weights   huggingface_hub not available; vLLM will fetch on launch")
        return
    log(f"  weights   pre-fetching {model} (outside the launch timeout) ...")
    try:
        import time as _t
        t0 = _t.time()
        p = snapshot_download(model, allow_patterns=[
            "*.safetensors", "*.json", "*.txt", "*.model", "*.py"])
        n = sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file())
        log(f"  weights   ready, {n/1e9:.1f} GB in {(_t.time()-t0)/60:.1f} min -> {p}")
    except Exception as e:
        log(f"  weights   pre-fetch failed ({type(e).__name__}: {e});")
        log(f"            vLLM will fetch on first launch, which may exceed its timeout")


def free_port(start: int) -> int:
    """First free port at or above `start`. The OCR server sits on 8813 and
    vLLM's default is 8000, so a fixed port is a collision waiting to happen."""
    import socket
    for p in range(start, start + 40):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    raise RuntimeError(f"no free port in {start}..{start+40}")


def _incumbent_node(res) -> str | None:
    """Which frontier node IS the incumbent.

    The traversal returns the incumbent as a CONFIG dict, while finalist_curves
    is keyed by node id, so joining them needs a lookup rather than a name. The
    last kept trial whose config matches is the one that was deployed.
    """
    for t in reversed(res.trials):
        if t.kept and t.config == res.incumbent:
            return t.node_id
    # Fall back to the best kept trial; a config can pick up later edits.
    kept = [t for t in res.trials if t.kept]
    return max(kept, key=lambda t: t.goodput).node_id if kept else None


def cmd_optimize(args) -> int:
    req = InferOptRequest(
        model=args.model, trace=args.trace,
        qps=args.qps,
        ttft_p99_ms=args.ttft_p99, itl_p99_ms=args.itl_p99,
        allow_loss=args.allow_loss, lossless_tolerance=args.lossless_tolerance,
        adapters=args.adapter or [],
        budget_minutes=args.budget_minutes,
    )
    fp, slo = build_fingerprint(req)

    print(f"\n  model     {fp.model.id}  ({fp.model.n_params_b}B {fp.model.attention_type}, "
          f"{fp.model.weight_gb:.0f}GB weights, {fp.model.kv_bytes_per_token/1024:.0f}KB KV/token)")
    print(f"  hardware  {fp.hw.gpu_name} x{fp.hw.gpu_count}  cc{fp.hw.compute_capability}  "
          f"{fp.hw.memory_gb:.0f}GB{' unified' if fp.hw.unified_memory else ''}  "
          f"{fp.hw.memory_bandwidth_gb_s:.0f} GB/s")
    # Where qps CAME FROM, not just its value: it is the denominator of every
    # replica count in the report, and a stated rate and a measured one are
    # different claims. calibration.py labels accept_band the same way and for
    # the same reason -- a number resting on a default must never look measured.
    qsrc = "stated" if args.qps else "from trace arrival_ts"
    print(f"  workload  {fp.workload.n_requests} reqs, in p99 {fp.workload.p99_input_tokens}, "
          f"out mean {fp.workload.mean_output_tokens:.0f}, "
          f"{fp.workload.request_rate_qps:.1f} qps [{qsrc}], "
          f"prefix {fp.workload.prefix_overlap:.0%}")
    print(f"  slo       ttft_p99 {slo.ttft_p99_ms}ms  itl_p99 {slo.itl_p99_ms}ms  "
          f"allow_loss {slo.quality_budget}")

    kv_one = fp.model.kv_bytes_per_token * fp.model.max_model_len / 1e9
    print(f"  headroom  one max-length sequence needs {kv_one:.1f}GB of KV\n")

    # BEFORE any launch, so the download is not inside a launch timeout. The
    # first Qwen3-30B-A3B run spent 40 of its 120 allotted minutes here, invisibly
    # -- vLLM logs "Starting to load model" and then says nothing while it
    # fetches, which is indistinguishable from a hang.
    prefetch_weights(fp.model.id)
    print()

    from evaluator import VllmEvaluator
    from fingerprint import Context

    cfg = seed_config(fp)
    if args.skip_predict:
        print(f"  stage 1.2 skipped by --skip-predict")
    else:
        from predictor import describe, predict
        try:
            pred = predict(fp, slo)
            describe(pred)
            if pred.seed_config:
                # The predictor chooses the SHAPE (batch size, parallelism); the
                # conservative defaults keep the safety rails it does not model
                # (unified-memory utilisation, eager mode, right-sized context).
                cfg.update(pred.seed_config)
                cfg.pop("tensor_parallel_size", None) if fp.hw.gpu_count == 1 else None
            if not pred.feasible:
                print(f"\n  proceeding anyway -- the roofline is a LOWER BOUND, so measurement "
                      f"can confirm it but never beat it. Expect stage 1.3 to report goodput 0 "
                      f"and stop, which costs one launch and yields a real measured ITL.")
        except Exception as e:
            print(f"  stage 1.2 unavailable ({type(e).__name__}: {e})")
            print(f"  falling back to the conservative seed")
    print(f"\n  seed      {json.dumps(cfg)}\n")

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    journal = run_dir / "trials.jsonl"
    journal.write_text("")          # truncate once, here; traverse only appends

    port = free_port(args.port)
    from provenance import banner, provenance, trial_stamp
    stamp = trial_stamp(fp, args.trace, slo)
    meta = provenance(ap_ref[0], args, fp, extra={
        "port": port,
        "seed_config": cfg,
        "dag": args.dag,
        "slo": slo.model_dump(),
        "trial_stamp": stamp,
    })
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(banner(meta, run_dir / "run_meta.json"))
    if port != args.port:
        print(f"  port      {args.port} is taken; using {port}")
    ev = VllmEvaluator(fp, slo, args.trace, str(run_dir), gpu=args.gpu, port=port)

    ctx = Context(fingerprint=fp, slo=slo, incumbent=cfg)
    baseline, curve, operating_L = None, [], None
    if not args.skip_stage13:
        print("  stage 1.3 measuring the seed config")
        # Quality is measured HERE, not only at lossless_complete. Every point
        # on the final frontier needs an accuracy coordinate, and a lossless
        # node inherits this one -- so without a baseline score the whole
        # quality axis is empty.
        # levels=SWEEP_LEVELS makes this one launch do both jobs: measure the
        # baseline AND characterise capacity across the full geometric range.
        from evaluator import SWEEP_LEVELS
        sweep_off = args.skip_sweep or args.fixed_concurrency
        t = ev.measure(cfg, probes=["goodput", "equivalence", "quality"],
                       benchmarks=BASELINE_BENCHMARKS, node_id="stage_1_3",
                       levels=None if sweep_off else SWEEP_LEVELS,
                       fixed_concurrency=args.fixed_concurrency)
        if t.diagnostics.get("launch_error"):
            # A launch failure is NOT an SLO failure. Reporting inf ttft as
            # "does not satisfy the SLO" sends you to tune a threshold when the
            # server never started.
            print(f"\n  the server did not start -- this is a launch failure, not an "
                  f"SLO problem, and nothing about the SLO or the seed config will "
                  f"fix it.\n\n    {t.diagnostics['launch_error']}\n")
            tail = (t.diagnostics.get("stderr_tail") or "").strip().splitlines()[-6:]
            for line in tail:
                print(f"    {line[:130]}")
            print(f"\n  full log: {run_dir}/launches/*/server.log")
            print(f"  if it says 'not healthy in Ns', the launch was still working when "
                  f"the clock ran out -- raise INFEROPT_LAUNCH_TIMEOUT_S (currently "
                  f"{__import__('evaluator').LAUNCH_TIMEOUT_S:.0f}s).")
            return 1
        if not t.slo_ok:
            print(f"\n  the seed config MEASURED ttft_p99 {t.ttft_p99_ms:.0f}ms against an "
                  f"SLO of {slo.ttft_p99_ms}ms, so its goodput is ~0. Everything "
                  f"downstream is a ratio against this, so every comparison would be "
                  f"against noise. Relax the SLO, or lower the offered load "
                  f"({fp.workload.request_rate_qps:.1f} qps at concurrency "
                  f"{fp.workload.max_concurrency}).")
            return 1
        ctx.incumbent_metrics = NodeMeasurement(
            goodput=t.goodput, ttft_p99_ms=t.ttft_p99_ms, itl_p99_ms=t.itl_p99_ms,
            quality={}, config=cfg)
        baseline = t
        ctx.quality_baseline.update(t.quality)

        # THE BASELINE IS THE RUN'S ANCHOR. Every percentage the traversal
        # prints, and every number in the frontier, is a ratio against it.
        # It used to be summarised in one console line and persisted nowhere:
        # stage 1.3 runs before traverse(), so it missed the journal AND
        # result.json, and after run four finished there was no way to say what
        # "+307%" was 307% OF. It had to be inferred from a previous run.
        #
        # So it is now written THREE times -- console block, journal line 1,
        # result.json["baseline"] -- because losing it makes the entire run
        # uninterpretable, and it costs nothing to keep.
        t.provenance = dict(stamp)
        journal.write_text(json.dumps(t.__dict__, default=str) + "\n")
        d = t.diagnostics or {}
        miss = lambda v, lim: ("" if not lim else
                               ("  <- MISSES the %g ms SLO" % lim if v > lim else "  (within SLO)"))
        print(f"\n  {'-'*68}")
        print(f"  BASELINE  stage 1.3, the seed config")
        print(f"            every percentage this run prints is against these numbers")
        print(f"  {'-'*68}")
        print(f"    goodput          {t.goodput:9.1f} tok/s   <- the objective")
        print(f"                     {d.get('goodput_req_s', 0):9.2f} req/s   "
              f"same number, the unit vLLM's own benchmark reports")
        print(f"    throughput       {d.get('throughput', float('nan')):9.1f} tok/s   "
              f"(goodput's ceiling)")
        print(f"    ttft p99         {t.ttft_p99_ms:9.0f} ms"
              f"{miss(t.ttft_p99_ms, slo.ttft_p99_ms)}")
        print(f"    itl p99          {t.itl_p99_ms:9.0f} ms"
              f"{miss(t.itl_p99_ms, slo.itl_p99_ms)}")
        print(f"    slo attainment   {d.get('slo_attainment', 0):9.0%}   "
              f"of requests met both targets")
        print(f"    memory           {t.memory_gb:9.1f} GB")
        print(f"    config           {json.dumps(cfg, default=str)}")
        print(f"    persisted to     {journal}  (line 1)  and  {run_dir}/result.json")
        print(f"  {'-'*68}\n")

        # --- capacity sweep -------------------------------------------------
        # Concurrency is not a knob. By Little's Law (L = lambda x W) it is an
        # OUTCOME of arrival rate and service time, and goodput as a function of
        # it has a peak: past capacity, extra concurrency only pushes requests
        # over the deadline, so goodput falls. That peak is simultaneously the
        # maximum goodput, the best concurrency, and the sustainable req/s that
        # divides into demand to size a fleet.
        #
        # Run four judged every node at L=30, from `int(qps*2)` -- a formula
        # that does not even typecheck, since qps is 1/time and concurrency is
        # dimensionless. Everything was measured at an arbitrary point on an
        # unnamed curve.
        # The sweep happened INSIDE the stage 1.3 launch, via levels=SWEEP_LEVELS.
        # It used to be a second launch of the identical config: ~9 wasted minutes,
        # and two independently-derived operating points that could disagree with
        # each other. The baseline Trial now carries both the curve and the peak.
        curve = baseline.curve
        operating_L = baseline.concurrency
        if args.fixed_concurrency:
            print(f"  concurrency PINNED at {args.fixed_concurrency} "
                  f"(--fixed-concurrency): open-loop driver, no sweep, no bracket.\n"
                  f"  This is run four's instrument. Numbers are comparable to that "
                  f"run and NOT to a swept run.\n")
        if curve:
            pk = max(curve, key=lambda m: m["goodput"])
            capacity_toks = pk["goodput"]
            print(f"  {'-'*68}")
            print(f"  CAPACITY  one replica, seed config")
            print(f"  {'-'*68}")
            for m in sorted(curve, key=lambda m: m["concurrency"]):
                mark = "  <- peak" if m["concurrency"] == operating_L else ""
                print(f"    L={m['concurrency']:<5d} goodput {m['goodput']:8.1f} tok/s  "
                      f"thru {m['throughput']:8.1f}  ttft_p99 {m['ttft_p99_ms']:7.0f}ms  "
                      f"slo {m['slo_attainment']:4.0%}{mark}")
            print()
            print(f"    peak goodput     {capacity_toks:9.1f} tok/s  at concurrency {operating_L}")
            print(f"    sustainable      {pk.get('goodput_req_s', 0):9.2f} req/s")
            demand = fp.workload.request_rate_qps * fp.workload.mean_output_tokens
            print(f"    demand           {demand:9.1f} tok/s  "
                  f"({fp.workload.request_rate_qps:.1f} req/s x "
                  f"{fp.workload.mean_output_tokens:.0f} tokens)")
            import math as _m
            print(f"    replicas needed  {_m.ceil(demand/max(1e-9, capacity_toks)):9d}   "
                  f"= demand / capacity, to serve this workload at the SLO")
            print(f"    nodes start bracketed around L={operating_L}; the bracket "
                  f"moves with the incumbent")
            print(f"  {'-'*68}\n")

    dag = json.loads(Path(args.dag).read_text())
    res = traverse(dag, ctx, ev, lossless_only=args.lossless_only,
                   journal=journal, baseline=baseline,
                   concurrency=operating_L,
                   fixed_concurrency=args.fixed_concurrency,
                   provenance=stamp,
                   force_benchmarks=(BASELINE_BENCHMARKS
                                     if args.quality_every_node else None))

    # Full sweep on the finalists. The traversal ranks configs at one operating
    # point, which is enough to CHOOSE between them -- most goodput curves sit
    # uniformly above or below each other rather than crossing. But the winner's
    # own curve is what sets capacity and therefore the replica count, so the
    # configs anyone might actually deploy get measured properly.
    finalist_curves = {}
    if not args.skip_sweep and not args.no_finalist_sweep and not args.fixed_concurrency:
        finalists = res.frontier()[: args.finalists]
        if finalists:
            print(f"\n  sweeping {len(finalists)} frontier finalist(s) for capacity\n")
        for t in finalists:
            print(f"    {t.node_id}")
            try:
                c, pk = ev.capacity(t.config, f"finalist-{t.node_id}")
                finalist_curves[t.node_id] = {"curve": c, "peak": pk}
                t.curve = c
                t.concurrency = pk["concurrency"]
            except Exception as e:
                print(f"      sweep failed ({type(e).__name__}: {e}); "
                      f"keeping the traversal measurement")

    _inc = _incumbent_node(res)
    report(res, demand_tok_s=fp.workload.request_rate_qps * fp.workload.mean_output_tokens,
           incumbent_curve=(finalist_curves.get(_inc) or {}).get('curve'))

    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "result.json"
    out.write_text(json.dumps({
        **meta,
        "fingerprint": fp.model_dump(),
        "baseline": (baseline.__dict__ if baseline else None),
        # NAMED FOR WHAT IT IS. This was "capacity_curve", which reads as the
        # run's capacity but holds the STAGE 1.3 SEED's sweep -- a config that
        # was replaced before the run ended. On the MoE run the two disagree
        # completely: the seed collapses from 41.8 goodput at L=2 to 7.2 at L=8,
        # while the incumbent peaks at 65.6 AT L=8. Reading the seed's curve as
        # the result would say the deployed config runs at 9x less than it does.
        "seed_capacity_curve": curve,
        "operating_concurrency": operating_L,
        "finalist_curves": finalist_curves,
        # The curve of the config actually chosen, hoisted out of
        # finalist_curves so the thing that sets the replica count is not buried
        # one level down under a node id the reader has to know to look up.
        "incumbent_capacity_curve": (
            (finalist_curves.get(_incumbent_node(res)) or {}).get("curve") or []),
        "incumbent_peak": (
            (finalist_curves.get(_incumbent_node(res)) or {}).get("peak") or {}),
        "demand_tok_s": fp.workload.request_rate_qps * fp.workload.mean_output_tokens,
        "incumbent": res.incumbent,
        "frontier": [t.__dict__ for t in res.frontier()],
        "trials": [t.__dict__ for t in res.trials],
        "visited": res.visited, "skipped": res.skipped,
        "launches": res.launches, "minutes": res.minutes,
        "stopped_early": res.stopped_early,
    }, indent=2, default=str))
    print(f"\n  wrote {out}")
    return 0


ap_ref: list = [None]      # so cmd_optimize can read argument DEFAULTS for the record


def main() -> int:
    ap = argparse.ArgumentParser(prog="inferopt")
    ap_ref[0] = ap
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("trace", help="build a replayable workload trace")
    t.add_argument("--sharegpt", default="../qwen-serve-opt/data/ShareGPT_V3_unfiltered_cleaned_split.json")
    t.add_argument("--out", default="data/trace.jsonl")
    t.add_argument("--n", type=int, default=800)
    t.add_argument("--qps", type=float, default=16.0)
    t.add_argument("--prefix-share", type=float, default=0.30)
    t.add_argument("--prefix-tokens-mult", type=int, default=8,
                   help="repeats of the shared system prompt; 8 gives ~800 tokens, "
                        "a realistic system-prompt length")
    t.add_argument("--seed", type=int, default=0)
    t.set_defaults(fn=cmd_trace)

    o = sub.add_parser("optimize", help="fingerprint, then traverse the DAG")
    o.add_argument("--model", required=True)
    o.add_argument("--trace", required=True)
    o.add_argument("--ttft-p99", type=float, default=None)
    o.add_argument("--itl-p99", type=float, default=None)
    o.add_argument("--quality-every-node", action="store_true",
                   help="score the accuracy benchmark on EVERY config, not just "
                        "the baseline and the lossy nodes. Slower, and needed "
                        "for a method comparison: an inherited score is an "
                        "assumption, and the comparison's claim is about the "
                        "configs each method actually shipped.")
    o.add_argument("--qps", type=float, default=None,
                   help="arrival rate in requests/second. Overrides the rate "
                        "implied by the trace's arrival_ts, which is the "
                        "fallback. Use this when you know your target rate and "
                        "your trace has no real timestamps -- fabricating "
                        "arrival_ts to express a rate you already know is "
                        "circular, and omitting it silently zeroed every "
                        "replica count.")
    o.add_argument("--allow-loss", type=float, default=None,
                   help="quality budget for the LOSSY branch, e.g. 0.1")
    o.add_argument("--lossless-tolerance", type=float, default=0.03,
                   help="how far the eval may move across the LOSSLESS branch before it "
                        "is flagged. Default 0.03. A lossless step should not move the "
                        "eval at all -- this is a defect threshold, not a budget.")
    o.add_argument("--adapter", action="append")
    o.add_argument("--dag", default="dag/llm.json")
    o.add_argument("--gpu", default="0")
    o.add_argument("--port", type=int, default=8100,
                   help="first port to try; the next free one is used if taken")
    o.add_argument("--run-dir", default="runs/latest")
    o.add_argument("--budget-minutes", type=int, default=180)
    o.add_argument("--skip-predict", action="store_true",
                   help="skip stage 1.2 and use the conservative seed")
    o.add_argument("--fixed-concurrency", type=int, default=None, metavar="N",
                   help="measure every node at exactly N in-flight requests using "
                        "the open-loop driver, with no sweep and no bracket. This is "
                        "the instrument run four used (N=30). Use it to reproduce a "
                        "previous run, or when the production arrival rate is known "
                        "and you want that specific operating point rather than the "
                        "peak of the curve.")
    o.add_argument("--no-finalist-sweep", action="store_true",
                   help="skip the per-finalist capacity sweeps at the end")
    o.add_argument("--finalists", type=int, default=3,
                   help="how many frontier configs get a full capacity sweep")
    o.add_argument("--skip-sweep", action="store_true",
                   help="skip the stage 1.3b capacity sweep and measure at the "
                        "fingerprint's concurrency. Faster, but every goodput number "
                        "is then a point on an unnamed curve.")
    o.add_argument("--lossless-only", action="store_true",
                   help="stop at the lossless branch; do not enter the lossy nodes. "
                        "The frontier still includes everything measured.")
    o.add_argument("--skip-stage13", action="store_true",
                   help="skip measuring the seed; the first node is then kept unconditionally")
    o.set_defaults(fn=cmd_optimize)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
