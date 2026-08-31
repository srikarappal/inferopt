"""Score a model on an eval set, N times, under one or more serving configs.

    # stock model, no optimizations -- the BEFORE number
    python eval_repro.py --model Qwen/Qwen3-14B --benchmark math_500 --repeats 3

    # a quantized checkpoint, same eval -- the AutoQuantize validation run
    python eval_repro.py --model artifacts/Qwen__Qwen3-14B--nvfp4 --benchmark math_500

    # a config inferopt produced -- the AFTER number
    python eval_repro.py --model Qwen/Qwen3-14B --config runs/ninth/result.json

    # both ends of a run, head to head
    python eval_repro.py --model Qwen/Qwen3-14B --from-run runs/ninth

Two jobs, and neither is useful without the other.

SCORE A MODEL. Quality on the eval set is the only thing that says whether an
optimization did a good job. This runs it standalone -- any model, any config,
before or after inferopt, stock weights or quantized.

KNOW WHAT THE SCORE CAN RESOLVE. Repeating the same measurement gives the eval's
own resolution limit. Without it a real quantization loss cannot be told apart
from run-to-run movement, and this project already made that mistake: MATH-500
moved +0.03 across the LOSSLESS branch in two consecutive runs (0.73->0.76,
0.71->0.74), on nodes that only change launch flags. That drift was then used as
a "tolerance" to widen the lossy gate -- backwards twice over. A lossless step
moving the eval is a defect to explain, not a budget to spend.

With more than one config the report separates:

    within-config spread    same weights, same flags, same problems, so anything
                            that moves is execution non-determinism
    between-config diff     what the config change actually did

    between <= within       the eval cannot resolve it at this sample size.
                            Raise --n, or serialise with --concurrency 1.
    between >  within       the difference is real and attributable.

The leading suspect for within-config movement is batch composition: the probe
batches at --concurrency, and which requests share a forward pass changes the
floating-point reduction order in attention and GEMM accumulation. Logits shift,
and a token near a decision boundary flips. Measured flip rate on this rig is
0.44% per token, and on a reasoning benchmark ONE flipped token early in a chain
changes the final boxed answer entirely. `--concurrency 1` serialises the probe
so there is a single batch composition; if it still moves, look elsewhere.

BOTH HALVES, ONE LAUNCH. Accuracy alone cannot say whether an optimization was
worth it, so each config is also measured for goodput, throughput, TTFT p99 and
ITL p99 -- on the workload TRACE at a fixed concurrency, using the same
closed-loop instrument the traversal uses, so a quantized checkpoint scored here
is directly comparable to a config measured there.

The accuracy probe cannot supply those numbers itself: it issues NON-STREAMING
requests, where ttft is set equal to the whole request latency, so every timing
it collects is meaningless as a TTFT. It also runs the benchmark's prompts rather
than production traffic. --no-serving skips it.

Every generation is written to disk. A summary statistic cannot be reopened, and
the first question after "the score moved" is "moved how".
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from pathlib import Path


def stock_config(fp) -> dict:
    """vLLM defaults, plus only what this hardware needs in order to boot.

    Deliberately NOT right-sized. This is the BEFORE number, and applying any of
    the optimizations under test would make it a better baseline than the one a
    user actually starts from, which flatters every later comparison.

    The one exception is unified memory, where gpu_memory_utilization is a
    fraction of SYSTEM memory that the CPU also competes for: 0.90 there leaves
    ~1.6GB of headroom on a 122GB box and runs it to the edge of the OOM killer.
    That is a requirement to start at all, not a tuning decision, and it is
    printed so it is never mistaken for one.
    """
    return {"gpu_memory_utilization": 0.75} if fp.hw.unified_memory else {}


def load_config(path: str) -> dict:
    """A raw config dict, or the incumbent out of an inferopt result.json."""
    d = json.loads(Path(path).read_text())
    return d["incumbent"] if "incumbent" in d else d


def configs_under_test(a, fp) -> list[tuple[str, dict]]:
    """What to serve. No implicit default run -- stock unless told otherwise."""
    if a.from_run:
        res = Path(a.from_run) / "result.json"
        if not res.exists():
            raise SystemExit(f"  {res} not found")
        r = json.loads(res.read_text())
        before = (r.get("baseline") or {}).get("config") or stock_config(fp)
        return [("A before", before), ("B after", r["incumbent"])]
    if a.config:
        return [(f"cfg{i+1}", load_config(p)) for i, p in enumerate(a.config)]
    return [("stock", stock_config(fp))]


def score_once(ev, model, rows, bench, concurrency, prompt=None):
    """One pass over the problem list. Returns (score, per-item verdicts, texts).

    JUDGING IS NOT DONE HERE. It used to be: this function picked between
    RULER's rule and MATH-500's by sniffing whether the row had an "answers"
    key, which was a third copy of logic that already existed in quality.py --
    the same duplication that once put MATH-500's prompt in two places, let them
    disagree, and killed a traversal nine launches in with a KeyError.

    Now the benchmark owns its judge and both callers use it. That is also what
    makes a code benchmark possible at all: MBPP+'s judge shells out to a
    sandboxed runner, and no amount of row-sniffing here could reproduce it.
    """
    import httpx

    from evaluator import _one

    prompt = prompt or bench.prompt
    prompts = [prompt(r) for r in rows]

    async def go_all():
        sem = asyncio.Semaphore(concurrency)
        async with httpx.AsyncClient(timeout=900.0) as c:
            async def go(p):
                async with sem:
                    return await _one(c, ev.base_url, model, p, bench.max_tokens, stream=False)
            return list(await asyncio.gather(*[go(p) for p in prompts]))

    outs = asyncio.run(go_all())
    texts = [o.text for o in outs]
    verdicts = bench.judge(rows, texts)
    return sum(verdicts) / len(verdicts), verdicts, texts


def _answer_of(row) -> object:
    """What the dump records as 'the right answer'.

    Code benchmarks have no answer string -- correctness is whether the tests
    passed -- so the task_id is recorded instead, which is what you need to look
    the problem up.
    """
    for k in ("answer", "answers", "task_id"):
        if k in row:
            return row[k]
    return None


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="eval_repro",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True,
                    help="HF id or local path, including a quantized checkpoint")
    ap.add_argument("--benchmark", default="math_500",
                    help="math_500 | mbpp_plus | ruler_multineedle (see quality.py). "
                         "mbpp_plus EXECUTES generated code in a subprocess -- see "
                         "mbpp_score.py for the isolation boundary.")
    ap.add_argument("--n", type=int, default=100, help="problems to score")
    ap.add_argument("--repeats", type=int, default=3,
                    help=">1 measures the eval's own resolution limit")
    ap.add_argument("--concurrency", type=int, default=32,
                    help="probe concurrency. 1 serialises it, testing whether batch "
                         "composition explains any movement.")
    ap.add_argument("--from-run", default=None,
                    help="an inferopt run dir: scores its BEFORE and AFTER configs")
    ap.add_argument("--config", action="append", default=None,
                    help="a config JSON, or an inferopt result.json. Repeatable.")
    ap.add_argument("--trace", default="data/trace_shared.jsonl",
                    help="only used to build the hardware/model fingerprint")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--serving-concurrency", type=int, default=30,
                    help="in-flight requests for the serving measurement. 30 matches "
                         "--fixed-concurrency in the traversal runs, so the numbers are "
                         "directly comparable.")
    ap.add_argument("--no-serving", action="store_true",
                    help="score accuracy only, skip the goodput/TTFT/ITL measurement")
    ap.add_argument("--run-dir", default="runs/eval")
    a = ap.parse_args()

    from evaluator import VllmEvaluator
    from quality import BENCHMARKS, _load
    from request import InferOptRequest, build_fingerprint
    from run import free_port

    if a.benchmark not in BENCHMARKS:
        raise SystemExit(f"  unknown benchmark {a.benchmark!r}; have {', '.join(BENCHMARKS)}")

    fp, slo = build_fingerprint(InferOptRequest(
        model=a.model, trace=a.trace, ttft_p99_ms=500, itl_p99_ms=250))
    tests = configs_under_test(a, fp)
    bench = BENCHMARKS[a.benchmark]
    rows = _load(a.benchmark, a.n)

    # ONE prompt builder for the filter, the generation and the flips dump.
    # A code benchmark on an instruct model needs the chat template applied (see
    # _chat_wrapper); measuring the filter against untemplated text while
    # generating from templated text would under-count the prompt by the
    # template's own tokens.
    from quality import _chat_wrapper
    prompt = _chat_wrapper(bench.prompt, a.model) if bench.chat else bench.prompt

    # Every config is scored on the IDENTICAL problem list. A traversal filters
    # by max_model_len, and two configs can differ there -- scoring them on
    # separately-filtered sets manufactures a difference belonging to neither.
    limits = [c["max_model_len"] for _, c in tests if c.get("max_model_len")]
    if limits:
        budget = min(limits) - bench.max_tokens
        keep = [r for r in rows if len(prompt(r)) // 4 < budget]
        if len(keep) < len(rows):
            print(f"  dropping {len(rows)-len(keep)} problems over the tightest config's "
                  f"{budget}-token budget, so all configs see the same set")
        rows = keep
    if not rows:
        raise SystemExit("  no problems fit the served context")

    print(f"\n  model      {a.model}")
    print(f"  benchmark  {a.benchmark}  ({len(rows)} problems, {bench.metric}, "
          f"{bench.max_tokens} max tokens)")
    print(f"  design     {len(tests)} config(s) x {a.repeats} repeat(s), "
          f"concurrency {a.concurrency}")
    for name, cfg in tests:
        shown = json.dumps({k: v for k, v in cfg.items() if k != "model"})
        print(f"    {name:12s} {shown if cfg else '(vLLM defaults)'}"[:100])
    if any(c == stock_config(fp) for _, c in tests) and fp.hw.unified_memory:
        print(f"    note: gpu_memory_utilization 0.75 is a boot requirement on this "
              f"unified-memory part, not a tuning choice")
    print()

    port = free_port(a.port)
    ev = VllmEvaluator(fp, slo, a.trace, a.run_dir, gpu=a.gpu, port=port)
    outdir = Path(a.run_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    from provenance import banner, provenance
    meta = provenance(ap, a, fp, extra={
        "port": port,                       # free_port may differ from --port
        "configs": {name: cfg for name, cfg in tests},
        "benchmark": {"name": a.benchmark, "metric": bench.metric,
                      "max_tokens": bench.max_tokens,
                      "problems_scored": len(rows)},
    })
    (outdir / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(banner(meta, outdir / "run_meta.json") + "\n")
    out: dict[str, dict] = {}
    keep_v: dict[str, list] = {}
    keep_t: dict[str, list] = {}

    for name, cfg in tests:
        tag = name.split()[0]
        print(f"  --- {name} ---")
        scores, verdicts, texts = [], [], []
        with ev._serve({**cfg, "model": a.model}, f"eval-{tag}") as model:
            for i in range(a.repeats):
                s, v, t = score_once(ev, model, rows, bench, a.concurrency, prompt)
                scores.append(s); verdicts.append(v); texts.append(t)
                print(f"      repeat {i+1}/{a.repeats}   {s:.4f}   ({sum(v)}/{len(v)})")
            # SERVING METRICS -- the traversal's exact instrument, not a
            # lookalike. serving_metrics() is the same function run.py's
            # --fixed-concurrency path calls: open loop at the trace's arrival
            # rate with a semaphore cap, WINDOW_S windows, REPEATS passes,
            # aggregated direction-aware so TTFT is the WORST pass rather than
            # the best.
            #
            # This previously called a single closed-loop _point with no warmup,
            # no repeats and no direction-aware aggregation, while a comment
            # claimed it was "directly comparable" to a traversal. It was a
            # different instrument, and the claim was wrong.
            serving = None
            if not a.no_serving:
                import time as _t
                t0_ = _t.time()
                serving, serving_passes = ev.serving_metrics(
                    model, a.serving_concurrency,
                    el=lambda: f"+{(_t.time()-t0_)/60:4.1f}m", cursor=[0])

        flips = sum(1 for i in range(len(rows)) if len({v[i] for v in verdicts}) > 1)
        moved = sum(1 for i in range(len(rows)) if len({t[i] for t in texts}) > 1)
        spread = max(scores) - min(scores)
        print(f"      score {statistics.fmean(scores):.4f}   spread {spread:.4f}   "
              f"verdict flips {flips}/{len(rows)}   text changed {moved}/{len(rows)}")
        if serving:
            print(f"      serving  goodput {serving['goodput']:.1f} tok/s "
                  f"({serving['goodput_req_s']:.2f} req/s)  thru {serving['throughput']:.1f}  "
                  f"ttft_p99 {serving['ttft_p99_ms']:.0f}ms  itl_p99 {serving['itl_p99_ms']:.1f}ms  "
                  f"slo {serving['slo_attainment']:.0%}  at L={a.serving_concurrency}")

        # Every generation to disk. A summary statistic cannot be reopened, and
        # the first question after "the score moved" is "moved how".
        gen = outdir / f"generations-{tag}.jsonl"
        with open(gen, "w") as fh:
            for rep in range(a.repeats):
                for i, row in enumerate(rows):
                    fh.write(json.dumps({
                        "config": name, "repeat": rep, "problem": i,
                        "answer": _answer_of(row), "correct": verdicts[rep][i],
                        "text": texts[rep][i]}) + "\n")
        print(f"      wrote {gen}  ({a.repeats * len(rows)} generations)")

        if flips:
            fl = outdir / f"flips-{tag}.jsonl"
            with open(fl, "w") as fh:
                for i in range(len(rows)):
                    if len({v[i] for v in verdicts}) > 1:
                        fh.write(json.dumps({
                            "problem": i, "answer": _answer_of(rows[i]),
                            "prompt": prompt(rows[i])[:400],
                            "verdicts": [v[i] for v in verdicts],
                            "texts": [t[i] for t in texts]}) + "\n")
            print(f"      wrote {fl}  ({flips} problems that disagreed with THEMSELVES)")
        print()
        out[name] = {"scores": scores, "mean": statistics.fmean(scores), "spread": spread,
                     "verdict_flips": flips, "text_changed": moved, "config": cfg,
                     "serving": serving}
        keep_v[name], keep_t[name] = verdicts, texts

    print(f"  {'='*70}")
    print(f"  {a.model}   {a.benchmark}   n={len(rows)}")
    hdr = f"    {'config':12s} {'accuracy':>9s} {'spread':>7s}"
    if not a.no_serving:
        hdr += f" {'goodput':>9s} {'thru':>8s} {'ttft p99':>9s} {'itl p99':>8s} {'slo':>5s}"
    print(hdr)
    for name, o in out.items():
        line = f"    {name:12s} {o['mean']:9.4f} {o['spread']:7.4f}"
        sv = o.get("serving")
        if sv:
            line += (f" {sv['goodput']:9.1f} {sv['throughput']:8.1f} "
                     f"{sv['ttft_p99_ms']:8.0f}m {sv['itl_p99_ms']:7.1f}m "
                     f"{sv['slo_attainment']:5.0%}")
        print(line)
    if not a.no_serving:
        print(f"    serving numbers measured on {a.trace} at concurrency "
              f"{a.serving_concurrency}, not on the benchmark")

    if len(tests) == 1:
        o = next(iter(out.values()))
        if a.repeats > 1:
            print(f"\n  The eval moves {o['spread']:.4f} on its own at n={len(rows)}, "
                  f"concurrency {a.concurrency}.")
            print(f"  Any difference smaller than that is not resolvable by this "
                  f"measurement.")
            if o["text_changed"] == 0:
                print(f"  Generation was bitwise identical across repeats.")
            elif o["verdict_flips"] == 0:
                print(f"  Output text wandered but no verdict changed -- the eval is "
                      f"more stable than the generation.")

    if len(tests) >= 2:
        names = [n for n, _ in tests]
        within = max(o["spread"] for o in out.values())
        between = abs(out[names[1]]["mean"] - out[names[0]]["mean"])
        print(f"  {'-'*70}")
        print(f"  within-config spread (the eval's resolution limit)   {within:.4f}")
        print(f"  between-config difference                            {between:.4f}")
        if between <= within:
            print(f"\n  The eval CANNOT RESOLVE this difference at n={len(rows)}. Whatever")
            print(f"  separates these configs is smaller than the eval moves on its own.")
            print(f"  Raise --n, or serialise with --concurrency 1, before gating on it.")
        else:
            print(f"\n  The difference is REAL and attributable to the config change.")

        # Only problems STABLE within each config that still differ between them
        # are attributable. Anything that flip-flopped within a config is telling
        # you about the eval, not about the change.
        ab = outdir / "ab-diff.jsonl"
        n_stable = 0
        with open(ab, "w") as fh:
            for i in range(len(rows)):
                va = {v[i] for v in keep_v[names[0]]}
                vb = {v[i] for v in keep_v[names[1]]}
                if len(va) == 1 and len(vb) == 1 and va != vb:
                    n_stable += 1
                    fh.write(json.dumps({
                        "problem": i, "answer": _answer_of(rows[i]),
                        "prompt": prompt(rows[i])[:400],
                        f"{names[0]}_correct": next(iter(va)),
                        f"{names[1]}_correct": next(iter(vb)),
                        f"{names[0]}_text": keep_t[names[0]][0][i],
                        f"{names[1]}_text": keep_t[names[1]][0][i]}) + "\n")
        print(f"\n  {n_stable} problems were stable within each config and still differ")
        print(f"  between them -- the only ones attributable to the change. See {ab}")

    p = outdir / "eval.json"
    p.write_text(json.dumps({**meta, "results": out}, indent=2, default=str))
    print(f"\n  wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
