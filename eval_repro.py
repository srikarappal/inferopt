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


def score_once(ev, model, rows, bench, concurrency):
    """One pass over the problem list. Returns (score, per-item verdicts, texts)."""
    import httpx

    from evaluator import _one
    from quality import _BOXED, _norm_latex

    prompts = [bench.prompt(r) for r in rows]

    async def go_all():
        sem = asyncio.Semaphore(concurrency)
        async with httpx.AsyncClient(timeout=900.0) as c:
            async def go(p):
                async with sem:
                    return await _one(c, ev.base_url, model, p, bench.max_tokens, stream=False)
            return list(await asyncio.gather(*[go(p) for p in prompts]))

    outs = asyncio.run(go_all())
    verdicts = []
    for r, o in zip(rows, outs):
        if "answers" in r:                  # RULER: every needle must appear
            needles = r["answers"] if isinstance(r["answers"], list) else [r["answers"]]
            verdicts.append(all(str(n).strip().lower() in o.text.lower() for n in needles))
        else:                               # MATH-500: the final boxed answer
            m = _BOXED.findall(o.text)
            verdicts.append(bool(m) and _norm_latex(m[-1]) == _norm_latex(r["answer"]))
    return sum(verdicts) / len(verdicts), verdicts, [o.text for o in outs]


def _answer_of(row) -> object:
    return row.get("answer") if "answer" in row else row.get("answers")


def provenance(ap, a, fp, tests, port, rows, bench) -> dict:
    """Everything needed to reproduce this invocation, or to distrust it later.

    Written BEFORE any measurement, so a crashed run still records what was
    attempted. A results file that does not say which model, which flags, which
    code and which machine produced it is a number without a claim attached --
    and this project has already lost one measurement that way.

    Args are recorded with their default alongside the value, and a flag for
    whether it was explicit, so reading it back answers "did they set that, or
    did it just happen to be the default at the time" -- which matters when a
    default later changes.
    """
    import platform
    import subprocess
    import time

    def _v(mod):
        try:
            return __import__(mod).__version__
        except Exception:
            return None

    def _git(*args):
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True,
                                  timeout=10, cwd=Path(__file__).parent).stdout.strip()
        except Exception:
            return None

    return {
        "when": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "command": " ".join(sys.argv),
        "args": {k: {"value": v, "default": ap.get_default(k),
                     "explicit": v != ap.get_default(k)}
                 for k, v in sorted(vars(a).items())},
        "resolved": {
            "port": port,                       # free_port may differ from --port
            "configs": {name: cfg for name, cfg in tests},
            "benchmark": {"name": a.benchmark, "metric": bench.metric,
                          "max_tokens": bench.max_tokens,
                          "file": str(Path("data") / f"{a.benchmark}.jsonl"),
                          "problems_scored": len(rows)},
        },
        "fingerprint": {
            "model": fp.model.model_dump(),
            "hardware": fp.hw.model_dump(),
        },
        "environment": {
            "python": platform.python_version(),
            "executable": sys.executable,
            "platform": platform.platform(),
            "vllm": _v("vllm"), "torch": _v("torch"),
            "transformers": _v("transformers"),
            "inferopt_commit": _git("rev-parse", "HEAD"),
            "inferopt_dirty": bool(_git("status", "--porcelain")),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="eval_repro",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True,
                    help="HF id or local path, including a quantized checkpoint")
    ap.add_argument("--benchmark", default="math_500",
                    help="math_500 | ruler_multineedle (see quality.py)")
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

    # Every config is scored on the IDENTICAL problem list. A traversal filters
    # by max_model_len, and two configs can differ there -- scoring them on
    # separately-filtered sets manufactures a difference belonging to neither.
    limits = [c["max_model_len"] for _, c in tests if c.get("max_model_len")]
    if limits:
        budget = min(limits) - bench.max_tokens
        keep = [r for r in rows if len(bench.prompt(r)) // 4 < budget]
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

    meta = provenance(ap, a, fp, tests, port, rows, bench)
    (outdir / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"  provenance -> {outdir/'run_meta.json'}  "
          f"(commit {(meta['environment']['inferopt_commit'] or '?')[:8]}"
          f"{', DIRTY' if meta['environment']['inferopt_dirty'] else ''}, "
          f"vllm {meta['environment']['vllm']})\n")
    out: dict[str, dict] = {}
    keep_v: dict[str, list] = {}
    keep_t: dict[str, list] = {}

    for name, cfg in tests:
        tag = name.split()[0]
        print(f"  --- {name} ---")
        scores, verdicts, texts = [], [], []
        with ev._serve({**cfg, "model": a.model}, f"eval-{tag}") as model:
            for i in range(a.repeats):
                s, v, t = score_once(ev, model, rows, bench, a.concurrency)
                scores.append(s); verdicts.append(v); texts.append(t)
                print(f"      repeat {i+1}/{a.repeats}   {s:.4f}   ({sum(v)}/{len(v)})")
        flips = sum(1 for i in range(len(rows)) if len({v[i] for v in verdicts}) > 1)
        moved = sum(1 for i in range(len(rows)) if len({t[i] for t in texts}) > 1)
        spread = max(scores) - min(scores)
        print(f"      score {statistics.fmean(scores):.4f}   spread {spread:.4f}   "
              f"verdict flips {flips}/{len(rows)}   text changed {moved}/{len(rows)}")

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
                            "prompt": bench.prompt(rows[i])[:400],
                            "verdicts": [v[i] for v in verdicts],
                            "texts": [t[i] for t in texts]}) + "\n")
            print(f"      wrote {fl}  ({flips} problems that disagreed with THEMSELVES)")
        print()
        out[name] = {"scores": scores, "mean": statistics.fmean(scores), "spread": spread,
                     "verdict_flips": flips, "text_changed": moved, "config": cfg}
        keep_v[name], keep_t[name] = verdicts, texts

    print(f"  {'='*70}")
    print(f"  {a.model}   {a.benchmark}   n={len(rows)}")
    for name, o in out.items():
        print(f"    {name:12s} {o['mean']:.4f}   spread {o['spread']:.4f}")

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
                        "prompt": bench.prompt(rows[i])[:400],
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
