"""Does the LOSSLESS BRANCH move the eval, or does the eval just wobble?

    python eval_repro.py --from-run runs/ninth --repeats 3

MATH-500 moved +0.03 across the lossless branch in two consecutive runs
(0.73->0.76, 0.71->0.74) on nodes that by definition cannot move quality. That
number was then used as a "tolerance" to widen the lossy gate, which is
backwards twice over: a lossless step changing the eval is a defect to explain,
and the drift it produced happened to equal the lossy budget, leaving that gate
with zero width.

One measurement each cannot separate the two explanations. This runs a 2xN
design:

    config A  the SEED config          scored N times
    config B  the config the LOSSLESS BRANCH produced   scored N times

    within-config spread    the eval's own resolution limit -- same weights,
                            same flags, same problems, so anything that moves is
                            execution non-determinism
    between-config diff     what the lossless branch actually did to quality

    between <= within   the eval cannot resolve the difference; the +0.03 was
                        never a property of the branch, and quality must not be
                        gated at this sample size
    between >  within   the lossless branch REALLY changes the eval. That is a
                        defect: these nodes are flags, not weight edits.

The leading suspect for within-config movement is batch composition. The quality
probe batches at concurrency 32, and which requests share a forward pass changes
the floating-point reduction order in attention and GEMM accumulation; logits
shift and a token near a decision boundary flips. Measured flip rate on this rig
is 0.44% per token, and on a reasoning benchmark ONE flipped token early in a
chain changes the final boxed answer entirely. `--concurrency 1` serialises the
probe so there is a single batch composition; if it still moves, look elsewhere.

Both configs are scored on the IDENTICAL problem list. run_benchmark normally
filters by max_model_len, and the seed and the incumbent can differ there --
scoring two configs on two different problem sets would produce a difference
that has nothing to do with either config.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from pathlib import Path


def score_once(ev, model, rows, bench, concurrency):
    """One full pass over the problem list. Returns (score, per-item verdicts, texts)."""
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
        m = _BOXED.findall(o.text)
        verdicts.append(bool(m) and _norm_latex(m[-1]) == _norm_latex(r["answer"]))
    return sum(verdicts) / len(verdicts), verdicts, [o.text for o in outs]


def main() -> int:
    ap = argparse.ArgumentParser(prog="eval_repro")
    ap.add_argument("--from-run", default="runs/ninth",
                    help="take the seed config and the lossless incumbent from this run")
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--trace", default="data/trace_shared.jsonl")
    ap.add_argument("--benchmark", default="math_500")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=32,
                    help="probe concurrency. 32 is what the traversal uses; 1 serialises "
                         "it, testing whether batch composition is the cause.")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--run-dir", default="runs/eval_repro")
    a = ap.parse_args()

    from evaluator import VllmEvaluator
    from quality import BENCHMARKS, _load
    from request import InferOptRequest, build_fingerprint
    from run import free_port, seed_config

    fp, slo = build_fingerprint(InferOptRequest(
        model=a.model, trace=a.trace, ttft_p99_ms=500, itl_p99_ms=250))

    # The two configs under test: where the branch started, and where it ended.
    seed = seed_config(fp)
    res = Path(a.from_run) / "result.json"
    if res.exists():
        r = json.loads(res.read_text())
        incumbent = r["incumbent"]
        if r.get("baseline", {}).get("config"):
            seed = r["baseline"]["config"]
    else:
        print(f"  {res} not found -- falling back to seed + prefix_caching")
        incumbent = {**seed, "enable_prefix_caching": True}

    bench = BENCHMARKS[a.benchmark]
    rows = _load(a.benchmark, a.n)

    # Identical problem list for both configs. run_benchmark would filter by
    # max_model_len, and the two configs differ there -- scoring them on
    # different problem sets manufactures a difference belonging to neither.
    budget = min(seed.get("max_model_len", 1 << 30),
                 incumbent.get("max_model_len", 1 << 30)) - bench.max_tokens
    too_long = [r for r in rows if len(bench.prompt(r)) // 4 >= budget]
    if too_long:
        print(f"  dropping {len(too_long)} problems that exceed the tighter config's "
              f"{budget}-token budget, so both are scored on the same set")
        rows = [r for r in rows if len(bench.prompt(r)) // 4 < budget]

    print(f"\n  {a.benchmark}: {len(rows)} problems x {a.repeats} repeats x 2 configs, "
          f"concurrency {a.concurrency}")
    print(f"  A  seed        {json.dumps({k: v for k, v in seed.items() if k != 'model'})[:88]}")
    print(f"  B  incumbent   {json.dumps({k: v for k, v in incumbent.items() if k != 'model'})[:88]}\n")

    ev = VllmEvaluator(fp, slo, a.trace, a.run_dir, gpu=a.gpu, port=free_port(a.port))
    out: dict[str, dict] = {}

    for name, cfg in (("A seed", seed), ("B lossless incumbent", incumbent)):
        print(f"  --- {name} ---")
        scores, verdicts, texts = [], [], []
        with ev._serve(cfg, f"eval_repro-{name.split()[0]}") as model:
            for i in range(a.repeats):
                s, v, t = score_once(ev, model, rows, bench, a.concurrency)
                scores.append(s); verdicts.append(v); texts.append(t)
                print(f"      repeat {i+1}/{a.repeats}   {s:.4f}   ({sum(v)}/{len(v)})")
        flips = sum(1 for i in range(len(rows)) if len({v[i] for v in verdicts}) > 1)
        moved = sum(1 for i in range(len(rows)) if len({t[i] for t in texts}) > 1)
        spread = max(scores) - min(scores)
        print(f"      spread {spread:.4f}   verdict flips {flips}/{len(rows)}   "
              f"text changed {moved}/{len(rows)}\n")
        out[name] = {"scores": scores, "spread": spread, "verdict_flips": flips,
                     "text_changed": moved, "mean": statistics.fmean(scores), "config": cfg}
        if name.startswith("A"):
            A_verdicts, A_texts = verdicts, texts
        else:
            B_verdicts, B_texts = verdicts, texts

        # EVERY generation goes to disk. If the eval turns out to move, the only
        # way to find out why is to read what actually changed -- and a summary
        # statistic cannot be re-opened. One line per (repeat, problem).
        tag = name.split()[0]
        gen = Path(a.run_dir) / f"generations-{tag}.jsonl"
        gen.parent.mkdir(parents=True, exist_ok=True)
        with open(gen, "w") as fh:
            for rep in range(a.repeats):
                for i, row in enumerate(rows):
                    fh.write(json.dumps({
                        "config": tag, "repeat": rep, "problem": i,
                        "answer": row["answer"], "correct": verdicts[rep][i],
                        "text": texts[rep][i],
                    }) + "\n")
        print(f"      wrote {gen}  ({a.repeats * len(rows)} generations)")

        # The problems that disagreed with themselves, side by side, so the
        # first question after "the eval moved" -- moved HOW -- is answerable
        # without re-running anything.
        if flips:
            diff = Path(a.run_dir) / f"flips-{tag}.jsonl"
            with open(diff, "w") as fh:
                for i in range(len(rows)):
                    if len({v[i] for v in verdicts}) > 1:
                        fh.write(json.dumps({
                            "problem": i, "answer": rows[i]["answer"],
                            "prompt": bench.prompt(rows[i])[:400],
                            "verdicts": [v[i] for v in verdicts],
                            "texts": [t[i] for t in texts],
                        }) + "\n")
            print(f"      wrote {diff}  ({flips} problems that flipped verdict)")

    A, B = out["A seed"], out["B lossless incumbent"]
    within = max(A["spread"], B["spread"])
    between = abs(B["mean"] - A["mean"])
    print(f"  {'='*68}")
    print(f"  within-config spread (the eval's resolution limit)   {within:.4f}")
    print(f"  between-config difference (what the branch did)      {between:.4f}")
    print(f"  {'='*68}")
    if between <= within:
        print(f"  The eval CANNOT RESOLVE the difference. The +0.03 seen in runs seven")
        print(f"  and nine was not a property of the lossless branch, and quality must")
        print(f"  not be gated at this sample size -- raise --n, or serialise the probe.")
    else:
        print(f"  The lossless branch REALLY MOVES the eval, by {between:.4f} against a")
        print(f"  {within:.4f} resolution limit. These nodes are launch flags, not weight")
        print(f"  edits, so this is a defect to explain -- not a tolerance to spend.")
    if A["text_changed"] == 0 and B["text_changed"] == 0:
        print(f"\n  Generation was bitwise identical across repeats at concurrency "
              f"{a.concurrency}.")
    elif max(A["verdict_flips"], B["verdict_flips"]) == 0:
        print(f"\n  Output text wandered but no verdict changed -- the eval is more "
              f"stable than the generation.")

    p = Path(a.run_dir) / "eval_repro.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"benchmark": a.benchmark, "n": len(rows),
                             "repeats": a.repeats, "concurrency": a.concurrency,
                             "within": within, "between": between, "configs": out,
                             "model": a.model, "trace": a.trace,
                             "problem_answers": [r["answer"] for r in rows]},
                            indent=2, default=str))
    print(f"\n  wrote {p}")
    print(f"  generations in {a.run_dir}/generations-A.jsonl and generations-B.jsonl")

    # A/B item comparison: the problems the two configs actually disagree on.
    # This is the real question -- not "did the score move" but "which problems
    # moved, and does B's output look damaged or merely different". A problem is
    # only interesting if it was STABLE within each config and still differs
    # between them; anything that flip-flopped within a config is telling you
    # about the eval, not about the branch.
    ab = Path(a.run_dir) / "ab-diff.jsonl"
    n_stable_diff = 0
    with open(ab, "w") as fh:
        for i in range(len(rows)):
            a_v = {v[i] for v in A_verdicts}
            b_v = {v[i] for v in B_verdicts}
            if len(a_v) == 1 and len(b_v) == 1 and a_v != b_v:
                n_stable_diff += 1
                fh.write(json.dumps({
                    "problem": i, "answer": rows[i]["answer"],
                    "prompt": bench.prompt(rows[i])[:400],
                    "seed_correct": a_v.pop(), "incumbent_correct": b_v.pop(),
                    "seed_text": A_texts[0][i], "incumbent_text": B_texts[0][i],
                }) + "\n")
    print(f"  {n_stable_diff} problems were STABLE within each config and still "
          f"disagree between them")
    print(f"  those are in {ab} -- the only ones attributable to the branch")
    return 0


if __name__ == "__main__":
    sys.exit(main())
