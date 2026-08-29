"""Measure the eval set's own reproducibility. One launch, N repeats, no config change.

    python eval_repro.py --model Qwen/Qwen3-14B --repeats 3

Exists because the lossless branch moved MATH-500 by +0.03 in two consecutive
runs (0.73->0.76, 0.71->0.74) across nodes that by definition cannot move
quality. That number was then used as a "tolerance" to widen the lossy gate,
which is backwards: a lossless step changing the eval is a defect to explain,
not a budget to spend.

Until this number exists, a real quantization loss cannot be told apart from a
batching artefact, and the entire lossy branch reports nothing trustworthy.

WHAT IS BEING TESTED

The same model, same config, same server, same 100 problems, greedy decoding
with a fixed seed, scored N times. Anything that moves is execution
non-determinism, not a property of the configuration.

The known mechanism is batch composition: vLLM's quality probe runs at
concurrency 32, and which requests share a forward pass changes the
floating-point reduction order in attention and GEMM accumulation. Logits shift
slightly, and a token near a decision boundary flips. Measured flip rate on this
rig is 0.44% per token -- and on a reasoning benchmark ONE flipped token early
in a chain changes the final boxed answer, so a sub-1% token flip rate can
produce several percent of answer flips.

`--concurrency 1` tests that hypothesis directly: serialised, there is one batch
composition and the run should be reproducible. If it still moves, the cause is
elsewhere.

WHAT TO DO WITH THE ANSWER

  spread ~0        the eval is a usable gate; any movement in the lossless
                   branch is a real defect, and allow_loss can be set close to
                   the smallest loss worth caring about
  spread ~0.03     the eval cannot resolve differences smaller than that at this
                   sample size. Either raise the sample count, serialise the
                   probe, or stop gating on it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(prog="eval_repro")
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--trace", default="data/trace_shared.jsonl")
    ap.add_argument("--benchmark", default="math_500")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=32,
                    help="probe concurrency. 32 is what the traversal uses; 1 "
                         "serialises it, which tests whether batch composition "
                         "is the cause.")
    ap.add_argument("--n", type=int, default=100, help="problems to score")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--run-dir", default="runs/eval_repro")
    a = ap.parse_args()

    from evaluator import VllmEvaluator, _one
    from quality import BENCHMARKS, _load
    from request import InferOptRequest, build_fingerprint
    from run import free_port, seed_config

    fp, slo = build_fingerprint(InferOptRequest(
        model=a.model, trace=a.trace, ttft_p99_ms=500, itl_p99_ms=250))
    cfg = seed_config(fp)
    cfg["enable_prefix_caching"] = True          # the kept incumbent

    ev = VllmEvaluator(fp, slo, a.trace, a.run_dir, gpu=a.gpu, port=free_port(a.port))
    b = BENCHMARKS[a.benchmark]
    rows = _load(a.benchmark, a.n)
    prompts = [b.prompt(r) for r in rows]

    print(f"\n  {a.benchmark}: {len(rows)} problems, {a.repeats} repeats at "
          f"concurrency {a.concurrency}")
    print(f"  same model, same config, same problems, greedy, seed 0 -- anything "
          f"that moves is execution non-determinism\n")

    async def run_once(model):
        sem = asyncio.Semaphore(a.concurrency)
        import httpx
        async with httpx.AsyncClient(timeout=900.0) as c:
            async def go(p):
                async with sem:
                    return await _one(c, ev.base_url, model, p, b.max_tokens, stream=False)
            return list(await asyncio.gather(*[go(p) for p in prompts]))

    scores, per_item, texts = [], [], []
    with ev._serve(cfg, "eval_repro") as model:
        for i in range(a.repeats):
            outs = asyncio.run(run_once(model))
            correct = []
            import re
            from quality import _BOXED, _norm_latex
            for r, o in zip(rows, outs):
                m = _BOXED.findall(o.text)
                correct.append(bool(m) and _norm_latex(m[-1]) == _norm_latex(r["answer"]))
            s = sum(correct) / len(correct)
            scores.append(s); per_item.append(correct); texts.append([o.text for o in outs])
            print(f"    repeat {i+1}/{a.repeats}   score {s:.4f}   ({sum(correct)}/{len(correct)})")

    print(f"\n  scores      {[f'{s:.4f}' for s in scores]}")
    print(f"  spread      {max(scores)-min(scores):.4f}   <- the eval's own resolution limit")

    # Item-level: WHICH problems changed verdict, and did the raw text change at all.
    flipped = [i for i in range(len(rows))
               if len({p[i] for p in per_item}) > 1]
    text_changed = [i for i in range(len(rows))
                    if len({t[i] for t in texts}) > 1]
    print(f"\n  problems whose VERDICT changed across repeats   {len(flipped):3d}/{len(rows)}")
    print(f"  problems whose OUTPUT TEXT changed at all        {len(text_changed):3d}/{len(rows)}")
    if text_changed and not flipped:
        print(f"    text moves but verdicts hold -- the eval is more stable than "
              f"the generation")
    if not text_changed:
        print(f"    bitwise identical across repeats -- the probe is deterministic "
              f"at this concurrency")

    out = Path(a.run_dir) / "eval_repro.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "benchmark": a.benchmark, "n": len(rows), "repeats": a.repeats,
        "concurrency": a.concurrency, "scores": scores,
        "spread": max(scores) - min(scores),
        "verdict_flips": len(flipped), "text_changes": len(text_changed),
        "flipped_indices": flipped, "config": cfg,
    }, indent=2))
    print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
