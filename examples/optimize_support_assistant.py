"""Find a serving config for our support assistant, and prove it still works.

    python optimize_support_assistant.py

TARGETS AN API THAT DOES NOT EXIST YET. This is the caller's side of
docs/api-design.md, written to see whether the design is pleasant to use before
any of it is built. Nothing here imports successfully today.

Context, so the choices below make sense: we serve Qwen3-14B behind a support
assistant. Traffic is ~16 req/s with a large shared system prompt, and every
answer must come back as JSON our downstream parser can read. We care about
three things, in this order:

  1. the JSON must parse                 -- a malformed answer is a hard failure
  2. the answer must be correct          -- graded against our labelled set
  3. reasoning must not degrade          -- MATH-500 as a canary for the model
                                            generally, not because we do maths
"""

from __future__ import annotations

import json
from pathlib import Path

from inferopt import (
    Benchmark,
    LLMJudge,
    Metric,
    Sample,
    SLO,
    Verdict,
    optimize,
)

HERE = Path(__file__).parent


# ---------------------------------------------------------------------------
# 1. Our own benchmark: 400 labelled support tickets
# ---------------------------------------------------------------------------
# Each row is {"ticket": str, "expected_category": str, "expected_refund": bool}.
# We ask for JSON, so the judge's first job is "did it parse at all".

def support_judge(samples: list[Sample]) -> list[Verdict]:
    """Parse the JSON answer and check the category and refund flag.

    Returns one Verdict per sample, IN ORDER -- the caller zips these against
    rows to find which ticket regressed, so order is load-bearing.

    A parse failure is a per-sample error, not an exception: one malformed
    answer must not abandon the other 399. If EVERY sample fails, inferopt
    raises on our behalf rather than reporting 0.0, which would look like the
    model collapsing rather than the judge breaking.
    """
    out = []
    for s in samples:
        text = s.text.strip()
        if text.startswith("```"):                 # strip a markdown fence
            text = text.split("```")[1].removeprefix("json").strip()
        try:
            got = json.loads(text)
        except json.JSONDecodeError as e:
            out.append(Verdict(score=0.0, passed=False,
                               error=f"not JSON: {e}",
                               extra={"raw": s.text[:200]}))
            continue

        cat_ok = got.get("category") == s.row["expected_category"]
        ref_ok = bool(got.get("refund")) == s.row["expected_refund"]
        out.append(Verdict(
            score=1.0 if (cat_ok and ref_ok) else 0.0,
            passed=cat_ok and ref_ok,
            extra={"parsed": True, "category_ok": cat_ok, "refund_ok": ref_ok},
        ))
    return out


def json_validity(samples: list[Sample], verdicts: list[Verdict]) -> float:
    """Fraction of answers that parsed at all, regardless of correctness.

    Separate from accuracy on purpose. A quantization that keeps accuracy but
    starts emitting unparseable JSON breaks our pipeline just as hard, and an
    accuracy number alone would not show it.
    """
    return sum(1 for v in verdicts if not v.error) / max(1, len(verdicts))


def refund_recall(samples: list[Sample], verdicts: list[Verdict]) -> float:
    """Of the tickets that DO warrant a refund, how many did we catch.

    A set-level metric -- it needs the labels across the whole set, which is
    why metrics take (samples, verdicts) and not just per-sample booleans.
    """
    tp = fn = 0
    for s, v in zip(samples, verdicts):
        if not s.row["expected_refund"]:
            continue
        if v.extra and v.extra.get("refund_ok"):
            tp += 1
        else:
            fn += 1
    return tp / max(1, tp + fn)


support = Benchmark(
    name="support_tickets",
    dataset=str(HERE / "data" / "support_labelled.jsonl"),   # local jsonl
    prompt=lambda r: (
        "You are a support triage assistant. Classify the ticket and decide "
        "whether it warrants a refund.\n\n"
        f"Ticket: {r['ticket']}\n\n"
        'Answer as JSON: {"category": "...", "refund": true|false}'
    ),
    judge=support_judge,
    metrics=[
        # Both of these move the frontier's quality axis, so a config that
        # degrades either one sits lower on the plot -- visibly, without being
        # taken away from us.
        Metric("exact_match"),
        Metric("json_validity", higher_is_better=True, fn=json_validity),

        # Worth recording, but we do not want it moving the frontier: recall
        # trades against precision and we would rather read it than optimise it.
        Metric("refund_recall", higher_is_better=True, fn=refund_recall,
               role="report"),
    ],
    max_tokens=256,
    chat=True,          # instruct model: without the template it never stops
)


# ---------------------------------------------------------------------------
# 2. A canary we do not otherwise care about
# ---------------------------------------------------------------------------
# We do not serve maths. MATH-500 is here because long reasoning chains are the
# most sensitive thing we can cheaply measure -- quantization damage shows up
# here before it shows up in 200-token JSON answers. Built-in, so no judge.

reasoning_canary = Benchmark.builtin("math_500", n=500)


# ---------------------------------------------------------------------------
# 3. A subjective check, graded by another model
# ---------------------------------------------------------------------------
# role="report": we do not trust an LLM judge enough to let it move the frontier,
# but a sharp drop is worth reading. The judge model runs on a different device
# than the one under test -- inferopt refuses it otherwise, because a judge
# sharing the GPU would contend with the measurement.

tone = Benchmark(
    name="tone",
    dataset=str(HERE / "data" / "tone_probe.jsonl"),
    prompt=lambda r: r["ticket"],
    judge=LLMJudge(
        model="claude-sonnet-4-5",
        prompt=("Rate 1-5 how appropriate this support reply's tone is.\n\n"
                "Ticket: {row[ticket]}\nReply: {text}\n\nReturn only a digit."),
        parse=lambda s: float(s.strip()[:1]),
        retries=2,
        on_parse_failure="error",
    ),
    metrics=[Metric("tone_1_5", higher_is_better=True, aggregation="mean",
                    role="report")],
    max_tokens=200,
    chat=True,
)


# ---------------------------------------------------------------------------
# 4. Run it
# ---------------------------------------------------------------------------

result = optimize(
    model="Qwen/Qwen3-14B",

    # Real captured traffic. arrival_ts is what makes this a trace rather than a
    # dataset -- without it there is no arrival pattern and prefix caching is
    # measured against a workload that cannot benefit from it.
    trace=str(HERE / "data" / "prod_trace_7d.jsonl"),

    slo=SLO(ttft_p99_ms=500, itl_p99_ms=250),

    benchmarks=[support, reasoning_canary, tone],

    # Deliberately no require= and no abandon_below=. We want every config
    # measured and on the frontier, including the ones that cost accuracy --
    # choosing between them is the whole point, and a gate would delete the
    # options before we saw them.

    hardware="auto",
    seed_from="aiconfigurator",     # predict cheaply, then measure the shortlist
    budget_minutes=300,
    run_dir="runs/support-2026-09",
)


# ---------------------------------------------------------------------------
# 5. What we do with it
# ---------------------------------------------------------------------------

best = result.best()
print(f"\nchosen config ({best.goodput:.1f} tok/s goodput, "
      f"{best.ttft_p99_ms:.0f}ms TTFT p99):")
print(json.dumps(best.config, indent=2))

# Capacity planning. replicas() uses the CHOSEN config's own capacity curve,
# not the seed's -- on our MoE run those disagreed ninefold.
print(f"\nat 16 req/s we need {result.replicas(qps=16.0)} replicas "
      f"(baseline needed {result.replicas(qps=16.0, config=result.baseline)})")

# Every RESOLVED quality movement, in either direction. Nothing here stopped
# the search; it is here so we can decide what we are willing to pay.
for c in result.quality_changes:
    print(f"  {c.direction:11s} {c.benchmark}.{c.metric}: "
          f"{c.baseline:.4f} -> {c.value:.4f} ({c.delta:+.4f}, "
          f"tolerance {c.tolerance:.4f})")

# The frontier includes REVERTED configs. "Less goodput, better latency" is a
# trade we might want -- our p99 matters more than our throughput.
print("\nfrontier -- these are the choices, we pick one:")
for t in result.frontier:
    print(f"  {t.node_id:24s} {t.goodput:7.1f} tok/s  "
          f"{t.ttft_p99_ms:6.0f}ms  quality {t.quality_axis:.3f}  "
          f"${result.cost_per_mtok(gpu_hourly=2.99, config=t.config):.3f}/Mtok")

result.plot.all(str(HERE / "report.html"))          # every plot, one page
result.save(str(HERE / "result.json"))

# Having READ the frontier, we choose. best() is the highest-goodput point; if
# we decide 0.72 accuracy is too low for us, we filter at THIS point -- after
# seeing the options, not before measuring them.
#
#   best = result.best(require={"support_tickets.exact_match": 0.74})
#
# Deploy the winner. best.config is a plain dict of vLLM flags.
Path(HERE / "serve.sh").write_text(
    "vllm serve Qwen/Qwen3-14B \\\n  " +
    " \\\n  ".join(f"--{k.replace('_','-')} {v}" for k, v in best.config.items())
    + "\n"
)
