"""Quality benchmarks, each scored by ITS OWN metric.

    run_benchmark("math_500", generate)   ->  0.4380

`generate(prompts, max_tokens) -> [Req]` is supplied by the evaluator, so a
benchmark never launches or owns a server.

Deliberately no perplexity, no substitute metrics: a benchmark is scored the way
it is meant to be scored, or it is not run. Introducing a proxy metric makes the
number incomparable to every published figure and to the user's own expectations.

Sample counts follow the traversal/certification split:
  traversal   a pinned subset (default 100) -- enough to RANK configs, because
              the comparison is paired on identical items and most sampling
              noise cancels
  finalists   the full dataset -- needed to state an ABSOLUTE number

HISTORY -- gates that could not be passed

  THE GATE THAT COULD NEVER PASS. An earlier accuracy gate demanded
  token-identical 48-token outputs -- a STRICT-equivalence test -- in a world
  where kernel selection and batching make runs only ALGORITHMICALLY equivalent.
  Measured per-token flip rate is 0.44%, giving a 22.6% false-positive floor on
  48 tokens. At the same time the NIAH tolerance was set to 1%, exactly the
  granularity of a 100-item set. The seed configuration failed its own gate, and
  nothing could ever pass. Three defects, one root cause: thresholds guessed
  rather than measured.

  RULER PROMPTS EXCEEDED THE SERVED CONTEXT. Generated at 16k-33k tokens, served
  under a right-sized max_model_len of 6144. Every prompt would be rejected and
  the benchmark would score 0.0 -- for EVERY config, so the gate reads "quality
  unchanged" rather than "probe broken". Exactly the shape of the bug above.
  run_benchmark now filters prompts that do not fit and RAISES if none do.

  The dataset was also simply missing: fetch_data.py had never been run for
  ruler_multineedle, and nothing noticed until the file was looked for.

  MATH-500 answers are LaTeX, not numbers. '\\left( 3, \\frac{\\pi}{2} \\right)'
  does not survive numeric extraction, hence _norm_latex and string matching.

  HumanEval+ deliberately raises. Generation is implemented; pass@1 scoring
  needs a real sandbox, and wiring it to bare exec() would be the most dangerous
  line in this project.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable, Protocol

DATA = Path(__file__).parent / "data"
TRAVERSAL_N = 100


class Generate(Protocol):
    def __call__(self, prompts: list[str], max_tokens: int): ...


# --------------------------------------------------------------------------

def _load(name: str, n: int | None) -> list[dict]:
    p = DATA / f"{name}.jsonl"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Fetch it first (see fetch_data.py); benchmarks are not "
            f"downloaded lazily mid-traversal, because a stall there costs a launch."
        )
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return rows[:n] if n else rows


_BOXED = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")


def _norm_latex(s: str) -> str:
    """Normalize a MATH-500 answer for comparison.

    Answers are LaTeX, not numbers -- e.g. '\\left( 3, \\frac{\\pi}{2} \\right)'.
    Extracting the last number, the obvious thing to do, scores every such
    answer wrong. Compare normalized strings instead: strip sizing commands and
    whitespace, drop trailing zeros, unify the few spellings that differ only
    typographically.
    """
    s = str(s).strip()
    for a, b in (("\\left", ""), ("\\right", ""), ("\\!", ""), ("\\,", ""),
                 ("\\ ", ""), ("dfrac", "frac"), ("tfrac", "frac"),
                 ("^{\\circ}", ""), ("^\\circ", ""), ("\\%", ""), ("%", ""),
                 ("\\$", ""), ("$", ""), (" ", "")):
        s = s.replace(a, b)
    s = s.rstrip(".").replace(",", "")
    if re.fullmatch(r"-?\d+\.\d+", s):
        s = s.rstrip("0").rstrip(".")
    return s


def _math_500(rows, gen: Generate) -> float:
    """exact_match on the final boxed answer -- MATH-500's own metric."""
    outs = gen([r["problem"] + "\n\nPut your final answer in \\boxed{}."
                for r in rows], 1024)
    hit = 0
    for r, o in zip(rows, outs):
        m = _BOXED.findall(o.text)
        if not m:
            continue          # no boxed answer produced == wrong, not skipped
        hit += _norm_latex(m[-1]) == _norm_latex(r["answer"])
    return hit / len(rows)


def _ruler(rows, gen: Generate) -> float:
    """accuracy -- every needle must appear. Multi-needle on purpose: the
    single-needle variant saturates at 1.00 and cannot show a regression."""
    outs = gen([r["prompt"] for r in rows], 64)
    hit = 0
    for r, o in zip(rows, outs):
        needles = r["answers"] if isinstance(r.get("answers"), list) else [r["answer"]]
        hit += all(str(n).strip().lower() in o.text.lower() for n in needles)
    return hit / len(rows)


def _humaneval_plus(rows, gen: Generate) -> float:
    """pass@1 by executing the generated code against the tests.

    Generation is implemented; SCORING is not, and deliberately so: running
    model-written code requires a real sandbox. Wiring it to bare exec() would
    be the single most dangerous line in this project.
    """
    outs = gen([r["prompt"] for r in rows], 512)
    raise NotImplementedError(
        f"generated {len(outs)} completions, but pass@1 needs sandboxed execution.\n"
        f"Wire this to a container/nsjail runner (or the evalplus harness) before "
        f"enabling humaneval_plus. Until then, drop it from quality_benchmarks in "
        f"dag/llm.json -- MATH-500 and RULER cover reasoning and long-context recall, "
        f"and running it unsandboxed is not an acceptable shortcut."
    )


BENCHMARKS: dict[str, tuple[Callable, str, int]] = {
    "math_500": (_math_500, "exact_match", 500),
    "ruler_multineedle": (_ruler, "accuracy", 200),
    "humaneval_plus": (_humaneval_plus, "pass@1", 164),
}


def run_benchmark(name: str, gen: Generate, *, full: bool = False,
                  max_input_tokens: int | None = None) -> float:
    """Score one benchmark, refusing to score it on prompts the server cannot take.

    RULER generates 16k-33k token documents. Served under a right-sized
    max_model_len (6144 on this workload) every one of them is rejected for
    exceeding context, and the benchmark returns 0.0 -- for EVERY config, so the
    gate reads "quality unchanged" instead of "probe broken". That is the same
    shape as the accuracy gate that could never pass in the previous run, so it
    fails loudly here instead.
    """
    if name not in BENCHMARKS:
        raise KeyError(f"unknown benchmark {name!r}; have {', '.join(BENCHMARKS)}")
    fn, _metric, n_full = BENCHMARKS[name]
    rows = _load(name, None if full else TRAVERSAL_N)

    if max_input_tokens:
        budget = max_input_tokens - 128          # leave room for the generation
        fits = [r for r in rows if len(r["prompt"]) // 4 < budget]
        if not fits:
            longest = min(len(r["prompt"]) // 4 for r in rows)
            raise ValueError(
                f"{name}: every prompt exceeds the served context. Shortest is "
                f"~{longest} tokens, max_model_len is {max_input_tokens}. The server "
                f"would reject all of them and the benchmark would score 0.0 for every "
                f"config -- indistinguishable from 'no quality change'.\n"
                f"Regenerate at a context that fits:\n"
                f"    python fetch_data.py --ruler-contexts {budget//2},{int(budget*0.9)}")
        if len(fits) < len(rows):
            print(f"        {name}: {len(rows)-len(fits)}/{len(rows)} prompts exceed "
                  f"max_model_len={max_input_tokens}, scoring on {len(fits)}")
        rows = fits
    return round(fn(rows, gen), 4)


def resolution(name: str, *, full: bool = False) -> float:
    """Smallest delta the sample size can express. Report it alongside the
    score: claiming a 0.5% regression on 100 items is claiming half an item."""
    n = BENCHMARKS[name][2] if full else TRAVERSAL_N
    return 1.0 / n
