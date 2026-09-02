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

  HumanEval+ deliberately raised for a year, because pass@1 scoring needs a real
  sandbox and wiring it to bare exec() would be the most dangerous line in this
  project. That refusal was right, and MBPP+ is how it was finally paid off:
  execution happens in mbpp_score.py, in a separate process, under evalplus's
  own guards. HumanEval+ is now unregistered by CHOICE, not by blocker.

  JUDGING WAS IN THREE PLACES. quality.py scored a benchmark, and
  eval_repro.score_once scored it again -- picking the rule by sniffing whether
  a row had an "answers" key -- because it needed per-item verdicts for its flip
  analysis. That is the same duplication that put MATH-500's prompt text in two
  places, let them disagree, and killed a traversal nine launches in. A code
  benchmark made it untenable rather than merely risky: its judge is a
  subprocess, and no amount of row-sniffing reproduces one. Benchmarks now own a
  `judge(rows, texts) -> [bool]` and every caller uses it.

  MAX_TOKENS WAS IN TWO PLACES. Benchmark.max_tokens sized the context filter
  while each scorer passed its own literal to gen(). They agreed by luck.

  THE GROUNDTRUTH CACHE CAN BE POISONED. evalplus caches expected outputs in a
  pickle keyed by a hash of the DATASET. Computing them from a 5-problem subset
  while passing the full dataset's hash -- which is what an interactive probe
  does -- writes 5 problems' expectations under the key that 378 problems will
  later be read from. It surfaced as a KeyError deep in scoring. mbpp_score.py
  now checks coverage up front and says which file to delete. Hit while building
  MBPP+, by exactly the shortcut its own docstring warned against.

  CONTAMINATION IS NOT A PROBLEM HERE, and it is worth being explicit about why,
  because it is the first objection anyone raises about MBPP. Every row of the
  frontier compares the SAME model against ITSELF under different serving
  configs. A memorised problem is memorised identically on both sides and
  cancels out of the delta. Contamination would only invalidate an ABSOLUTE
  claim about the model's coding ability, which this project never makes.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Protocol

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
TRAVERSAL_N = 100

# Chat templates, cached per model. Building one costs a tokenizer load.
_TEMPLATES: dict[str, object] = {}


def _chat_wrapper(prompt: Callable, model: str | None) -> Callable:
    """Wrap a prompt builder so its text is sent as a chat turn, not a raw completion.

    The evaluator talks to /v1/completions, which does NOT apply a chat
    template. For MATH-500 that is fine and is what every number in this project
    was measured with.

    WHY IT IS ON FOR CODE, measured rather than assumed. The reason first written
    here was that an instruct model asked to continue a bare docstring would
    write prose instead of a function. That was a guess, and it was WRONG: on 60
    MBPP+ problems both styles emitted a fenced code block 60/60 times, and
    accuracy differed by 0.0167 -- one problem, exactly the resolution limit at
    n=60, so not a difference at all.

    The real reason showed up in the output LENGTHS:

        raw   pass@1 0.8167   mean 653 output tokens, p95 894
        chat  pass@1 0.8333   mean  37 output tokens, p95  73

    A raw completion has no stop token, so every generation runs to max_tokens --
    the model answers, then keeps going. The chat template ends the turn at
    <|im_end|>. Same answers, 18x the tokens to get them, and a decode-bound
    workload profile that belongs to the prompt format rather than to the model.
    Keeping it raw would have made a code benchmark cost 18x its own weight in
    GPU time and reported serving numbers shaped by that.

    The template is applied HERE, in text, and the request still goes through the
    same completions path as every other measurement. Nothing about the serving
    instrument changes -- only the characters in the prompt.

    THINKING IS DISABLED. Qwen3's template defaults to emitting a <think> block,
    which would (a) blow past max_tokens on most problems and (b) change output
    length by an order of magnitude, making the serving numbers incomparable to
    every other row in the table. `enable_thinking` is Qwen-specific; other
    templates ignore an unknown kwarg, and if one raises we retry without it.
    """
    if not model:
        return prompt
    if model not in _TEMPLATES:
        try:
            from transformers import AutoTokenizer
            _TEMPLATES[model] = AutoTokenizer.from_pretrained(
                model, trust_remote_code=True)
        except Exception as e:
            print(f"        no chat template for {model} ({type(e).__name__}); "
                  f"sending raw text")
            _TEMPLATES[model] = None
    tok = _TEMPLATES[model]
    if tok is None or not getattr(tok, "chat_template", None):
        return prompt

    def wrapped(r):
        msgs = [{"role": "user", "content": prompt(r)}]
        for kw in ({"enable_thinking": False}, {}):
            try:
                return tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True, **kw)
            except Exception:
                continue
        return prompt(r)
    return wrapped


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


def _math_500_prompt(r) -> str:
    return r["problem"] + "\n\nPut your final answer in \\boxed{}."


def _ruler_prompt(r) -> str:
    return r["prompt"]


def _judge_math_500(rows, texts) -> list[bool]:
    """exact_match on the final boxed answer -- MATH-500's own metric."""
    out = []
    for r, t in zip(rows, texts):
        m = _BOXED.findall(t)
        # no boxed answer produced == wrong, not skipped
        out.append(bool(m) and _norm_latex(m[-1]) == _norm_latex(r["answer"]))
    return out


def _judge_ruler(rows, texts) -> list[bool]:
    """accuracy -- every needle must appear. Multi-needle on purpose: the
    single-needle variant saturates at 1.00 and cannot show a regression."""
    out = []
    for r, t in zip(rows, texts):
        needles = r["answers"] if isinstance(r.get("answers"), list) else [r["answer"]]
        out.append(all(str(n).strip().lower() in t.lower() for n in needles))
    return out


def _mbpp_plus_prompt(r) -> str:
    """The user turn. Row `prompt` is MBPP+'s own docstring, assertion included.

    The assertion is left in deliberately -- it is the only thing that pins the
    function NAME, and evalplus scores by calling `entry_point`. Strip it and a
    correct solution under a different name scores zero, which would read as
    quantization damage.
    """
    return (
        "Write a self-contained Python function that solves this problem.\n\n"
        f"{r['prompt'].strip()}\n\n"
        "Respond with only the code, in a single Python markdown block. "
        "Do not include tests or explanation."
    )


def _judge_mbpp_plus(rows, texts) -> list[bool]:
    """pass@1 -- MBPP+'s own metric, by EXECUTING the generated code.

    Execution happens in mbpp_score.py, in a separate process, under evalplus's
    guards. See that file for why the boundary is there and what it does and does
    not protect against. The short version: this project spent a year refusing to
    call exec() on model output, and the refusal was correct; a subprocess is the
    cheapest thing that is actually a boundary.

    Scoring is a SUBSET of the dataset during traversal (100 of 378) and the full
    set for a finalist, exactly like every other benchmark here. mbpp_score
    computes groundtruth over all 378 regardless, because evalplus caches it
    under a hash of the dataset -- caching a subset's expectations under the full
    set's key would silently corrupt every later run.
    """
    with tempfile.TemporaryDirectory(prefix="mbpp-") as td:
        samples = Path(td) / "samples.jsonl"
        verdicts = Path(td) / "verdicts.json"
        samples.write_text("\n".join(
            json.dumps({"task_id": r["task_id"], "raw": t})
            for r, t in zip(rows, texts)) + "\n")

        proc = subprocess.run(
            [sys.executable, str(HERE / "mbpp_score.py"),
             "--samples", str(samples), "--out", str(verdicts)],
            capture_output=True, text=True, timeout=3600, cwd=HERE)
        if not verdicts.exists():
            raise RuntimeError(
                f"MBPP+ scoring produced no verdicts (exit {proc.returncode}).\n"
                f"Scoring silently returning 0.0 would read as 'quality unchanged' "
                f"for every config, so this raises instead.\n"
                f"--- stderr ---\n{proc.stderr[-2000:]}")

        v = json.loads(verdicts.read_text())

    if v["n_scored"] != len(rows):
        print(f"        mbpp_plus: {v['n_scored']}/{len(rows)} scored -- "
              f"task_ids missing from the dataset were dropped")

    # A clean sweep of zeros is almost never a model result, and this project has
    # now lost time to it three separate ways: a missing tree-sitter made every
    # sample raise; fixed temp-file names let two scorers overwrite each other;
    # and the dataset needed a mid-benchmark download. Each time the number said
    # only "0.0000", which is indistinguishable from total quality collapse.
    # So when nothing passes, say what the scorer actually saw.
    if v["n_scored"] and not any(v["verdicts"].values()):
        blank = v.get("n_unsanitizable", 0)
        # evalplus writes tqdm progress bars to stderr, so "stderr is non-empty"
        # is not a signal. Only keep lines that look like a real problem.
        noise = ("it/s", "it [", "%|", "\r")
        bad = [l for l in proc.stderr.splitlines()
               if l.strip() and not any(n in l for n in noise)]
        print(f"        mbpp_plus: 0/{v['n_scored']} passed -- verify this is the "
              f"model and not the probe")
        print(f"          unsanitizable: {blank}/{v['n_scored']}"
              + ("  <- NOTHING parsed as code, this is a probe failure"
                 if blank == v["n_scored"] else "  (generations did parse)"))
        if bad:
            print(f"          scorer said  : {' | '.join(bad)[-300:]}")
        print(f"          inspect      : python diagnose_mbpp.py <run-dir>")
    # Back into ROW ORDER. mbpp_score runs a process pool and returns verdicts as
    # they complete, so its dict order is arrival order, not row order. Callers
    # zip these against rows to find which problem flipped -- returning them
    # unordered would attribute every flip to the wrong problem.
    return [bool(v["verdicts"].get(r["task_id"], False)) for r in rows]


def _judge_humaneval_plus(rows, texts) -> list[bool]:
    """Still not wired -- but no longer for want of a sandbox.

    The blocker named here for a year ("pass@1 needs sandboxed execution") is
    solved: mbpp_score.py is that runner, and evalplus serves HumanEval+ from the
    same API (`get_human_eval_plus`, and `untrusted_check` already takes the
    dataset as its first argument). Finishing this is roughly: teach
    mbpp_score.py a --dataset flag, add a fetch_human_eval_plus to fetch_data.py,
    and register the benchmark.

    It stays unregistered because nothing has asked for it. MBPP+ covers code
    generation, and a second code benchmark measuring the same axis costs a full
    ladder of GPU time to tell us what the first one already did.
    """
    raise NotImplementedError(
        f"humaneval_plus has {len(rows)} rows to judge but is not wired up.\n"
        f"The sandbox exists now -- see mbpp_score.py -- so this is a small job, "
        f"not a blocked one. Use mbpp_plus for code quality in the meantime.")


@dataclass(frozen=True)
class Benchmark:
    """One benchmark, with its prompt construction in ONE place.

    `prompt` is here rather than inline in the scorer because it used to be in
    both: the scorer built its own text, and the context-length filter in
    run_benchmark guessed at `row["prompt"]`. MATH-500 rows carry `problem`, not
    `prompt`, so the filter raised KeyError after a full two-hour traversal had
    already completed. One source of truth removes the whole class of bug.

    `max_tokens` is here for the same reason -- the filter needs to reserve room
    for the GENERATION, and MATH-500 asks for 1024 tokens where RULER asks 64.
    A fixed reserve would be wrong for one of them.
    """
    judge: Callable
    """(rows, texts) -> [bool], one verdict per row, IN ROW ORDER.

    Judging is separated from generation so that the two callers who need it --
    run_benchmark here, and eval_repro's score_once, which generates
    non-streaming with its own semaphore -- share one implementation. They did
    not: score_once sniffed the row shape ("answers" in r) to pick between
    RULER's and MATH-500's rule, a third copy of logic that already lives here.
    A code benchmark makes that untenable, since its judge is a subprocess.
    """
    prompt: Callable
    metric: str
    n_full: int
    max_tokens: int
    chat: bool = False
    """Whether to send the prompt through the model's chat template.

    False for math_500 and ruler on purpose. Every accuracy number this project
    has recorded was measured raw, and turning the template on for them would
    make new rows incomparable to old ones for a reason unrelated to the config
    being tested. mbpp_plus is new, so it starts on the correct setting.
    """


BENCHMARKS: dict[str, Benchmark] = {
    "math_500": Benchmark(_judge_math_500, _math_500_prompt, "exact_match", 500, 1024),
    "ruler_multineedle": Benchmark(_judge_ruler, _ruler_prompt, "accuracy", 200, 64),
    # max_tokens 512: measured p95 output is 73 tokens and the longest seen was
    # 142, so this is ~3.5x the worst observed. It is a runaway cap, not a
    # target. It matters most if a model with no chat template falls back to raw
    # completion, where nothing emits a stop token and every generation runs to
    # the cap.
    "mbpp_plus": Benchmark(_judge_mbpp_plus, _mbpp_plus_prompt, "pass@1", 378, 512,
                           chat=True),
    "humaneval_plus": Benchmark(_judge_humaneval_plus, _ruler_prompt, "pass@1", 164, 512),
}


def run_benchmark(name: str, gen: Generate, *, full: bool = False,
                  max_input_tokens: int | None = None,
                  model: str | None = None) -> float:
    """Score one benchmark, refusing to score prompts the server cannot take.

    RULER generates long documents. Served under a right-sized max_model_len,
    every prompt can exceed the context and be rejected, and the benchmark then
    returns 0.0 -- for EVERY config, so the gate reads "quality unchanged"
    instead of "probe broken". That is the same shape as the accuracy gate that
    could never pass earlier in this project, so it fails loudly here instead.
    """
    if name not in BENCHMARKS:
        raise KeyError(f"unknown benchmark {name!r}; have {', '.join(BENCHMARKS)}")
    b = BENCHMARKS[name]
    rows = _load(name, None if full else TRAVERSAL_N)

    # ONE prompt builder from here down -- the context filter and the scorer must
    # see identical text. They did not once before: the scorer built its own
    # string while the filter guessed at row["prompt"], and MATH-500 rows carry
    # "problem", so a full traversal died on KeyError after it had finished.
    prompt = _chat_wrapper(b.prompt, model) if b.chat else b.prompt

    if max_input_tokens:
        # Reserve room for the generation: the prompt plus what the model is
        # asked to produce must both fit inside max_model_len.
        budget = max_input_tokens - b.max_tokens
        est = lambda r: len(prompt(r)) // 4
        if budget <= 0:
            raise ValueError(
                f"{name}: max_model_len={max_input_tokens} leaves no room for a "
                f"{b.max_tokens}-token generation. Nothing can be scored.")
        fits = [r for r in rows if est(r) < budget]
        if not fits:
            raise ValueError(
                f"{name}: every prompt exceeds the served context. Shortest is "
                f"~{min(est(r) for r in rows)} tokens and the budget is {budget} "
                f"(max_model_len {max_input_tokens} minus {b.max_tokens} for the "
                f"generation). Every prompt would be rejected and the benchmark "
                f"would score 0.0 for every config -- indistinguishable from "
                f"'no quality change'.\n"
                f"Regenerate at a context that fits:\n"
                f"    python fetch_data.py --ruler-contexts {budget//2},{int(budget*0.9)}")
        if len(fits) < len(rows):
            print(f"        {name}: {len(rows)-len(fits)}/{len(rows)} prompts exceed "
                  f"the {budget}-token budget, scoring on {len(fits)}")
        rows = fits

    # max_tokens comes from the Benchmark, not from a literal inside each scorer.
    # It was in both places -- Benchmark.max_tokens sized the context filter while
    # the scorer passed its own constant to gen() -- so changing one silently left
    # the filter reserving room for a generation length nothing would produce.
    outs = gen([prompt(r) for r in rows], b.max_tokens)
    verdicts = b.judge(rows, [o.text for o in outs])
    return round(sum(verdicts) / len(verdicts), 4)


def resolution(name: str, *, full: bool = False) -> float:
    """Smallest delta the sample size can express. Report it alongside the
    score: claiming a 0.5% regression on 100 items is claiming half an item."""
    n = BENCHMARKS[name].n_full if full else TRAVERSAL_N
    return 1.0 / n
