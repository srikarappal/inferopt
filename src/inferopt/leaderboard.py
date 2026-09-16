"""The six Open LLM Leaderboard benchmarks, run against a live serving config.

    from inferopt import leaderboard
    leaderboard.run(["leaderboard_math_hard"], base_url, model, limit=50)

WHY THESE SIX AND NOT OUR OWN. math_500, mbpp_plus and humaneval_plus are ours:
we chose the prompts, the budgets and the graders, which makes them internally
consistent and externally unquotable. Nobody can check a number they cannot
reproduce. These six are the Open LLM Leaderboard's, run through
lm-evaluation-harness, which is the implementation the published numbers come
from. A score here is comparable to a score anyone else publishes.

EACH BENCHMARK DICTATES ITS OWN OUTPUT BUDGET, and they are not close to each
other: leaderboard MATH stops at 1024 generated tokens, IFEval at 1280, and the
four multiple-choice benchmarks generate nothing at all, because they are scored
by comparing the log-likelihood of fixed continuations. A single global
max_tokens is meaningless across that spread, which is why the limit lives on
the task.

PARITY IS A PROPERTY WE TRACK, NOT ASSUME. Raising a budget is sometimes the
right call -- a reasoning model whose traces run thousands of tokens is being
cut off at 1024, and the leaderboard default was set before such models were
common -- but the result is then no longer the leaderboard's measurement. Every
result carries `parity`, false as soon as any limit or shot count is overridden,
so a tuned number cannot be quoted as a leaderboard score by accident.

THE HARNESS RUNS IN ITS OWN INTERPRETER. lm_eval needs transformers, datasets,
sympy, nltk and antlr; vLLM needs its own pinned stack. Putting both in one
environment makes each upgrade a negotiation. So lm_eval lives wherever it
likes, is found through INFEROPT_LMEVAL_PYTHON, and talks to the server over
HTTP like any other client. It never loads the model: it measures the
configuration that is already serving, which is the whole point.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

RUNNER = Path(__file__).resolve().parent / "lmeval_runner.py"
"""Executed by the lm_eval interpreter, by path. See the note in run()."""

MULTIPLE_CHOICE = "multiple_choice"
GENERATE = "generate_until"


@dataclass(frozen=True)
class Task:
    """One leaderboard benchmark, as lm_eval defines it."""
    task: str
    """The lm_eval group id. Groups fan out to subtasks and aggregate."""
    output_type: str
    num_fewshot: int
    max_gen_toks: int | None
    """The task's OWN generation budget. None for multiple-choice, which
    generates nothing: the model scores fixed continuations by log-likelihood,
    so there is no output length to cap."""
    metric: str
    """Which key in lm_eval's result dict is the headline number."""
    subtasks: int
    approx_docs: int
    """Roughly how many documents a full run scores. Cost, not correctness."""
    gated: bool = False
    note: str = ""

    @property
    def generative(self) -> bool:
        return self.output_type == GENERATE

    def metric_key(self) -> str:
        return f"{self.metric},none"


# Values read from lm_eval 0.4.13's own task YAMLs, not from the README.
TASKS: dict[str, Task] = {
    "leaderboard_bbh": Task(
        "leaderboard_bbh", MULTIPLE_CHOICE, 3, None, "acc_norm", 24, 6511,
        note="23 BIG-Bench Hard reasoning tasks where models once trailed humans"),
    "leaderboard_gpqa": Task(
        "leaderboard_gpqa", MULTIPLE_CHOICE, 0, None, "acc_norm", 3, 1192,
        gated=True,
        note="graduate-level biology, physics and chemistry; Google-proof"),
    "leaderboard_mmlu_pro": Task(
        "leaderboard_mmlu_pro", MULTIPLE_CHOICE, 5, None, "acc", 1, 12032,
        note="24 subjects, ten options rather than four"),
    "leaderboard_musr": Task(
        "leaderboard_musr", MULTIPLE_CHOICE, 0, None, "acc_norm", 3, 756,
        note="multistep reasoning over narratives of 1000+ words"),
    "leaderboard_math_hard": Task(
        "leaderboard_math_hard", GENERATE, 4, 1024, "exact_match", 7, 1324,
        note="level-5 MATH only, 4-shot Minerva prompting, sympy-verified"),
    "leaderboard_ifeval": Task(
        "leaderboard_ifeval", GENERATE, 0, 1280, "prompt_level_strict_acc", 1, 541,
        note="verifiable instruction following; strict prompt-level accuracy"),
}

ALL = tuple(TASKS)
GENERATIVE = tuple(k for k, t in TASKS.items() if t.generative)
"""The two that exercise the decode path, and so the two that a quantization
step can actually damage in a way the others cannot see."""


class LeaderboardError(RuntimeError):
    pass


def lmeval_python() -> str | None:
    """The interpreter that has lm_eval, or None.

    Order: an explicit INFEROPT_LMEVAL_PYTHON, then this interpreter, then a
    sibling `lmeval-env` beside the project. Explicit always wins and is never
    quietly replaced, for the same reason rerun_all.sh refuses to substitute a
    different vLLM than the one you named.
    """
    def has_lmeval(py: str) -> bool:
        try:
            return subprocess.run([py, "-c", "import lm_eval"],
                                  capture_output=True, timeout=120).returncode == 0
        except Exception:
            return False

    explicit = os.environ.get("INFEROPT_LMEVAL_PYTHON")
    if explicit:
        if shutil.which(explicit) or Path(explicit).exists():
            if has_lmeval(explicit):
                return explicit
        raise LeaderboardError(
            f"INFEROPT_LMEVAL_PYTHON={explicit} has no importable lm_eval. "
            f"Check with: {explicit} -c 'import lm_eval'")
    for cand in (sys.executable,
                 str(Path(__file__).resolve().parents[3] / "lmeval-env/bin/python")):
        if cand and Path(cand).exists() and has_lmeval(cand):
            return cand
    return None


INSTALL_HINT = (
    "lm-evaluation-harness is not installed in any interpreter this can find.\n"
    "It is intentionally NOT a dependency of inferopt: it pulls transformers,\n"
    "datasets, sympy, nltk and antlr, which do not belong beside a pinned vLLM.\n"
    "Put it in its own environment and point at it:\n\n"
    "  python -m venv lmeval-env\n"
    "  ./lmeval-env/bin/pip install 'lm_eval[api,math,ifeval]' transformers\n"
    "  export INFEROPT_LMEVAL_PYTHON=$PWD/lmeval-env/bin/python\n")


def preflight(names=ALL) -> dict[str, str]:
    """Why each requested task cannot run, empty string if it can.

    Checked before a GPU is held, because discovering that a dataset is gated
    after a two-hour sweep has already paid for the server is the expensive
    order to find out in.
    """
    out: dict[str, str] = {}
    for n in names:
        t = TASKS.get(n)
        if t is None:
            out[n] = f"unknown task; known: {', '.join(ALL)}"
        elif t.gated:
            out[n] = ("dataset is gated on HuggingFace. Accept the terms at "
                      "https://huggingface.co/datasets/Idavidrein/gpqa while "
                      "signed in as the owner of HF_TOKEN, then rerun. "
                      "A valid token is not enough on its own.")
        else:
            out[n] = ""
    return out


def run(names, base_url: str, model: str, *, limit: int | None = None,
        gen_toks: dict[str, int] | None = None,
        num_fewshot: dict[str, int] | None = None,
        max_length: int = 15500, concurrent: int = 16, seed: int = 0,
        out_dir=None, python: str | None = None, timeout_s: float = 21600,
        log=print) -> dict[str, dict]:
    """Score `names` against the server already listening at `base_url`.

    base_url is the server root; /v1/completions is appended. Returns one entry
    per task:

        {"leaderboard_math_hard": {"score": 0.41, "metric": "exact_match",
                                   "parity": True, "max_gen_toks": 1024,
                                   "num_fewshot": 4, "n": 1324, "raw": {...}}}

    `parity` is False as soon as this call overrode a budget or a shot count,
    so the number can still be used and cannot be mistaken for the
    leaderboard's own measurement.
    """
    names = [names] if isinstance(names, str) else list(names)
    blocked = {k: v for k, v in preflight(names).items() if v}
    if blocked:
        raise LeaderboardError("; ".join(f"{k}: {v}" for k, v in blocked.items()))

    py = python or lmeval_python()
    if not py:
        raise LeaderboardError(INSTALL_HINT)

    gen_toks = dict(gen_toks or {})
    num_fewshot = dict(num_fewshot or {})
    url = base_url.rstrip("/")
    if not url.endswith("/v1/completions"):
        url = f"{url}/v1/completions"

    out: dict[str, dict] = {}
    # One subprocess per task rather than one for all of them. A task that
    # cannot load its dataset then costs that task, not the whole set, and the
    # partial results of a long sweep survive.
    for name in names:
        t = TASKS[name]
        toks = gen_toks.get(name, t.max_gen_toks)
        shots = num_fewshot.get(name, t.num_fewshot)
        parity = (toks == t.max_gen_toks) and (shots == t.num_fewshot) and limit is None
        cfg = {
            "tasks": [t.task], "model": model, "base_url": url,
            "limit": limit,
            "num_fewshot": shots if shots != t.num_fewshot else None,
            # Only generative tasks have a budget to set. Passing max_gen_toks
            # to a multiple-choice task is not merely useless: lm_eval warns
            # and it reads as though a limit were in force when none is.
            "gen_kwargs": ({"max_gen_toks": toks} if t.generative and toks else None),
            "max_length": max_length, "concurrent": concurrent, "seed": seed,
        }
        with tempfile.TemporaryDirectory() as td:
            cpath, rpath = Path(td) / "cfg.json", Path(td) / "res.json"
            cpath.write_text(json.dumps(cfg))
            log(f"        leaderboard {name:26s} "
                f"{'gen ' + str(toks) + ' tok' if t.generative else 'logprob'}, "
                f"{shots}-shot, {t.subtasks} subtask(s)"
                + ("" if parity else "  [NOT leaderboard parity]"))
            # BY PATH, never `-m inferopt.lmeval_runner`. The module form
            # imports the inferopt package first, and inferopt/__init__ pulls
            # pydantic, which the lm_eval environment has no reason to carry.
            # Running the file directly never touches the package.
            p = subprocess.run([py, str(RUNNER), str(cpath), str(rpath)],
                               capture_output=True, text=True, timeout=timeout_s)
            if p.returncode != 0 or not rpath.exists():
                tail = (p.stdout + p.stderr).strip().splitlines()[-4:]
                out[name] = {"score": None, "error": " | ".join(tail)[:500],
                             "parity": parity}
                log(f"          FAILED: {out[name]['error'][:160]}")
                continue
            res = json.loads(rpath.read_text())

        scores = res.get("results") or {}
        row = scores.get(t.task) or {}
        score = row.get(t.metric_key())
        out[name] = {"score": score, "metric": t.metric, "parity": parity,
                     "max_gen_toks": toks if t.generative else None,
                     "num_fewshot": shots, "limit": limit,
                     "n": t.approx_docs if limit is None else limit,
                     "raw": {k: v for k, v in row.items() if "stderr" not in k}}
        if out_dir:
            d = Path(out_dir); d.mkdir(parents=True, exist_ok=True)
            (d / f"{name}.json").write_text(json.dumps(res, indent=2, default=str))
        log(f"          {name:26s} {t.metric} "
            + (f"{score:.4f}" if isinstance(score, float) else str(score))
            + ("" if parity else "   (not leaderboard parity)"))
    return out


def describe() -> str:
    """The table, for a --help or a run banner."""
    w = [f"  {'task':26s} {'type':16s} {'shots':>5s} {'gen tok':>8s} "
         f"{'subtasks':>8s} {'docs':>6s}  metric",
         "  " + "-" * 92]
    for n, t in TASKS.items():
        w.append(f"  {n:26s} {t.output_type:16s} {t.num_fewshot:5d} "
                 f"{(str(t.max_gen_toks) if t.max_gen_toks else '-'):>8s} "
                 f"{t.subtasks:8d} {t.approx_docs:6d}  {t.metric}"
                 + ("   [gated]" if t.gated else ""))
    return "\n".join(w)
