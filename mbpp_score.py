"""Execute model-written Python against MBPP+'s tests. THE SANDBOX BOUNDARY.

    python mbpp_score.py --samples samples.jsonl --out verdicts.json

Runs as a SUBPROCESS, never imported into the serving process, and that is the
entire point of the file existing separately. Two independent reasons:

  ISOLATION. This module executes code a language model wrote. quality.py's
  `_humaneval_plus` refused to do that for a year with the note "wiring this to
  bare exec() would be the single most dangerous line in this project", and that
  judgement stands. A separate process is a real boundary: it can be run inside
  a container, given its own rlimits, and killed without taking the traversal
  with it. Importing evalplus into the process that owns the vLLM server would
  put model-written code one `exec` away from the GPU job and the run directory.

  DEPENDENCIES. evalplus pulls openai, anthropic, google-generativeai and
  tree-sitter for ITS generation backends, none of which we use -- we generate
  with vLLM. Installing that into the serving environment risks the dependency
  breakage this project has already paid for twice (modelopt pulled
  setuptools 81 and vLLM requires <81; ninja vanished from three subprocesses
  for want of child_env). So evalplus lives in .evalplus-pkgs/ and is reached
  only through PYTHONPATH, here.

WHAT THE ISOLATION IS AND IS NOT

evalplus's guards are `RLIMIT_AS`, `RLIMIT_CPU`, signal timeouts, and
monkeypatched os.system/subprocess. That stops accidents -- infinite loops,
memory bombs, a stray open('w'). It is NOT containment against an adversary; a
determined exploit escapes it. That is an acceptable threat model here because
the code is written by a quantized Qwen3-14B answering "write a function to find
shared elements from two lists", not by an attacker. If that ever stops being
true, run this whole file in a container: the boundary is already in the right
place, at batch granularity rather than per problem. Containerising per
execution -- which is what llm-sandbox does -- would pay container startup ~7000
times for a single six-variant ladder, and startup would dominate the run.

NOTHING HERE TOUCHES THE NETWORK

The dataset is read from data/mbpp_plus_full.jsonl via MBPP_OVERRIDE_PATH, and
expected outputs are computed locally by executing the canonical solutions --
about 30s cold, then cached. That is deliberate and was learned the hard way:
the tests originally stayed in evalplus's machine-local cache, so on a second
machine the first ladder run tried to DOWNLOAD MBPP+ in the middle of judging
and died on an SSL certificate failure, after the model had loaded and 378
completions had been generated. A benchmark must not need the network at the
moment it is judging. `rsync data/` now makes another machine work offline.

The groundtruth pickle is ~750MB. It is a cache, not an input: delete it and it
recomputes in 30s.

GROUNDTRUTH IS COMPUTED ON THE FULL SET, ALWAYS

evalplus caches expected outputs in a pickle keyed by a hash of the DATASET, not
of the subset being scored. Computing groundtruth from a 100-problem subset and
storing it under the full dataset's hash would poison that cache for every later
full run -- the second run would load 100 problems' expectations and score the
other 278 against nothing. So groundtruth always covers all 378; only the
scoring loop is subset-aware.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKGS = HERE / ".evalplus-pkgs"


def _extract(raw: str, entry_point: str) -> str:
    """Model output -> runnable code.

    evalplus.sanitize walks a tree-sitter parse and keeps the entry-point
    function plus whatever it actually depends on, which is what handles the
    three things an instruct model reliably does: wraps the answer in a markdown
    fence, writes a paragraph of explanation around it, and defines helpers
    before the function asked for.

    Falls back to fence-stripping if sanitize throws -- it parses Python, and a
    truncated generation (max_tokens reached mid-function) is not valid Python.
    A failed extraction must degrade to "this sample fails", never to an
    exception that aborts scoring for the other 377.
    """
    from evalplus.sanitize import sanitize
    try:
        out = sanitize(raw, entrypoint=entry_point)
        if out.strip():
            return out
    except Exception:
        pass
    if "```" in raw:                       # ```python ... ``` -> the middle
        parts = raw.split("```")
        if len(parts) >= 2:
            body = parts[1]
            if body.lstrip().lower().startswith("python"):
                body = body.lstrip()[6:]
            return body
    return raw


def _check_one(args: tuple) -> tuple[str, bool, str]:
    """One problem, in a worker process. Returns (task_id, passed, status)."""
    from evalplus.eval import untrusted_check, PASS

    task_id, raw, entry_point, inputs, expected, atol, ref_time = args
    code = _extract(raw, entry_point)
    try:
        status, _ = untrusted_check(
            "mbpp", code, inputs, entry_point, expected, atol, ref_time,
            fast_check=True,          # stop at the first failing test
            min_time_limit=1.0,
            gt_time_limit_factor=4.0,
        )
        return task_id, status == PASS, status
    except Exception as e:                       # a crash is a failure, not an abort
        return task_id, False, f"error: {type(e).__name__}: {e}"


def main() -> int:
    ap = argparse.ArgumentParser(prog="mbpp_score")
    ap.add_argument("--samples", required=True,
                    help="jsonl of {task_id, raw} -- raw model output, extracted here")
    ap.add_argument("--out", required=True, help="where to write the verdicts json")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    if PKGS.is_dir() and str(PKGS) not in sys.path:
        sys.path.insert(0, str(PKGS))

    # Point evalplus at the local dataset BEFORE importing it -- MBPP_OVERRIDE_PATH
    # is read at import time, and without it evalplus downloads MBPP+ into a
    # machine-local cache. That download is a network call in the middle of
    # judging: on a fresh machine the first ladder run died on an SSL certificate
    # failure AFTER loading the model and generating 378 completions. Scoring must
    # not need the network.
    full = HERE / "data" / "mbpp_plus_full.jsonl"
    if full.exists():
        os.environ.setdefault("MBPP_OVERRIDE_PATH", str(full))
    elif "MBPP_OVERRIDE_PATH" not in os.environ:
        print(f"{full} not found -- evalplus would try to DOWNLOAD MBPP+ while "
              f"scoring.\nMaterialize it first (costs nothing, needs the network "
              f"once):\n    python fetch_data.py\n"
              f"If this box cannot verify SSL certificates:\n"
              f"    export SSL_CERT_FILE=$(python -m certifi)\n"
              f"or copy the 2.6MB file from a machine that has it:\n"
              f"    rsync <other-host>:<repo>/data/mbpp_plus_full.jsonl {full}",
              file=sys.stderr)
        return 4

    try:
        from evalplus.data import get_mbpp_plus, get_mbpp_plus_hash
        from evalplus.evaluate import get_groundtruth
        from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
    except ImportError as e:
        print(f"evalplus is not importable: {e}\n"
              f"Install it isolated (it must NOT go in the serving env):\n"
              f"    python -m pip install --target .evalplus-pkgs --no-deps \\\n"
              f"        evalplus tempdir appdirs multipledispatch wget termcolor \\\n"
              f"        fire tree-sitter tree-sitter-python", file=sys.stderr)
        return 2

    samples = [json.loads(l) for l in
               Path(a.samples).read_text().splitlines() if l.strip()]
    problems = get_mbpp_plus()

    # Full set, always -- see the module docstring on cache poisoning.
    expected = get_groundtruth(problems, get_mbpp_plus_hash(),
                               MBPP_OUTPUT_NOT_NONE_TASKS)

    # The cache is a pickle keyed by the DATASET hash, so a partial groundtruth
    # written by anything else -- an interactive session, an interrupted run --
    # loads silently and is missing most problems. That surfaced here as a bare
    # KeyError mid-scoring, which is the least useful place to learn it. Check
    # the whole set up front and say exactly what to delete.
    missing = [k for k in problems if k not in expected]
    if missing:
        cache = Path.home() / ".cache" / "evalplus"
        print(f"groundtruth cache holds {len(expected)} of {len(problems)} problems "
              f"(missing e.g. {missing[:3]}).\n"
              f"It was written from a SUBSET but keyed by the full dataset's hash.\n"
              f"Delete it and re-run -- it recomputes in seconds:\n"
              f"    rm -f {cache}/*.pkl", file=sys.stderr)
        return 3

    work = []
    for s in samples:
        tid = s["task_id"]
        if tid not in problems:
            continue
        p, gt = problems[tid], expected[tid]
        work.append((
            tid, s.get("raw", s.get("solution", "")), p["entry_point"],
            p["base_input"] + list(p["plus_input"]),
            list(gt["base"]) + list(gt["plus"]),
            p["atol"], gt["base_time"] + gt["plus_time"],
        ))

    verdicts: dict[str, bool] = {}
    status: dict[str, str] = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(_check_one, w) for w in work]
        for f in as_completed(futs):
            tid, ok, st = f.result()
            verdicts[tid], status[tid] = ok, st

    n = len(verdicts)
    out = {
        "pass_at_1": (sum(verdicts.values()) / n) if n else 0.0,
        "n_scored": n,
        "n_submitted": len(samples),
        "verdicts": verdicts,
        "status": status,
    }
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"  scored {n}/{len(samples)}  pass@1 = {out['pass_at_1']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
