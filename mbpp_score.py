"""Score MBPP+ with evalplus's own harness. THE SANDBOX BOUNDARY.

    python mbpp_score.py --samples samples.jsonl --out verdicts.json

Does exactly what `evalplus.sanitize` + `evalplus.evaluate` do from the command
line, because that IS the official path:

    sanitize(code, entrypoint)                 per sample, what the CLI applies
    evaluate(dataset="mbpp", samples=...)      writes <samples>_eval_results.json
    base_status == pass and plus_status == pass    -> the sample passed MBPP+

An earlier version of this file drove `untrusted_check` directly, computing
groundtruth itself and running its own process pool -- a reimplementation of
`evaluate()` that had to be told about the groundtruth cache, the special MBPP
oracle, expert timing factors and completion ordering. Every one of those was a
chance to be subtly wrong about a number meant to be comparable to published
results, and one of them was: the sanitize import sat outside its try, so a
missing tree_sitter_python raised for every sample, each raise was recorded as a
failed sample, and a whole seven-config ladder reported pass@1 = 0.0000 with
perfect generations on disk. Calling the harness deletes that entire class of
bug. Do not reimplement it again.

THE CEILING IS 0.9947, NOT 1.0

Scoring MBPP+'s own canonical solutions gives 376/378. Mbpp/255 and Mbpp/630
time out on their "plus" inputs -- combinations_with_replacement over 109 extra
tests, and a recursive coordinate generator over 120. evalplus ships a
`noextreme` flag for exactly these. They are left in, at evalplus's default time
limits, because raising the limits would make our numbers incomparable to
published MBPP+ figures and would make the score depend on how loaded the box is.

It costs nothing for our purpose: test execution happens on CPU and is
independent of the model, so those two fail identically for every variant in the
ladder and cancel out of every delta. It only means the top of the scale is
0.9947.

WHY A SUBPROCESS, still

  ISOLATION. This executes code a language model wrote. quality.py refused to do
  that for a year, noting that wiring pass@1 to bare exec() would be the most
  dangerous line in the project, and that judgement stands. A separate process is
  a real boundary: it can be containerised, given its own rlimits, and killed
  without taking the traversal with it.

  DEPENDENCIES. evalplus pulls openai, anthropic, google-generativeai and
  tree-sitter for ITS generation backends, none of which we use -- we generate
  with vLLM. This project has been broken twice by a dependency installed for a
  side feature (modelopt pulled setuptools 81 against vLLM's <81; ninja vanished
  from three subprocesses for want of child_env). So evalplus lives in
  .evalplus-pkgs/ and is reached only through sys.path, here.

WHAT THE ISOLATION IS AND IS NOT

evalplus's guards are RLIMIT_AS, RLIMIT_CPU, signal timeouts and monkeypatched
os.system/subprocess. That stops accidents -- infinite loops, memory bombs, a
stray open('w'). It is NOT containment against an adversary. Acceptable here
because the code is written by a quantized Qwen3-14B answering "write a function
to find shared elements from two lists". If that changes, run this file in a
container: the boundary is already at the right granularity, per batch rather
than per problem. Containerising per execution -- what llm-sandbox does -- would
pay container startup ~7000 times for one six-variant ladder.

NOTHING HERE TOUCHES THE NETWORK

The dataset is read from data/mbpp_plus_full.jsonl via MBPP_OVERRIDE_PATH, which
must be set BEFORE evalplus is imported. Expected outputs are computed locally by
executing the canonical solutions, ~30s cold, then cached under ~/.cache/evalplus
(~750MB; a cache, not an input -- delete it and it recomputes).

Learned the hard way: the tests originally stayed in evalplus's machine-local
cache, so on a second machine the first ladder run tried to DOWNLOAD MBPP+ in the
middle of judging and died on an SSL certificate failure -- after the model had
loaded and 378 completions had been generated. A benchmark must not need the
network at the moment it is judging.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKGS = HERE / ".evalplus-pkgs"


def main() -> int:
    ap = argparse.ArgumentParser(prog="mbpp_score")
    ap.add_argument("--samples", required=True,
                    help="jsonl of {task_id, raw} -- raw model output, sanitized here")
    ap.add_argument("--out", required=True, help="where to write the verdicts json")
    ap.add_argument("--parallel", type=int, default=None,
                    help="evalplus worker count; its default is cpu_count-ish")
    a = ap.parse_args()

    if PKGS.is_dir() and str(PKGS) not in sys.path:
        sys.path.insert(0, str(PKGS))

    # Before the import: MBPP_OVERRIDE_PATH is read at module load.
    #
    # SUBSET SCORING GOES THROUGH THE DATASET, NOT THE SAMPLES. evaluate()
    # asserts len(samples) == len(problems) -- it will not score a subset, and
    # padding the missing problems with empty solutions would put 378 in the
    # pass@1 denominator while only n were attempted. So when a run scores fewer
    # than all of them, point evalplus at a dataset containing exactly those
    # problems. That is what the override is for, the assert is satisfied
    # honestly, and pass@1 is over what was actually attempted.
    #
    # The full set keeps the original path so its ~750MB groundtruth cache is
    # reused; a subset hashes differently and gets its own entry, which is
    # correct rather than wasteful -- mixing them is the cache poisoning this
    # file used to have to defend against by hand.
    full = HERE / "data" / "mbpp_plus_full.jsonl"
    wanted: list[str] = []
    if full.exists():
        wanted = [json.loads(l)["task_id"] for l in
                  Path(a.samples).read_text().splitlines() if l.strip()]
        lines = {json.loads(l)["task_id"]: l
                 for l in full.read_text().splitlines() if l.strip()}
        if set(wanted) < set(lines):
            subset = HERE / ".mbpp-subset.jsonl"
            subset.write_text("\n".join(lines[t] for t in wanted if t in lines) + "\n")
            os.environ["MBPP_OVERRIDE_PATH"] = str(subset)
        else:
            os.environ.setdefault("MBPP_OVERRIDE_PATH", str(full))
    elif "MBPP_OVERRIDE_PATH" not in os.environ:
        print(f"{full} not found -- evalplus would try to DOWNLOAD MBPP+ while "
              f"scoring.\nMaterialize it first (needs the network once):\n"
              f"    python fetch_data.py\n"
              f"If this box cannot verify SSL certificates:\n"
              f"    export SSL_CERT_FILE=$(python -m certifi)\n"
              f"or copy the 2.6MB file from a machine that has it:\n"
              f"    rsync <other-host>:<repo>/data/mbpp_plus_full.jsonl {full}",
              file=sys.stderr)
        return 4

    try:
        from evalplus.data import get_mbpp_plus
        from evalplus.evaluate import evaluate
        from evalplus.eval import PASS
        from evalplus.sanitize import sanitize
    except ImportError as e:
        print(f"evalplus is not importable: {e}\n"
              f"Install it isolated (it must NOT go in the serving env):\n"
              f"    python -m pip install --target .evalplus-pkgs --no-deps \\\n"
              f"        evalplus tempdir appdirs multipledispatch wget termcolor \\\n"
              f"        fire tree-sitter tree-sitter-python", file=sys.stderr)
        return 2

    raw_samples = [json.loads(l) for l in
                   Path(a.samples).read_text().splitlines() if l.strip()]
    problems = get_mbpp_plus()

    # Sanitize into evalplus's expected shape: {"task_id", "solution"}, where
    # solution is standalone runnable code. This is what `evalplus.sanitize`
    # does; sanitize() walks a tree-sitter parse and keeps the entry-point
    # function plus what it depends on, which handles the markdown fence, the
    # surrounding explanation, and helper functions defined first.
    #
    # A sample that cannot be sanitized becomes the empty string rather than an
    # exception: it must fail as one sample, never abort the other 377.
    prepared = HERE / ".mbpp-samples.jsonl"
    n_blank = 0
    with open(prepared, "w") as fh:
        for s in raw_samples:
            tid = s["task_id"]
            if tid not in problems:
                continue
            raw = s.get("raw", s.get("solution", ""))
            try:
                sol = sanitize(raw, entrypoint=problems[tid]["entry_point"])
            except Exception:
                sol = ""
            if not sol.strip():
                n_blank += 1
            fh.write(json.dumps({"task_id": tid, "solution": sol}) + "\n")

    result_path = Path(str(prepared).replace(".jsonl", "_eval_results.json"))
    result_path.unlink(missing_ok=True)      # evaluate() reuses a stale one

    evaluate(dataset="mbpp", samples=str(prepared), parallel=a.parallel)

    if not result_path.exists():
        print(f"evalplus.evaluate wrote no results to {result_path}", file=sys.stderr)
        return 5

    res = json.loads(result_path.read_text())
    verdicts: dict[str, bool] = {}
    status: dict[str, str] = {}
    for tid, runs in res["eval"].items():
        r = runs[0]                          # greedy: one completion per task
        ok = r["base_status"] == PASS and r["plus_status"] == PASS
        verdicts[tid] = ok
        status[tid] = f"base={r['base_status']} plus={r['plus_status']}"

    n = len(verdicts)
    out = {
        "pass_at_1": (sum(verdicts.values()) / n) if n else 0.0,
        "n_scored": n,
        "n_submitted": len(raw_samples),
        "n_unsanitizable": n_blank,
        "verdicts": verdicts,
        "status": status,
    }
    Path(a.out).write_text(json.dumps(out, indent=2))
    prepared.unlink(missing_ok=True)
    result_path.unlink(missing_ok=True)
    (HERE / ".mbpp-subset.jsonl").unlink(missing_ok=True)

    # Every sample producing no code at all is a broken pipeline, not a bad
    # model: a model can genuinely score 0 on MBPP+, but not by emitting
    # unparseable output 378 times. Saying so here is what distinguishes it from
    # a real quality collapse in the ladder's table.
    if n and n_blank == n:
        print(f"NONE of the {n} generations could be sanitized into runnable code.\n"
              f"That is a generation or prompt-format problem, not a quality result. "
              f"Inspect the raw text:\n    python diagnose_mbpp.py <run-dir>",
              file=sys.stderr)
        return 6

    print(f"  scored {n}/{len(raw_samples)}  pass@1 = {out['pass_at_1']:.4f}"
          + (f"  ({n_blank} unsanitizable)" if n_blank else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
