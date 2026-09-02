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
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKGS = HERE / ".evalplus-pkgs"


def _configure_memory_guard() -> str:
    """Make evalplus's rlimit guard applicable on THIS shell, or turn it off loudly.

    evalplus caps each execution with setrlimit on RLIMIT_AS, RLIMIT_DATA and
    RLIMIT_STACK, all to the same value (4 GiB by default, or
    EVALPLUS_MAX_MEMORY_BYTES). Setting a HARD limit above the current hard
    limit is not permitted, so on a shell where any of those three is capped
    below 4 GiB -- RLIMIT_STACK commonly is -- every execution raises

        ValueError: not allowed to raise maximum limit

    and every sample fails. The score is then 0.0000 with the generations
    perfectly fine, which is indistinguishable from total quality collapse. It
    is also environment-dependent: identical code passes in a shell with
    unlimited hard limits and fails in one without, which is exactly the kind of
    difference that gets blamed on the model.

    So: pick the largest cap this shell can actually apply. If even a reduced
    cap is impossible, disable the guard (-1) and say so, because a missing
    memory cap is a much smaller problem than a benchmark that always reads
    zero. The guard was never containment anyway -- see the module docstring --
    it stops runaway generations, and the execution timeout still does that.
    """
    import resource

    want = int(os.environ.get("EVALPLUS_MAX_MEMORY_BYTES", 4 * 1024 ** 3))
    if want == -1:
        return "disabled by caller"

    hards = []
    for name in ("RLIMIT_AS", "RLIMIT_DATA", "RLIMIT_STACK"):
        r = getattr(resource, name, None)
        if r is None:
            continue
        _, hard = resource.getrlimit(r)
        if hard != resource.RLIM_INFINITY:
            hards.append(hard)

    if not hards or min(hards) >= want:
        return ""                                # the default already applies

    usable = min(hards)
    # A cap this small would fail legitimate solutions rather than protect
    # anything, so below ~256 MiB it is not worth having.
    if usable >= 256 * 1024 ** 2:
        os.environ["EVALPLUS_MAX_MEMORY_BYTES"] = str(usable)
        msg = (f"memory guard reduced to {usable/1024**3:.2f} GiB, this shell's "
               f"hard rlimit (evalplus defaults to 4 GiB)")
        print(f"  {msg}")
        return msg
    else:
        os.environ["EVALPLUS_MAX_MEMORY_BYTES"] = "-1"
        own = _install_own_cap()
        msg = (f"evalplus's memory guard is unusable here (hard rlimit "
               f"{usable/1024**2:.0f} MiB, it wants 4 GiB); {own}")
        print(f"  {msg}")
        return msg


def _install_own_cap() -> str:
    """Cap address space ourselves, since evalplus's guard could not be applied.

    This works where evalplus's does not for one reason: it only LOWERS the soft
    limit of RLIMIT_AS, which is always permitted, where evalplus tries to raise
    the HARD limit of three separate limits to the same 4 GiB -- absurd for
    RLIMIT_STACK, and forbidden when the hard limit is lower. Worker processes
    inherit rlimits, so a cap set here applies to every execution.

    Leaving it uncapped is not a neutral choice on this hardware. Ten workers run
    concurrently, and while a slow allocation LOOP is bounded by evalplus's
    few-second timeout, a single large allocation is not -- time_limit uses
    signals, which cannot interrupt a C-level allocation in flight. MBPP+ is full
    of combinatorial problems (Mbpp/255 and Mbpp/630 time out on their own
    reference solutions), and a model-written `list(permutations(range(15)))`
    asks for 1.3 trillion tuples in one call. On a unified-memory part that
    competes with the GPU: exhausting host RAM can take down the vLLM server
    holding the model, not just the test process.

    The cap is per process, so the worst case is n_workers x cap rather than the
    whole machine. Sized to leave the box usable, and generous enough that the
    parent can still hold the groundtruth table.
    """
    import resource
    try:
        import psutil
        total = psutil.virtual_memory().total
    except Exception:
        total = 16 * 1024 ** 3
    workers = max(1, (os.cpu_count() or 4) // 2)

    # SIZED FROM MEASUREMENT, and the numbers argue for a SMALL cap:
    #
    #   largest MBPP+ test-input set          0.19 MB   (Mbpp/301)
    #   median across all 378 problems        0.004 MB
    #   peak allocation running the heaviest  0.1 MB
    #   parent process holding groundtruth    1.85 GB   <- the real floor
    #
    # No legitimate solution needs even a megabyte; these are one-line problems.
    # The floor is set by the PARENT, which holds the groundtruth table, not by
    # anything the generated code does.
    #
    # Raising the cap does not help anything and costs the guard its point: at
    # 20 GiB, ten workers could ask for 200 GiB on a 131 GiB machine, so the cap
    # could no longer prevent exhaustion. A ceiling is not a reservation --
    # capping at 6 GiB does not consume 6 GiB, and healthy usage stays near
    # 0.2 GB per worker.
    #
    # Override with INFEROPT_MBPP_MEM_CAP_GB if a workload genuinely needs it.
    env_cap = os.environ.get("INFEROPT_MBPP_MEM_CAP_GB")
    if env_cap:
        per = int(float(env_cap) * 1024 ** 3)
    else:
        # Half the machine shared across workers, clamped to [3, 8] GiB: above
        # the 1.85 GB floor with real headroom, far below what hurts.
        per = int(min(8 * 1024 ** 3, max(3 * 1024 ** 3, total * 0.5 / workers)))
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    if hard != resource.RLIM_INFINITY:
        per = min(per, hard)
    try:
        resource.setrlimit(resource.RLIMIT_AS, (per, hard))
        return (f"capped address space at {per/1024**3:.1f} GiB per process "
                f"instead ({workers} workers)")
    except (ValueError, OSError) as e:
        return (f"and our own cap failed too ({type(e).__name__}) -- executions "
                f"are bounded only by the timeout")


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
            subset = Path(tempfile.mkdtemp(prefix="mbpp-ds-")) / "subset.jsonl"
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

    guard = _configure_memory_guard()

    raw_samples = [json.loads(l) for l in
                   Path(a.samples).read_text().splitlines() if l.strip()]
    problems = get_mbpp_plus()

    # EVERY intermediate goes in a private temp directory. They used to be fixed
    # names in the repo root -- .mbpp-samples.jsonl and the
    # .mbpp-samples_eval_results.json that evaluate() derives from it -- which
    # made two concurrent invocations silently corrupt each other. That is not
    # hypothetical: a seven-config ladder ran while a selftest was scoring in the
    # same checkout, the ladder read the selftest's results file, found none of
    # its own task_ids, and recorded 0.0000 for all seven configs with no error
    # anywhere. Identical code re-judging the same generations afterwards gave
    # 0.7169. A scorer must be safe to run twice at once in one directory.
    tmpdir = tempfile.TemporaryDirectory(prefix="mbpp-score-")
    work = Path(tmpdir.name)

    # Sanitize into evalplus's expected shape: {"task_id", "solution"}, where
    # solution is standalone runnable code. This is what `evalplus.sanitize`
    # does; sanitize() walks a tree-sitter parse and keeps the entry-point
    # function plus what it depends on, which handles the markdown fence, the
    # surrounding explanation, and helper functions defined first.
    #
    # A sample that cannot be sanitized becomes the empty string rather than an
    # exception: it must fail as one sample, never abort the other 377.
    prepared = work / "samples.jsonl"
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
        "memory_guard": guard,
    }
    Path(a.out).write_text(json.dumps(out, indent=2))
    tmpdir.cleanup()

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
