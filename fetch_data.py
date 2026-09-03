"""Materialize inferopt/data/. CPU and network only -- safe to run beside a GPU job.

Requires `datasets`.

    python fetch_data.py

  math_500           HuggingFaceH4/MATH-500, exact_match on the boxed answer
  mbpp_plus          evalplus MBPP+, pass@1 by executing generated code

HISTORY

  ruler_multineedle was REMOVED. It never earned its place. On Qwen3-14B it
  saturated at 1.00 and could not show a regression -- the whole reason the
  multi-needle variant was chosen over vanilla NIAH. On Qwen3-30B-A3B it read
  0.05, which is not a plausible score for a 30B model retrieving four numbers
  from a 2-4k context, and then MOVED to 0.11 across a lossless node, where
  quality cannot change by construction. Saturated on one model and incoherent
  on the other is not a probe. It also carried an operational hazard: its
  contexts are generated, and regenerating them at the default 16k/32k silently
  produced a corpus where 0 of 200 prompts fit a right-sized max_model_len,
  which would have scored 0.0 for every config and read as "quality unchanged".

  cais/hle was removed. It kept re-downloading and is not the right eval for a
  serving-configuration search.
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

DATA = Path(__file__).parent / "data"



def _write(name: str, rows: list[dict]) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / f"{name}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"  wrote {name}.jsonl  ({len(rows)} rows)")


def fetch_math_500() -> None:
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    rows = [{"problem": r["problem"], "answer": r["answer"],
             "subject": r.get("subject"), "level": r.get("level")} for r in ds]
    _write("math_500", rows)
    print(f"    answers are LaTeX, not numbers (e.g. {rows[0]['answer']!r}) -- "
          f"scored by normalized string match, not numeric extraction")


def fetch_mbpp_plus() -> None:
    """MBPP+ prompts AND tests, both into data/. Nothing is left in a machine cache.

    Writes two files:
      mbpp_plus.jsonl       prompts, for building requests
      mbpp_plus_full.jsonl  the whole dataset including every test input, which
                            mbpp_score.py points evalplus at via MBPP_OVERRIDE_PATH

    THE TESTS USED TO STAY IN evalplus's OWN CACHE, on the reasoning that copying
    them into data/ would create a second copy to drift out of sync. That was
    wrong twice over. The file is 2.6MB, so there was nothing to save. And it
    made SCORING depend on the network: on a second machine the cache is empty,
    so the first eval_repro run tried to download MBPP+ mid-benchmark and died on
    an SSL certificate failure -- after the model had loaded and 378 generations
    had already been produced. A benchmark must not need the network at the
    moment it is judging.

    Now the dependency is here, at fetch time, where a failure costs nothing and
    where `rsync data/` is enough to make another machine work offline.

    evalplus is imported through .evalplus-pkgs rather than the serving
    environment: it pulls openai, anthropic and google-generativeai for its own
    generation backends, and this project has twice been broken by a dependency
    installed for a side feature.
    """
    import subprocess
    pkgs = Path(__file__).parent / ".evalplus-pkgs"
    if not pkgs.is_dir():
        raise RuntimeError(
            f"{pkgs} not found. Install evalplus isolated -- NOT into the serving "
            f"environment:\n"
            f"    python -m pip install --target .evalplus-pkgs --no-deps \\\n"
            f"        evalplus tempdir appdirs multipledispatch wget termcolor \\\n"
            f"        fire tree-sitter tree-sitter-python")

    # A subprocess, not an import: putting .evalplus-pkgs on this process's
    # sys.path would shadow the serving env's transformers/datasets for whatever
    # runs next in the same interpreter.
    #
    # Reuse an already-materialized copy if there is one, so re-running this is
    # free and works with no network at all.
    full = DATA / "mbpp_plus_full.jsonl"
    code = ("import json,os,shutil,sys;"
            "from evalplus.data.mbpp import _ready_mbpp_plus_path;"
            "from evalplus.data import get_mbpp_plus;"
            "p=_ready_mbpp_plus_path();"
            "d=json.dumps([{'task_id':k,'prompt':v['prompt'],"
            "'entry_point':v['entry_point'],'assertion':v.get('assertion','')}"
            " for k,v in get_mbpp_plus().items()]);"
            "shutil.copyfile(p, os.environ['FULL_OUT']) "
            "if os.path.abspath(p)!=os.path.abspath(os.environ['FULL_OUT']) else None;"
            "print(d)")
    env = {**os.environ, "PYTHONPATH": str(pkgs), "FULL_OUT": str(full)}
    if full.exists():
        env["MBPP_OVERRIDE_PATH"] = str(full)
    DATA.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([sys.executable, "-c", code],
                       capture_output=True, text=True, env=env, timeout=600)
    if r.returncode != 0:
        hint = ""
        if "CERTIFICATE_VERIFY_FAILED" in r.stderr:
            hint = (f"\n\n  This box cannot verify the download's SSL certificate. "
                    f"Either point it at a CA bundle:\n"
                    f"      export SSL_CERT_FILE=$(python -m certifi)\n"
                    f"  or copy the 2.6MB dataset from a machine that already has it:\n"
                    f"      rsync <other-host>:<repo>/data/mbpp_plus_full.jsonl {full}\n"
                    f"      rsync <other-host>:~/.cache/evalplus/MbppPlus-*.jsonl {full}\n"
                    f"  then re-run. Once {full.name} exists, nothing here touches the "
                    f"network again.")
        raise RuntimeError(f"evalplus failed to load MBPP+:\n{r.stderr[-1500:]}{hint}")
    rows = json.loads(r.stdout.strip().splitlines()[-1])
    _write("mbpp_plus", rows)
    print(f"    {len(rows)} problems -> mbpp_plus.jsonl (prompts)")
    print(f"    full dataset incl. tests -> {full.name} "
          f"({full.stat().st_size/1e6:.1f} MB), read by mbpp_score.py")
    print(f"    scoring is now OFFLINE -- rsync data/ and another machine works")
    print(f"    scored by EXECUTING generated code -- pass@1, in a subprocess")


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="fetch_data")
    a = ap.parse_args()

    failed = []
    for label, fn in (("math_500  (HuggingFaceH4/MATH-500)", fetch_math_500),
                      ("mbpp_plus  (evalplus MBPP+)", fetch_mbpp_plus)):
        print(f"{label} ...")
        try:
            fn()
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            failed.append(label)
    print()
    if failed:
        print(f"incomplete: {', '.join(failed)}")
        return 1
    print(f"data ready under {DATA}")
    print("note: humaneval_plus is still absent -- mbpp_plus now covers code via")
    print("      evalplus, executing in a subprocess (mbpp_score.py). humaneval_plus")
    print("      would work the same way but has not been wired up.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
