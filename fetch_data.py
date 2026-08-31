"""Materialize inferopt/data/. CPU and network only -- safe to run beside a GPU job.

Requires `datasets`.

    python fetch_data.py

  math_500           HuggingFaceH4/MATH-500, exact_match on the boxed answer
  ruler_multineedle  GENERATED. RULER ships a generator, not a corpus, and the
                     multi-query variant is used on purpose: single-needle NIAH
                     saturates at 1.00 on a 9B model and cannot show a
                     regression, which is what made the previous guard useless.

HISTORY

  RULER context lengths were hardcoded at 16384/32768, which exceeded every
  served max_model_len this project has used. See the history in quality.py for
  what that does to a quality gate. --ruler-contexts makes them fit the context
  the server will actually run.

  RULER ships a generator, not a corpus, so the haystack is built here rather
  than downloaded. Multi-needle on purpose: the single-needle variant saturates
  at 1.00 and cannot show a regression.

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
N_RULER = 200
NEEDLES_PER_DOC = 4
CONTEXTS = (16384, 32768)
DEPTHS = (0.12, 0.37, 0.62, 0.88)


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
    """MBPP+ prompts. The TESTS deliberately stay in evalplus's own cache.

    Only what is needed to build a prompt is copied here -- task_id, the
    docstring, and the entry point. The 100-plus test inputs per problem stay
    where evalplus put them, because scoring runs in a different process with a
    different dependency set (see mbpp_score.py) and duplicating the test data
    into data/ would create a second copy to drift out of sync with it.

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
    code = ("import json,sys;from evalplus.data import get_mbpp_plus;"
            "print(json.dumps([{'task_id':k,'prompt':v['prompt'],"
            "'entry_point':v['entry_point'],'assertion':v.get('assertion','')}"
            " for k,v in get_mbpp_plus().items()]))")
    r = subprocess.run([sys.executable, "-c", code],
                       capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": str(pkgs)}, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"evalplus failed to load MBPP+:\n{r.stderr[-2000:]}")
    rows = json.loads(r.stdout.strip().splitlines()[-1])
    _write("mbpp_plus", rows)
    print(f"    {len(rows)} problems; tests stay in evalplus's cache and are read "
          f"by mbpp_score.py")
    print(f"    scored by EXECUTING generated code -- pass@1, in a subprocess")


def _haystack(rng: random.Random) -> list[str]:
    """Long benign prose. Needs to be text the model has no reason to have
    memorised alongside the needles."""
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    ds = ds.select(range(min(300_000, len(ds))))
    chunks = [t for t in ds["text"] if len(t) > 200]
    if not chunks:
        raise RuntimeError("wikitext returned no usable chunks")
    rng.shuffle(chunks)
    return chunks


def build_ruler_multineedle(seed: int = 0) -> None:
    """RULER multi-query NIAH: plant several key/value pairs, ask for all of them.

    Scored all-or-nothing per document. Retrieving 3 of 4 is a failure, because
    partial recall under KV-cache quantization is exactly the damage this is
    meant to expose -- averaging it away would hide the thing being measured.
    """
    rng = random.Random(seed)
    chunks = _haystack(rng)
    rows, cursor = [], 0
    per_ctx = N_RULER // len(CONTEXTS) + 1

    for n_tok in CONTEXTS:
        for _ in range(per_ctx):
            keys = rng.sample(range(1000, 9999), NEEDLES_PER_DOC)
            vals = [f"{rng.randint(100000, 999999)}" for _ in range(NEEDLES_PER_DOC)]
            body, target = [], n_tok * 4              # ~4 chars per token
            while sum(len(c) for c in body) < target:
                body.append(chunks[cursor % len(chunks)])
                cursor += 1
            text = "".join(body)

            # Plant at fixed depths so a failure is attributable to position
            # rather than to where the sampler happened to put them.
            out, prev = [], 0
            for d, k, v in zip(DEPTHS, keys, vals):
                cut = int(len(text) * d)
                out.append(text[prev:cut])
                out.append(f"\nThe access code for vault {k} is {v}.\n")
                prev = cut
            out.append(text[prev:])
            doc = "".join(out)

            asked = ", ".join(str(k) for k in keys)
            rows.append({
                "prompt": (
                    "Read the document and answer the question at the end.\n\n"
                    f"<document>\n{doc}\n</document>\n\n"
                    f"Question: list the access codes for vaults {asked}, in that order, "
                    f"separated by commas. Numbers only.\nAnswer:"),
                "answers": vals,
                "n_tokens_approx": n_tok,
                "n_needles": NEEDLES_PER_DOC,
            })
    rng.shuffle(rows)
    _write("ruler_multineedle", rows[:N_RULER])


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="fetch_data")
    ap.add_argument("--ruler-contexts", default=None,
                    help="comma-separated context lengths in tokens (default 16384,32768). "
                         "These must fit under the max_model_len the server will actually "
                         "run, or every prompt is rejected and the benchmark scores 0.0 for "
                         "every config -- which reads as 'no quality change'.")
    a = ap.parse_args()
    if a.ruler_contexts:
        global CONTEXTS
        CONTEXTS = tuple(int(x) for x in a.ruler_contexts.split(","))
        print(f"ruler contexts overridden -> {CONTEXTS}")

    failed = []
    for label, fn in (("math_500  (HuggingFaceH4/MATH-500)", fetch_math_500),
                      ("mbpp_plus  (evalplus MBPP+)", fetch_mbpp_plus),
                      ("ruler_multineedle  (generated)", build_ruler_multineedle)):
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
