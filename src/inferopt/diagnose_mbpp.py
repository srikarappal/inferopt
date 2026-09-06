"""Why did an MBPP+ run score 0.0000? Reads generations already on disk.

    python diagnose_mbpp.py runs/ladder-mbpp_plus/q_stock

An all-zero column across every config, with zero spread, is a BROKEN PROBE, not
a result -- the same shape as the RULER bug where every prompt exceeded the
served context and every config scored 0.0, reading as "quality unchanged".

There are only a few ways it happens, and they are distinguishable from the raw
generations without re-running anything on the GPU:

  the model wrote nothing         empty or whitespace text
  the model is THINKING           text starts with <think> and never closes
  the model wrote prose           no code fence, no `def`
  the model wrote fine but the
  JUDGE could not run             text is good code, yet verdicts are all False

The last one is the dangerous case, because the generations look perfect and the
number looks like catastrophic quality loss. It is what happens if the scoring
subprocess cannot import its own dependencies -- every problem raises, and each
raise is recorded as a failed sample.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    d = Path(sys.argv[1])
    gens = sorted(d.glob("generations-*.jsonl"))
    if not gens:
        print(f"  no generations-*.jsonl under {d}")
        return 1

    rows = [json.loads(l) for l in gens[0].read_text().splitlines() if l.strip()]
    rep0 = [r for r in rows if r.get("repeat") == 0] or rows
    print(f"\n  {gens[0]}   {len(rep0)} generations (repeat 0)\n")

    empty = sum(1 for r in rep0 if not r["text"].strip())
    think = sum(1 for r in rep0 if "<think>" in r["text"])
    unclosed = sum(1 for r in rep0 if "<think>" in r["text"] and "</think>" not in r["text"])
    fenced = sum(1 for r in rep0 if "```" in r["text"])
    hasdef = sum(1 for r in rep0 if "def " in r["text"])
    correct = sum(1 for r in rep0 if r.get("correct"))
    lens = sorted(len(r["text"]) // 4 for r in rep0)

    n = len(rep0)
    print(f"    empty output            {empty:4d}/{n}")
    print(f"    contains <think>        {think:4d}/{n}   (unclosed: {unclosed})")
    print(f"    contains a code fence   {fenced:4d}/{n}")
    print(f"    contains 'def '         {hasdef:4d}/{n}")
    print(f"    judged correct          {correct:4d}/{n}")
    print(f"    output tokens ~         p50 {lens[len(lens)//2]}, "
          f"p95 {lens[int(.95*len(lens))]}, max {lens[-1]}")

    print(f"\n  --- first generation, verbatim ---")
    print("    " + repr(rep0[0]["text"][:600]))

    # The decisive test: re-judge a handful and read the STATUS strings, which
    # record why each sample failed rather than just that it did.
    print(f"\n  --- re-judging 5 samples to read the failure reason ---")
    here = Path(__file__).resolve().parent
    tmp = here / ".diag-samples.jsonl"
    out = here / ".diag-verdicts.json"
    sample = [r for r in rep0 if r.get("answer")][:5]
    if not sample:
        print("    generations carry no task_id (answer field) -- cannot re-judge")
        return 0
    tmp.write_text("\n".join(
        json.dumps({"task_id": r["answer"], "raw": r["text"]}) for r in sample) + "\n")
    p = subprocess.run([sys.executable, str(here / "mbpp_score.py"),
                        "--samples", str(tmp), "--out", str(out)],
                       capture_output=True, text=True, timeout=1800, cwd=here)
    if out.exists():
        v = json.loads(out.read_text())
        for tid, st in v["status"].items():
            print(f"    {tid:12s} {'PASS' if v['verdicts'][tid] else 'FAIL'}   {st[:150]}")
        reasons = Counter(v["status"].values())
        print(f"\n    reasons: {dict(reasons)}")
        blank = v.get("n_unsanitizable", 0)
        if blank == v["n_scored"]:
            print(f"\n    NONE of the generations could be sanitized into runnable code.")
            print(f"    That is a generation/prompt-format problem, not a quality result --")
            print(f"    look at the verbatim text above. If it starts with <think>, the")
            print(f"    chat template did not disable thinking and every generation is")
            print(f"    reasoning that hit max_tokens before writing any code.")
        elif blank:
            print(f"\n    {blank}/{v['n_scored']} generations were unsanitizable.")
    else:
        print(f"    scorer produced nothing (exit {p.returncode})")
        print(f"    {p.stderr[-1200:]}")
    tmp.unlink(missing_ok=True)
    out.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
