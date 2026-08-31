"""Re-judge generations already on disk. No GPU, no re-generation.

    python rescore.py runs/ladder-mbpp_plus

Exists because a scoring bug costs a whole ladder otherwise. eval_repro writes
every generation to generations-*.jsonl precisely so a summary statistic is not
the only thing that survives -- this is what that foresight is for. The model
outputs are the expensive part and they are already correct; only the verdicts
were wrong.

Recomputes mean, spread and verdict flips from the stored text, then rewrites
eval.json in place, PRESERVING the serving metrics (goodput, TTFT, ITL) since
those were measured correctly and cannot be recovered from text.

The original scores are kept under "rescored_from" so a corrected row is never
mistaken for an original measurement.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def rescore_dir(d: Path, benchmark: str) -> bool:
    from quality import BENCHMARKS

    ev = d / "eval.json"
    gens = sorted(d.glob("generations-*.jsonl"))
    if not ev.exists() or not gens:
        print(f"  {d.name:20s} no eval.json or generations, skipping")
        return False

    bench = BENCHMARKS[benchmark]
    rows = [json.loads(l) for l in gens[0].read_text().splitlines() if l.strip()]

    # group by repeat, preserving the problem order each was written in
    by_rep: dict[int, list] = defaultdict(list)
    for r in rows:
        by_rep[r["repeat"]].append(r)

    scores, all_verdicts, all_texts = [], [], []
    for rep in sorted(by_rep):
        items = sorted(by_rep[rep], key=lambda r: r["problem"])
        # `answer` carries the task_id for code benchmarks (see _answer_of)
        judged_rows = [{"task_id": r["answer"]} for r in items]
        texts = [r["text"] for r in items]
        v = bench.judge(judged_rows, texts)
        scores.append(sum(v) / len(v))
        all_verdicts.append(v)
        all_texts.append(texts)

    n = len(all_verdicts[0])
    flips = sum(1 for i in range(n) if len({v[i] for v in all_verdicts}) > 1)
    moved = sum(1 for i in range(n) if len({t[i] for t in all_texts}) > 1)

    r = json.loads(ev.read_text())
    key = next(iter(r["results"]))
    res = r["results"][key]
    old = res["mean"]
    res["rescored_from"] = {"mean": old, "scores": res["scores"],
                            "why": "verdicts recomputed from stored generations"}
    res["scores"] = scores
    res["mean"] = statistics.fmean(scores)
    res["spread"] = max(scores) - min(scores)
    res["verdict_flips"] = flips
    res["text_changed"] = moved
    ev.write_text(json.dumps(r, indent=2, default=str))

    print(f"  {d.name:20s} {old:.4f} -> {res['mean']:.4f}   "
          f"spread {res['spread']:.4f}   flips {flips}/{n}")
    return True


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    root = Path(sys.argv[1])
    benchmark = sys.argv[2] if len(sys.argv) > 2 else None

    dirs = sorted(p.parent for p in root.glob("*/eval.json"))
    if not dirs:
        print(f"  no eval.json under {root}")
        return 1

    if benchmark is None:
        # take it from the run's own record rather than guessing
        meta = dirs[0] / "run_meta.json"
        if meta.exists():
            m = json.loads(meta.read_text())
            benchmark = (m.get("args", {}).get("benchmark") or {}).get("value")
    if not benchmark:
        print("  cannot tell which benchmark; pass it as the second argument")
        return 1

    print(f"\n  re-judging {len(dirs)} runs under {root}  (benchmark {benchmark})\n")
    for d in dirs:
        rescore_dir(d, benchmark)
    print(f"\n  done. Serving metrics were preserved -- only verdicts changed.")
    print(f"  python summarize.py {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
