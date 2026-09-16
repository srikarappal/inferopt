# Leaderboard benchmarks

The six Open LLM Leaderboard benchmarks, scored against a live serving
configuration through lm-evaluation-harness.

They exist alongside this project's own benchmarks, not instead of them, and the
difference is the point. `math_500`, `mbpp_plus` and `humaneval_plus` use our
prompts, our budgets and our graders. That makes them internally consistent,
which is all a walk needs in order to rank configurations, and externally
unquotable, because nobody can reproduce a number whose harness they do not
have. The six here are the leaderboard's own, run through the implementation the
published numbers come from, so a score means the same thing as anyone else's.

---

## The six

| task | type | shots | gen budget | subtasks | docs | metric |
|---|---|---|---|---|---|---|
| `leaderboard_bbh` | multiple_choice | 3 | n/a | 24 | 6511 | `acc_norm` |
| `leaderboard_gpqa` | multiple_choice | 0 | n/a | 3 | 1192 | `acc_norm` |
| `leaderboard_mmlu_pro` | multiple_choice | 5 | n/a | 1 | 12032 | `acc` |
| `leaderboard_musr` | multiple_choice | 0 | n/a | 3 | 756 | `acc_norm` |
| `leaderboard_math_hard` | generate_until | 4 | **1024** | 7 | 1324 | `exact_match` |
| `leaderboard_ifeval` | generate_until | 0 | **1280** | 1 | 541 | `prompt_level_strict_acc` |

Values read from lm_eval 0.4.13's task YAMLs, not from the README, and asserted
in `tests/test_leaderboard.py` so a drift upstream shows up as a test failure
rather than as a number quietly off parity.

**Each benchmark dictates its own output budget, and they are not close.** MATH
stops at 1024 generated tokens, IFEval at 1280, and the four multiple-choice
benchmarks generate nothing at all: they are scored by comparing the
log-likelihood of fixed continuations, so there is no output length to cap. A
single global `max_tokens` is meaningless across that spread, which is why the
limit lives on the task and why passing `gen_toks` for a multiple-choice task is
ignored rather than honoured.

**Only two of the six exercise the decode path.** `leaderboard_math_hard` and
`leaderboard_ifeval` generate; the other four score logprobs over fixed choices.
If the question is whether a quantization step damaged the model, those two can
see it and the other four largely cannot. `leaderboard.GENERATIVE` names them.

---

## Install

lm-evaluation-harness goes in **its own environment**, on purpose. It pulls
transformers, datasets, sympy, nltk and antlr, none of which belongs beside a
pinned vLLM. It never needs to share: it talks to the running server over HTTP
like any other client and never loads the model itself.

`setup.sh` does this by default into `./lmeval-env` (skip with
`INFEROPT_SKIP_LEADERBOARD=1`). By hand:

```bash
python -m venv lmeval-env
./lmeval-env/bin/pip install 'lm_eval[api,math,ifeval]' transformers \
    math-verify 'sympy>=1.12' 'antlr4-python3-runtime==4.11'
export INFEROPT_LMEVAL_PYTHON=$PWD/lmeval-env/bin/python
```

`INFEROPT_LMEVAL_PYTHON` is honoured or refused, never quietly replaced. If it
names an interpreter without `lm_eval`, that is an error rather than a silent
fallback, for the same reason `rerun_all.sh` will not substitute a different
vLLM than the one you named.

### GPQA needs one manual step

`Idavidrein/gpqa` is gated on HuggingFace, and **a valid `HF_TOKEN` is not
sufficient**: the account behind it must have accepted the dataset terms. Visit
<https://huggingface.co/datasets/Idavidrein/gpqa> signed in as that account,
accept, and rerun. The other five need nothing.

This is checked in preflight, before a GPU is held, because discovering it after
a two-hour sweep has already paid for the server is the expensive order to find
out in.

---

## Use

From the CLI, on any run:

```bash
python -m inferopt.run optimize --model Qwen/Qwen3-1.7B --trace data/trace_shared.jsonl \
    --benchmarks leaderboard_math_hard,leaderboard_ifeval
```

Names are validated immediately, including the gating and harness checks, so a
typo costs a second rather than an hour.

From Python, against a server you already have listening:

```python
from inferopt import leaderboard

leaderboard.run(["leaderboard_math_hard"], "http://127.0.0.1:8100",
                "Qwen/Qwen3-1.7B", limit=50)
```

```python
{"leaderboard_math_hard": {
    "score": 0.41, "metric": "exact_match", "parity": False,
    "max_gen_toks": 1024, "num_fewshot": 4, "limit": 50, "n": 50,
    "raw": {"exact_match,none": 0.41, "exact_match_original,none": 0.41}}}
```

Useful arguments: `limit` caps documents per task, `gen_toks={task: n}` overrides
a generation budget, `num_fewshot={task: n}` overrides a shot count,
`max_length` bounds prompt plus generation and must fit the served
`max_model_len`, `out_dir` keeps lm_eval's full result JSON.

`leaderboard.describe()` prints the table above. `leaderboard.preflight()` says
why anything cannot run.

---

## `parity` is tracked, not assumed

Raising a budget is sometimes right. A reasoning model whose traces run thousands
of tokens is being cut off at 1024, and the leaderboard default was set before
such models were common. But the result is then no longer the leaderboard's
measurement, and a tuned number that gets quoted as a leaderboard score is worse
than no number.

So every result carries `parity`, false as soon as **any** of these is true:

- a generation budget was overridden
- a shot count was overridden
- `limit` was set, because scoring a subset is a different measurement

The flag is reported in the log line and in the returned dict. Use the tuned
number freely; just do not publish it as a leaderboard score.

---

## What this changed about our own `math_500`

Worth recording, because it corrects an earlier conclusion in this project.

Measured on Qwen3-1.7B, same server, same 500 problems, our `math_500` scored
**0.5440** while lighteval, driven by AIPerf, scored **0.9267**. The gap looked
like our 1024-token cap truncating a reasoning model whose traces run p50 3749
tokens, and it does: 99.8% of generations exceed 1024.

But **the leaderboard's own MATH task also caps at 1024**. Our budget is not off
parity; it matches. The 0.9267 came from a 32768-token budget, which is a
reasonable setting for a reasoning model and is not the leaderboard's.

So there are two defensible measurements and they answer different questions:

- at 1024, what the leaderboard measures, comparable to published numbers
- at 8192 or more, what the model can do when allowed to finish reasoning

Both are available here, and `parity` is what keeps them apart.

---

## Cost

A full pass is expensive and mostly in the four logprob benchmarks, which are
large: MMLU-Pro alone is 12032 documents. `limit` exists for that. The two
generative ones are the small ones by document count and the slow ones per
document, because they decode.

Run a few rather than all six unless you need the composite. `--benchmarks`
takes any subset.

---

## An unscored benchmark is `None`, never `0.0`

A gated dataset, a missing harness and a model that answers nothing are
different facts. A benchmark that could not be scored records `None`, and the
quality gate in `traverse.py` skips it rather than treating it as zero. Without
that, a missing HuggingFace licence would reject a configuration on quality
grounds.
