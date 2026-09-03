# inferopt — the API, as a user would meet it

**NOTHING HERE IS IMPLEMENTED.** This is the caller's view, written first so the
shape can be argued with before any of it is built. Where a decision follows
prior art it says whose; where it departs, it says why.

The pitch this API is built around is not "an inference optimizer" — NVIDIA's
[AIConfigurator](https://github.com/ai-dynamo/aiconfigurator) already searches
model + hardware + SLA and even emits a Pareto frontier. It is **"prove the
optimization did not break the model."** AIConfigurator states plainly that it
"focuses exclusively on throughput and latency — no accuracy evaluation occurs,"
and that quantization defaults come from the HF config without verifying
accuracy impact. That is the gap. Quality is therefore the product, not a side
output, and it shows in every signature below.

---

## 1. The 30-second path

```python
from inferopt import optimize, SLO

result = optimize(
    model="Qwen/Qwen3-14B",
    trace="data/trace_shared.jsonl",
    slo=SLO(ttft_p99_ms=500, itl_p99_ms=250),
    benchmarks=["math_500"],            # built-in names are allowed
    run_dir="runs/exp1",
)

print(result.best().config)             # the config to deploy
print(result.best().goodput)            # 54.4
print(result.best().quality_axis)       # 0.998 -- nothing resolved as regressed
result.plot.all("report.html")
```

Built-in benchmark names (`"math_500"`, `"mbpp_plus"`) are accepted as a
shorthand. Anything beyond the built-ins is a `Benchmark` object — see §3.

---

## 2. Workload: `qps` is an input, concurrency is not

Concurrency is **not** a parameter you pass. By Little's Law, `L = λW`:
arrival rate is a property of your traffic, concurrency is an outcome of that
rate and how long each request takes. Asking the caller for `L` invites the
mistake we already made — pinning L=30 on a server that sustained 8, which
measures the concurrency cliff rather than the config.

Three ways to say what your traffic looks like, in descending fidelity:

```python
# 1. captured traffic (best) -- arrival_ts is what makes it a trace
trace = "data/trace_shared.jsonl"
#    {"prompt": "...", "input_tokens": 78, "output_tokens": 318,
#     "arrival_ts": 0.0053, "prefix_id": null, "adapter_id": null}

# 2. synthetic from a spec, when you know the shape but have no capture
from inferopt import Workload
trace = Workload(
    input_tokens=(620, 2660),      # (mean, p99)
    output_tokens=(260, 804),
    qps=16.0,
    burstiness=1.56,
    prefix_overlap=0.31,           # SHARED SYSTEM PROMPTS, see the warning
)

# 3. derived from the benchmark prompts, when you have neither
trace = None                       # requires qps=... on optimize()
```

**The warning that belongs on tier 3.** Prefix caching scored **+307%** on our
trace *because* it has 31% shared-prefix overlap. Earlier that trace carried
`prefix_id` labels with no actually-shared text — measured overlap 0 — and the
same node would have been rejected as useless. Benchmark prompts share ~0%
prefix, so tier 3 will reject prefix caching. That is the correct answer for
eval-shaped traffic and the wrong answer if you serve a chat product with a
shared system prompt. The report says so on the run.

---

## 3. Benchmarks: judge and metric are different layers

The single most confusing question was "judge or metric?" — the answer is
**both, because they do different jobs**, which is how
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md)
(`process_results` per document, then `metric_list` with `aggregation`) and
[LangSmith](https://langsmith-sdk.readthedocs.io/en/latest/evaluation/langsmith.evaluation.evaluator.EvaluationResult.html)
(evaluator, then `EvaluationResults`) both split it.

```
judge   per SAMPLE      raw generation -> a score / verdict for THAT sample
metric  AGGREGATION     the per-sample scores -> one number, with a direction
```

```python
from inferopt import Benchmark, Metric

math500 = Benchmark(
    name="math_500",
    dataset="HuggingFaceH4/MATH-500",       # HF id, or a path, or a list[dict]
    split="test",
    prompt=lambda r: r["problem"] + "\n\nPut your final answer in \\boxed{}.",
    judge=boxed_answer_judge,               # optional; see §4
    metrics=[Metric("exact_match")],        # optional; see §5
    max_tokens=1024,
    chat=False,                             # apply the model's chat template?
)
```

### Which of judge / metrics you must supply

| supplied | behaviour |
|---|---|
| `metrics` only | the metric's built-in judge is used (`exact_match` implies a comparison judge) |
| `judge` only | aggregation defaults to `mean` over the per-sample scores |
| **both** | both are used. No conflict, no precedence — they are different layers |
| neither | **`ValueError` at construction**, before a GPU is touched |

That last row matters. Every failure this project has paid for was discovered
*after* the expensive part: a benchmark that cannot be scored must fail while
you are still typing, not after 378 generations.

### `chat`

Off by default, and this is a measured decision rather than a stylistic one. The
evaluator talks to `/v1/completions`, which applies no chat template. On MBPP+
both styles emitted a fenced code block 60/60 times and accuracy differed by one
problem — but a raw completion has **no stop token**, so every generation runs to
`max_tokens`:

```
raw   pass@1 0.8167   mean 653 output tokens
chat  pass@1 0.8333   mean  37 output tokens
```

Same answers, 18x the tokens, and a serving profile belonging to the prompt
format rather than the model. Turn it on for instruct models; leave it off to
stay comparable with numbers measured raw.

---

## 4. The judge contract

```python
from inferopt import Sample, Verdict

@dataclass(frozen=True)
class Sample:
    row: dict          # the dataset record, untouched
    text: str          # the model's raw output
    index: int         # position in the scored set

@dataclass
class Verdict:
    score: float                  # per sample; a bool is 0.0 / 1.0
    passed: bool | None = None    # if the judge has a categorical opinion
    error: str | None = None      # record, do NOT raise -- see below
    extra: dict | None = None     # anything worth keeping (parsed answer, etc.)

def judge(samples: list[Sample]) -> list[Verdict]: ...
```

Returned verdicts must be **in sample order**. A judge that parallelises must
reorder before returning; ours runs a process pool and had to be fixed for
exactly this, because callers zip verdicts against rows to find which problem
flipped, and unordered verdicts blame the wrong one every time.

### When a judge fails

Modelled on [DeepEval](https://deepeval.com/docs/metrics-custom), which records
`self.error` and reports `is_successful()` false rather than exploding, plus one
rule of our own:

| situation | behaviour |
|---|---|
| one sample raises | `Verdict.error` set, score 0, run continues |
| **every** sample raises | **`JudgeError` raised** |
| judge returns the wrong length | `JudgeError` at once |

The middle row is ours and it is not defensive programming. A scorer that fails
everything reports `0.0000`, which is indistinguishable from total quality
collapse — we lost a seven-config ladder to precisely that (a missing
`tree_sitter_python` made every sample raise, and each raise was recorded as a
failed sample). A benchmark that cannot score must say so, not return a number.

### LLM-as-judge is an implementation, not a concept

```python
from inferopt import LLMJudge

helpfulness = LLMJudge(
    model="claude-sonnet-4-5",               # NOT the model under test
    prompt="Rate 1-5 how well the answer addresses the question.\n\n"
           "Q: {row[question]}\nA: {text}\n\nReturn only a digit.",
    parse=lambda s: float(s.strip()[:1]),
    retries=2,
    on_parse_failure="error",                # "error" | "zero" | "skip"
)
```

It satisfies the same `judge` callable, so nothing else in the API changes. One
constraint the eval frameworks do not have: **the judge model must not run on
the GPU under test.** It would contend with the thing being measured, and every
serving number in the run would be wrong. `optimize()` refuses a local
`LLMJudge` pointed at the same device.

---

## 5. Metrics: direction is explicit, and there can be several

```python
@dataclass(frozen=True)
class Metric:
    name: str
    higher_is_better: bool | None = None   # REQUIRED for custom, defaulted for built-ins
    aggregation: str | Callable = "mean"   # mean|median|min|max|p90, or a callable
    role: str = "axis"                     # axis | report      -- see §6
    fn: Callable | None = None             # custom: (list[Sample], list[Verdict]) -> float
```

`higher_is_better` is not optional for custom metrics, and my first draft of
this API got it wrong by assuming higher-is-better universally — WER,
perplexity and calibration error are all lower-is-better. Both
[lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md)
(`higher_is_better`) and [MLflow](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.metrics.html)
(`greater_is_better` on `make_metric`) make it explicit, and both default it for
built-ins while requiring it for custom.

Built-ins ship with a direction so you never state it twice:

```
exact_match  f1  precision  recall  accuracy  pass@k  auc  rouge_l   higher
wer  cer  perplexity  ece                                            lower
```

**An unknown metric name is an error, not a custom metric.** Otherwise a typo
(`"exact-match"`) silently becomes a custom metric with no direction and no
`fn`, and fails much later with a confusing message.

Several metrics per benchmark, each with its own role:

```python
asr = Benchmark(
    name="librispeech",
    dataset="openslr/librispeech_asr", split="test.clean",
    prompt=lambda r: r["audio_prompt"],
    judge=transcribe_judge,
    metrics=[
        Metric("wer", higher_is_better=False),                     # an axis
        Metric("cer", higher_is_better=False, role="report"),      # recorded only
        Metric("latency_p50_ms", higher_is_better=False, role="report"),
    ],
    max_tokens=256,
)
```

A custom metric receives everything and returns one number:

```python
def macro_f1(samples: list[Sample], verdicts: list[Verdict]) -> float:
    ...

Metric("macro_f1", higher_is_better=True, fn=macro_f1)
```

The signature takes **both** samples and verdicts, because set-level metrics
(f1, precision, recall, auc) cannot be computed from per-sample booleans — they
need predictions and references across the whole set. That is why the
judge/metric split exists rather than a single `judge -> list[bool]`.

---

## 6. Quality is an AXIS, not a gate

**Nothing stops the search by default.** This tool exists to produce a Pareto
frontier that a human chooses from; a gate that halts exploration destroys the
very points the frontier is meant to offer. A config that costs 5% accuracy for
26x goodput is not an error — it is one of the choices, and whether it is
acceptable is not ours to decide.

That is a correction to an earlier draft of this document, which defaulted
metrics to `gate="block"`. Blocking meant the traversal reverted and stopped
accumulating down that branch, so the search never reached the configs beyond
it. The frontier was poorer for it and the user never saw the trade.

```
role="axis"     (default)  measured, recorded, and part of the frontier's
                           quality coordinate. A regression is REPORTED, never
                           blocking.
role="report"   measured and recorded, kept out of the quality coordinate.
                For things worth watching that should not move the frontier --
                a secondary metric, a diagnostic, a latency number.
```

Every metric is always measured, always recorded, always available for plotting.
The only question `role` answers is whether it moves the frontier's quality axis.

### Opting IN to a constraint

Some users do want a floor — "never recommend anything under 0.70". That is
opt-in, applies to the **recommendation** rather than the search, and the
violating configs stay on the frontier, marked:

```python
optimize(..., require={"support_tickets.exact_match": 0.70})

result.best()                  # best config SATISFYING the requirement
result.best(require=None)      # best regardless -- the requirement is a view
result.frontier                # everything, with .satisfies_requirements per point
```

The one thing that can genuinely stop exploration is separate, explicit, and
off by default — for when you know a branch is worthless and do not want to
spend the remaining budget in it:

```python
optimize(..., abandon_below={"support_tickets.exact_match": 0.40})
```

Even then it abandons a BRANCH, not the run, and everything measured up to that
point stays on the frontier.

### Reporting a regression without acting on it

Two numbers decide what a delta *means*, and neither of them stops anything:

```
delta <= tolerance    not resolvable -> not a finding, in either direction
delta >  tolerance    a real change  -> recorded, surfaced, plotted
```

`tolerance` is **measured, not guessed**: the run scores the same config three
times and takes the spread. Measured values on this rig — MBPP+ 0.0053-0.0159,
MATH-500 0.02-0.06 at n=100, 0.006 at n=500. That is why NVFP4's -0.0253 was
invisible at n=100 and a finding at n=500.

```python
@dataclass(frozen=True)
class QualityChange:
    benchmark: str; metric: str
    baseline: float; value: float; delta: float
    tolerance: float
    resolved: bool          # abs(delta) > tolerance
    direction: str          # "regression" | "improvement"

result.quality_changes      # every RESOLVED change, across every trial
result.regressions          # the subset that went the wrong way
```

Only resolved changes appear. A list padded with noise trains people to ignore
it, which is worse than not having one.

### No aggregation across metrics

[HELM](https://crfm.stanford.edu/2025/03/20/helm-capabilities.html) moved from
mean-win-rate to mean-score for leaderboards precisely because aggregate scores
are "sensitive to small variations that invert ranks" — but that is a *ranking*
problem. Our frontier needs ONE quality coordinate per config to plot against
goodput, and averaging hides exactly the case that matters: NVFP4 is a real loss
on MATH-500 at n=500 and unresolvable on MBPP+ at n=378.

So the coordinate is a **worst-case, unit-free regression fraction** — mixed
units (0-1 accuracy against unbounded WER) make a raw `min()` meaningless:

```
regression = (baseline - value) * (+1 if higher_is_better else -1)
fraction   = regression / max(tolerance, abs(baseline))    # unit-free
Trial.quality_axis = 1 - max(fraction over role="axis" metrics)
```

`quality_axis` is the attribute the frontier sorts and plots on. Higher is
better, 1.0 means nothing regressed, and it is comparable across benchmarks
whose raw scores are not.

---

## 7. The full call

```python
from inferopt import optimize, Benchmark, Metric, SLO, Workload

result = optimize(
    model="Qwen/Qwen3-30B-A3B",          # HF id, local path, or a quantized artifact
    trace="data/trace_shared.jsonl",     # path | Workload | None (+ qps=)
    slo=SLO(ttft_p99_ms=500, itl_p99_ms=250),
    benchmarks=[math500, mbpp, asr],
    # NOTHING BLOCKS BY DEFAULT. Both of these are opt-in and neither is set
    # here: `require` filters the RECOMMENDATION and leaves the frontier whole;
    # `abandon_below` is the only thing that stops exploring a branch.
    require=None,
    abandon_below=None,
    hardware="auto",                     # or HardwareSpec(...) to plan for another box
    lossless_only=False,
    budget_minutes=300,
    run_dir="runs/moe-full",
    seed_from="aiconfigurator",          # "aiconfigurator" | "conservative" | dict
)
```

`seed_from="aiconfigurator"` is deliberate and is how this composes rather than
competes. Prediction is cheap and covers thousands of configs; measurement is
expensive and covers about twenty. Let the predictor pick the starting point,
then measure the shortlist and gate it on quality:

```
predict (AIConfigurator)   10k configs   seconds   no hardware, no quality
measure (inferopt DAG)      ~20 configs    hours   real box, quality gated
```

On hardware the predictor has never profiled — its DB covers H100/H200/B200/
GB200/A100, not GB10 — it falls back to the nearest same-architecture part and
applies a roofline correction, and says so.

---


## 8. What comes back

```python
result.best()                # Trial: highest goodput (see require= for a floor)
result.best().config         # dict -> pass straight to `vllm serve`
result.frontier              # list[Trial], non-dominated, INCLUDING reverted ones
result.baseline              # what everything is a ratio against
result.trials                # EVERY measurement, frontier or not
result.quality_changes       # resolved quality movements, across all trials
result.to_df()               # pandas, one row per trial, all metrics as columns
result.save("result.json")
```

`result.frontier` includes **reverted** configs on purpose. "Less goodput, better
latency" is a trade someone may want — in one run three of five frontier points
were reverted configs.

### Cost and capacity

```python
result.replicas(qps=16.0)                          # for the chosen config
result.replicas(qps=16.0, config=result.baseline)  # for comparison: 141 -> 63
result.cost_per_mtok(gpu_hourly=2.99)              # $/M tokens at that replica count
result.capacity_curve                              # the CHOSEN config's own sweep
```

`capacity_curve` is the chosen config's own sweep, and the distinction is not
pedantic: on one MoE run the seed peaked at L=2 with 41.8 goodput and collapsed
to 7.2 by L=8, while the config actually chosen **peaks at L=8 with 65.6**.
Reporting the seed's curve beside the winner understates deployed capacity
roughly ninefold.

### The plots

The frontier is the deliverable, so the plots are part of the API rather than
something a user reassembles from a dataframe:

```python
result.plot.before_after("ba.html")        # baseline vs chosen, every metric
result.plot.quality_vs_goodput("qg.html")  # the frontier itself, the core plot
result.plot.cost_surface("cs.html", gpu_hourly=2.99)
                                           # goodput x quality x $/M tokens
result.plot.replicas("rep.html", qps=16.0) # replica count, before vs after
result.plot.capacity("cap.html")           # goodput vs concurrency, with the peak
result.plot.all("report.html")             # every plot above, one page
```

Every point on every plot carries its config, so hovering a frontier point shows
the flags that produced it — the plot is how a user chooses, not decoration.

## 9. What a user sees while it runs

```
  model     Qwen/Qwen3-30B-A3B  (30.5B gqa, 61GB weights, 3.4B active)
  hardware  NVIDIA GB10 x1  cc12.1  122GB unified  273 GB/s
  workload  800 reqs, in p99 2660, out mean 260, 16.0 qps, prefix 31%
  slo       ttft_p99 500ms  itl_p99 250ms
  quality   math_500, support_tickets  (axis)   tone  (report)
            nothing blocks; every config measured lands on the frontier
  weights   pre-fetching Qwen/Qwen3-30B-A3B (outside the launch timeout) ...
  weights   ready, 61.1 GB in 41.2 min

  stage 1.2 aiconfigurator: no GB10 in its database; using GB200 and
            correcting by roofline (273 GB/s vs 8000 GB/s)
  stage 1.3 seed  29.7 goodput  513ms ttft  L=2

  KEEP   prefix_caching       35.4 goodput  +19.2%  (1 variant)
  revert chunked_prefill      35.6 goodput   +0.3%  (2 variants)
  KEEP   spec_decode_ngram    66.6 goodput +124.2%  (2 variants)
         operating point 2 -> 8 (kept config sustains more)
  revert graph_capture        48.0 goodput  -27.9%  (1 variant)

  lossless_complete  measured quality tolerance: math_500=0.0100
  KEEP   weight_autoquantize  212.5 goodput
         quality: math_500 0.7400 -> 0.7190 (-0.0210, tolerance 0.0100) RESOLVED
  revert weight_autoquantize  312.6 goodput   (lower goodput than incumbent)
         quality: math_500 0.7400 -> 0.6800 (-0.0600, tolerance 0.0100) RESOLVED

  frontier   5 points spanning 0.6800-0.7400 quality and 11.9-312.6 goodput
             2 resolved regressions recorded; none of them stopped the search
```

Both quantized rows are on the frontier. A tool that only optimises throughput
recommends the 312.6 row and never mentions that it cost 6 points of accuracy;
one that BLOCKED on quality would have discarded it and never shown the trade at
all. The point is that both numbers reach the user attached to each other.

---

## 10. Errors a user will actually hit

```python
Benchmark(name="x", dataset="...", prompt=...)
# ValueError: benchmark 'x' has neither `judge` nor `metrics`, so nothing can
# score it. Supply metrics=[Metric("exact_match")] for the built-in judge, or a
# judge= callable.

Metric("my_score", fn=my_fn)
# ValueError: custom metric 'my_score' must state higher_is_better. It cannot be
# inferred, and guessing it inverts the frontier: WER and perplexity are
# lower-is-better while accuracy and f1 are higher.

optimize(..., trace=None)
# ValueError: trace=None derives the workload from benchmark prompts, which
# carry no arrival pattern. Supply qps=... , or a trace with arrival_ts.

optimize(..., benchmarks=[b], judge=LLMJudge(model="local/qwen", device="cuda:0"))
# ValueError: the judge model would share cuda:0 with the model under test.
# Every serving number in the run would include the judge's contention.
```

---


## Open questions

1. **Do quality probes run at every node, or only at checkpoints?** Today every
   lossy node re-measures and lossless nodes inherit. With three benchmarks that
   is 3x the cost at each lossy node.
2. **Should `optimize()` be resumable?** A four-hour run that dies at hour three
   currently keeps its journal but has no `resume=`.
3. **Multi-GPU.** Parallelism nodes are parked. AIConfigurator covers TP/PP and
   disaggregated prefill/decode; we do not, and the API says nothing about it.
4. **Who owns artifacts?** Quantized checkpoints are 10-60 GB. `run_dir` is the
   wrong home for something reused across runs.

## Blockers before any of this can be built

```
fetch_data.py   DATA = Path(__file__).parent / "data"     -> importlib.resources
mbpp_score.py   PKGS = HERE / ".evalplus-pkgs"            -> user cache dir
                results keyed to the runs/ layout
                no pyproject.toml -- not packaged at all
```

`traverse()` already returns a `Result` with `trials`, `incumbent`, `baseline`
and a working `frontier()`, so §8 is mostly a rename. The work is path plumbing,
roughly a day, and the run-directory completeness gap (a traversal saves no
generations, unlike `eval_repro`) is worth closing first — it is why RULER's
0.05 on the MoE is still undiagnosed.

---


## 11. Gaps the example script exposed

`examples/optimize_support_assistant.py` was written as a user would write it,
against the sections above. Four things it needed did not exist, and all four
are now folded in. That is the point of writing the caller first: a design reads
as complete until someone tries to use it.

| gap | resolved in |
|---|---|
| nowhere to surface a non-blocking regression | §6 `result.quality_changes` / `.regressions`, as `QualityChange` |
| the frontier's quality coordinate was never named | §6 `Trial.quality_axis`, with the formula |
| a bare string built-in cannot carry `n=` or a role | §1 / §3 `Benchmark.builtin(name, **overrides)` |
| `replicas()` could not compare against the baseline | §8 `replicas(qps, config=...)` |

Plus one hazard the example surfaced rather than needed: `Metric("exact_match")`
relies on a built-in default for direction, so a typo (`"exact-match"`) would
silently become a custom metric with no direction and no `fn`. §5 now requires
unknown metric names to raise at construction.
