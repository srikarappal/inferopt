# Inferopt Architecture

Given `{model, hardware, workload, SLOs}`, discover the best inference implementation
and serving configuration, then expose the latency / throughput / cost / quality
frontier.

This document records not just what each stage does but **why it is shaped that way**.
Most of the design is scar tissue from a specific failure, and the reasoning is the
expensive part.

| | |
|---|---|
| Target | Qwen3-14B on NVIDIA GB10 (DGX Spark) |
| Backend | vLLM 0.26.0 |
| Written | 27 August 2026 |

---

## Orientation: three levels, three kinds of problem

The central idea, and the one that determines everything below: inference optimization
is not one big hyperparameter search. It is **hierarchical**, and each level wants a
different tool.

```
which idea should I try?      a reasoning problem      -> an LLM, eventually (stage 3)
which implementation?         a systems problem        -> a rule DAG (stage 2)
what should this number be?   a black-box problem      -> enumerated sweeps (stage 2)
is this even possible?        an arithmetic problem    -> a roofline (stage 1.2)
```

Putting an LLM at the bottom of that stack — asking it to choose
`max_num_batched_tokens = 12288` vs `16384` by reflection — is the mistake this
architecture exists to avoid. Putting a rule engine at the top, where the question is
"which family of technique is even worth trying given these diagnostics", is the
mirror mistake.

### The economics that drive every decision

A serving benchmark costs **~8 minutes per configuration**: a launch, a warmup, two
measurement windows. That single number sets the shape of everything. It is why the DAG
has ~19 nodes and not 200, why the traversal is greedy rather than combinatorial, why
prediction happens before measurement, and why a technique that requires producing an
artifact (a quantized checkpoint) is treated differently from one that is a launch flag.

---

## Stage 0 — Fingerprint the problem  `[built]`

Before optimizing anything, characterize what is being optimized. The fingerprint is the
contract every later stage reads, and it prunes more of the search space than any search
algorithm will.

The user supplies at most six things. Everything else — roughly 40 fields — is detected,
because **a field a human types is a field a human can get wrong**, and a wrong
fingerprint optimizes for a machine or a workload that is not there.

```python
InferOptRequest(
    model="Qwen/Qwen3-14B",
    trace="prod_traffic.jsonl",
    ttft_p99_ms=500, itl_p99_ms=250,
    allow_loss=0.03,
)
```

### Detection sources

| source | yields | cost |
|---|---|---|
| `config.json` | layers, heads, head_dim, dtype, context, MoE shape, multimodality | free |
| safetensors index | weight bytes — authoritative, never arithmetic | free |
| `nvidia-smi` + `/proc` | GPU, compute capability, memory, cores, system RAM | free |
| lookup table | memory bandwidth (nvidia-smi does not report it) | free |
| `adapter_config.json` | LoRA rank, target modules, adapter count | free |
| **the trace** | length distributions, QPS, burstiness, prefix overlap, sampling | needs a trace |

**The trace is required, not optional.** Summary statistics can *describe* a workload but
cannot *reproduce* one — the benchmark replays real prompt text, real arrival times, real
prefix sharing. A hand-typed length distribution optimizes for traffic that does not
exist. Every workload statistic is derived from the trace so none of them can be guessed
wrong.

### Four derivations that are easy to get wrong

**Hybrid attention — a 4x error.** Modern architectures interleave full attention with
linear/SSM layers. Only the full-attention layers keep a KV cache that *grows* with
context; linear layers hold fixed-size state. Qwen3.5-9B has `full_attention_interval: 4`
— 8 of 32 layers cache. Assuming every layer caches gives **128 KB/token instead of
32 KB**. At 256k context that is 34.4 GB predicted against 8.6 GB actual: the difference
between "won't run" and "fits comfortably."

**MoE — two different weights.** Every expert is *resident* (memory) but only the routed
ones are *read* (bandwidth). `weight_gb` and `active_weight_gb` are separate fields for
that reason. Conflating them sizes a 235B-A22B model as if it were 22B of memory or 235B
of bandwidth — wrong in opposite directions.

**Nested config.** Multimodal checkpoints put the language model under `text_config`.
Reading the top level finds nothing and raises `KeyError: num_attention_heads`.

**[CORRECTION] Silent fallback produced a believable wrong number.** Parameter count
originally came from arithmetic over config fields when the checkpoint index could not be
read. A `NameError` (an import out of scope) was swallowed by a bare
`except Exception: return None`, and the fallback formula reported **16.45B for a 14.8B
model** — an 11% overestimate that looks entirely reasonable. Now: the index is
authoritative, failures are printed with their reason, and the fallback is GQA-aware
rather than assuming MHA. *A wrong value that looks fine is worse than a crash.*

---

## Stage 1 — Predict cheaply, then measure once  `[built]`

Stage 1 answers "where should we start?" in about five seconds, then spends one launch
confirming it. Its output is an *incumbent*: a config plus a measured goodput that every
later decision is a ratio against.

### 1.1 — Which tool, and a category error to avoid

| system | category | needs a GPU? |
|---|---|---|
| AIConfigurator | predictor — analytical kernel model | no, ~5s |
| Vidur / Frontier | predictor — simulator | no, ~CPU-hour |
| llm-tuna | *search method* | yes, many evals |
| SLO-Guard | *search method* | yes, many evals |

llm-tuna and SLO-Guard are stage-2 search strategies wearing stage-1 clothes — they
*find* a config by running things rather than predicting one. AIConfigurator is chosen
because it uniquely answers the backend question and emits launch files for
vLLM / SGLang / TRT-LLM.

### 1.2 — Predict on a proxy, correct with physics

AIConfigurator has a calibrated kernel database for `h100_sxm, h200_sxm, b200_sxm,
gb200, a100_sxm` and estimate-only support for a few more. **GB10 is not among them.**
So an unsupported part is mapped to the nearest member of its architecture family, under
two rules that keep it honest:

```
RANK on the proxy       config rankings survive a monotone rescaling -- a config that
                        batches better or preempts less wins on both parts, for the
                        same reason. The proxy picks the SHAPE.

SCALE with a roofline   absolute numbers do not survive. GB10 is ~273 GB/s against
                        B200's ~8000 -- a 29x gap -- so the proxy's tokens/s is
                        meaningless as a forecast. The floor comes from physics:
                        a decode step must read every active weight, so
                        ITL >= weight_bytes / bandwidth, whatever any database says.
```

**The second rule is the valuable half.** It answers *"is this SLO reachable on this
hardware at all?"* in milliseconds. For Qwen3-14B on GB10: 29.5 GB of weights at
273 GB/s means **ITL >= 108 ms**. A 30 ms SLO is unreachable by arithmetic — no batching,
scheduler or kernel choice moves it. The tool reports the remedies (FP8 -> 54 ms,
INT4 -> 27 ms, a smaller model, or a relaxed SLO) rather than letting a search discover
it over three hours.

**The roofline was validated against measurement, not asserted.** On the smaller
Qwen3.5-9B (18 GB, floor 66 ms), measured ITL across every workload slice ran
74-197 ms — best case within 12% of the floor, never below it. Its error is
one-directional: reality sits *above* the roofline, never under. That is what makes it
safe to act on.

### 1.3 — Measure the seed, and gate on it

Launch the predicted config, warm up 15 s, run two 45-second windows replaying the trace.
The result becomes the incumbent. If it cannot meet the SLO, the run **stops here** —
everything downstream is a ratio against this measurement, so a starting point at
goodput ~ 0 makes every later comparison a comparison of noise.

Running it twice also yields the **accept band** for free: the across-launch spread,
doubled. That number is never hardcoded.

### 1.4 — Show both, label the first as predicted

Prediction takes 5 s and measurement ~10 min; making a user wait for the second when the
first exists is bad product. Two rules: never present a prediction as a result
(`predicted 7,430 tok/s (+/-20%)`, then `measured 5,912`), and **store the delta**.
Accumulated per hardware and model, those deltas become the calibration dataset a future
predictor needs — a UX decision doubling as a data-collection strategy.

---

## Stage 2 — Traverse a deterministic technique DAG  `[built]`

A directed acyclic graph of techniques, walked by a state machine: apply a node, launch,
measure, keep or revert, follow the corresponding edge. No LLM anywhere in this stage.

### Why greedy, and why that is not a compromise

Each node is "apply, measure, keep or revert" against an accumulating incumbent.
**N nodes cost N launches, not 2^N.** At 8 minutes a launch, that is the difference
between a 2.5-hour run and one that never finishes.

Greedy misses interactions, which is real. The mitigation is not to abandon it but to
name the interactions that matter and give them their own nodes — see *interaction
nodes* below.

### Fingerprint pruning does more work than the search

```
single GPU              -> prune every parallelism node (TP/PP/EP/DP)
dense model             -> prune MoE / expert-placement nodes
prefix overlap < 5%     -> prune the prefix-caching subtree
mean output <= 64 tok   -> prune speculative decoding (it only helps decode)
p99 input < 4096        -> prune long-context KV strategies
n_adapters <= 1         -> prune the entire multi-LoRA subtree
```

On a single-GPU dense model roughly half the tree is deleted before any measurement.
Predicates are evaluated by a restricted AST evaluator — comparisons, arithmetic,
attribute paths and a short function allowlist — and every path is **checked against the
pydantic schema at validation time**, so a typo fails at parse rather than evaluating
falsy and silently disabling a node for the rest of the project.

### The spine

```
LOSSLESS                                                     launches
  max_model_len_rightsize   free KV blocks by right-sizing ctx        1
  chunked_prefill           if p99 input > 1024                       2
  prefix_caching            if per-adapter overlap > 5%               1
  kv_block_size             {16,32}, if long context                  1
  spec_decode_ngram         if mean output > 64                       2
  spec_decode_depth         if a proposer was kept                    2
  graph_capture             torch.compile -- deliberately LAST        1
                                                                    ---
CHECKPOINT  lossless_complete    emit frontier + measure tolerance    1

LOSSY  (only if a quality budget exists)
  kv_cache_fp8              best gain-to-damage ratio, so first       1
  retune_batching_after_kv  <- interaction node                       2
  lora_unmerge_for_weight_quant  representation switch                1
  weight_fp8 -> weight_int4_awq -> (weight_nvfp4: TODO)               2
  retune_batching_after_weight  <- interaction node                   2
                                                                    ---
TERMINAL  frontier             true non-dominated set                 0
```

### Three structural decisions

**Interaction nodes.** Quantization frees memory, which moves the batching optimum — so
the `max_num_seqs` chosen earlier is now wrong. The fix is not a back-edge (that breaks
the state machine) but a distinct *forward* node that re-searches the same parameter
under new conditions. Acyclic, still a tree, captures the one interaction that genuinely
matters. Two or three of these are enough; more and you are back to exponential.

**revert != discard.** A reverted config still enters the trial database and can still
land on the Pareto frontier. In a dry run, **three of five frontier points were reverted
configs** — including an INT4 variant at 2.9x throughput and a quarter the memory, which
failed the stated quality budget but is unambiguously an operating point someone would
choose. A lineage-only frontier throws those away.

**graph_capture is last.** `torch.compile` costs ~30-40 min on this hardware and is keyed
on shapes, so a cache warmed at one `max_num_seqs` is invalidated by the next sweep
value. Paying it per node would cost more than the entire search. Every earlier node runs
eager; compilation is tested once, against the config they accumulated — which is also
when the compile is worth most, on final shapes rather than shapes about to change.

### Probe discipline

| probe | measures | runs on | cost |
|---|---|---|---|
| goodput | throughput counting only SLO-satisfying requests | every node | 45s x 2 |
| equivalence | first-K tokens vs the incumbent | **lossless only** | seconds |
| quality | MATH-500 exact_match, RULER multi-needle accuracy | **lossy only** | 3-6 min |

**Lossless nodes never run a quality benchmark** — not to save time, but because
equivalence is a *stronger* test. If the tokens are identical, quality is identical by
construction; a 100-sample benchmark could only give a noisier version of the same
answer. That single decision is most of what makes the 2-3 hour budget work.

**Goodput, not throughput.** Raw tokens/sec rewards a config that serves everything
slowly. Goodput counts only requests that met the SLO, so blowing the latency target
scores zero. Three configs with identical 80 tok/s raw throughput score **80, 40 and 0**
depending on SLO attainment.

### Thresholds: which are frozen, which evolve

| threshold | lifecycle | why |
|---|---|---|
| SLO, allow_loss | static | user inputs |
| accept_band | **frozen per run**, evolves across runs | a band that moved mid-traversal would leave early and late keep/revert decisions resting on different criteria, with the frontier built from both |
| quality_tolerance | **measured mid-run** | default until `lossless_complete`, measured after |
| incumbent | evolves | that is the point |

**Quality tolerance is measured for free.** Lossless nodes cannot move quality by
definition — so the difference between quality at `lossless_complete` and quality at
stage 1.3 *is* the noise, and it spans the whole lossless traversal: every launch, config
change and thermal state in between, which is exactly the variation the lossy gate must
survive. A dedicated repeat would measure less.

The lossy gate then needs *both* numbers: a delta under the measured tolerance is not
real (rejecting on it rejects noise), a delta over the user's budget is unacceptable
however real. Between them the loss is genuine and affordable — which is precisely a
frontier point.

### The LoRA subtree

Most of the spine survives LoRA unchanged. Three things do not:

- **Prefix caching fragments per adapter.** The cache keys on `(prefix, lora_id)`, so N
  adapters split the hit rate N ways — the predicate uses per-adapter overlap, never the
  global number.
- **Draft-model speculation breaks.** The draft must approximate the *adapted* target;
  acceptance collapses, and you would need one draft per adapter. n-gram is unaffected
  because it never models the network.
- **Weight quantization compounds.** The adapter was trained against unquantized weights,
  so on a quantized base it corrects for weights that no longer exist. Published
  base-model benchmarks understate the damage.

**[CORRECTION] Single-adapter merge moved out of stage 2.** A merged model is
architecturally *identical* to the base — same GEMM shapes, no SGMV kernels — so the
predictor becomes valid again and no LoRA node needs to fire. That makes merge a
**stage-0 precondition**, not a stage-2 node: it is strictly smaller and strictly faster,
so there is nothing to measure.

But it is *quality-equivalent (verify)*, not strictly equivalent. `(W0+BA)x == W0x + BAx`
holds algebraically; the merged weights are rounded back to storage dtype while the
unmerged path accumulates separately, so outputs are not bit-identical. No paper
establishes identical held-out performance for single-adapter merge — the merge
literature studies merging *multiple* adapters, a different and demonstrably lossy
operation. So stage 0 *verifies* the merge rather than assuming it.

A second consequence, structural rather than about quality: v1 quantizes only from
published checkpoints, and those are of the *original base*. Once an adapter is merged
the served weights match no published checkpoint, so the entire weight-quantization
branch becomes unreachable without calibration. Hence `lora_unmerge_for_weight_quant`.

### The frontier is computed independently

GEPA's built-in `pareto` selector tracks candidates that are best on individual instances
— *not* the classical multi-objective non-dominated set. So the trial database is our
own, and the frontier is computed over **every measurement taken**, across goodput /
quality / TTFT p99 / memory, with per-benchmark quality taken as the *worst* benchmark
rather than the mean: a config that preserves its average by collapsing on one task is
not an acceptable operating point.

---

## Stage 3 — Introduce GEPA, and measure whether it earns its place  `[not built]`

Stage 3 replaces the static `applicable_when` predicates with a reasoner that reads
diagnostics and decides which experiment to run next. GEPA is the *experiment selector*,
never the numeric optimizer.

```
Bad GEPA mutation     max_num_batched_tokens: 8192 -> 9216

Good GEPA mutation    Observation: TTFT fine, ITL degrades above concurrency 64.
                      GPU compute utilization only 58%. KV pressure 94%,
                      preemptions spiking.
                      Hypothesis: decode is limited by KV pressure, not compute.
                      Experiment: raise KV allocation, lower scheduler token
                      budget, hold concurrency, don't touch speculation.

Then the sweep        kv_fraction=0.94  max_tokens=6144
```

### The question this stage answers

Not *"can an LLM optimize inference?"* but: **can a reasoning-guided optimizer reach a
superior configuration using significantly fewer hardware evaluations than black-box
search?**

The reflector is not valuable because it sounds intelligent. It is valuable if it
**reduces experimental sample complexity** — because GPU experiments are the scarce
resource. That is falsifiable, and the falsification is a bake-off at matched wall-clock,
GPU-experiment and launch budgets:

| optimizer | role under test |
|---|---|
| Random | the bar any method must clear |
| Grid | exhaustive floor |
| Bayesian / TPE | black-box, no domain knowledge |
| Rule DAG (stage 2) | encoded expertise, no LLM |
| GEPA | reasoning, no numeric optimizer |
| GEPA + sweeps | the proposed architecture |

Scored on best goodput found, Pareto hypervolume, evaluations to reach a target, and
total cost. **The gate is explicit: if GEPA + sweeps does not beat the rule DAG at
matched budget, ship stage 2 without it and say so.**

### The loop, concretely

The selector is an agent with a turn budget, not a genetic algorithm. Each turn it
hypothesises one experiment, runs it, reads the result, and decides whether to continue.

```
INPUT   criteria      model, dataset, SLO, host count
        priors        a curated list of techniques it MAY explore, and their
                      preconditions -- not a free hand over every vLLM flag
        history       every trial from stages 1-2: config, goodput, TTFT, ITL,
                      SLO attainment, kv_cache_util, preemptions, prefix hit
                      rate, the concurrency curve -- AND every failed launch
                      with its error
        incumbent     the best config stage 2 shipped, and what it measured

LOOP    <= 16 turns
        hypothesise -> validate -> launch -> observe -> keep or discard
        exits early when two consecutive turns fail to clear the noise band

OUTPUT  a config, and the frontier it sits on, scored against stage 2's
```

Sixteen turns is roughly 3.5 hours on a 14B, which is the same order as PB's twenty
launches. The budget is the point: a reasoner that needs a hundred evaluations has not
reduced sample complexity, it has renamed random search.

### Four things the loop needs that stage 2 does not

**A noise band it must clear before believing a result.** This is the one that decides
whether stage 3 works at all. Two identical launches of the same config measured 633.9
and 375.1 tok/s -- a 1.69x spread -- and three identical MoE traversals disagreed about
which factors to keep. An agent that hypothesises from one measurement per turn will
spend sixteen turns building a confident story out of noise. Decisive turns need
repeats, and the prompt must state the band rather than leaving the model to infer it
from numbers that do not carry it.

**Failed launches, with their errors.** Half of what this project learned came from
launches that died: that `moe_backend=triton` is refused for an NvFP4 MoE, that
`max_num_batched_tokens` below `max_model_len` is illegal without chunked prefill, that
an infeasible bit budget has a floor set by the layers modelopt never quantizes. A
selector shown only successes will re-propose all of them.

**A combination validator, not just a flag check.** The existing check catches a flag
vLLM does not accept. It does not catch a flag it accepts in an illegal combination,
which is the failure an agent will actually generate. Catching those before launch is
the difference between sixteen useful turns and six.

**Provenance for a non-deterministic optimizer.** Two stage-3 runs will differ. The
stamp needs the reasoner's model, the prompt, and the seed, or the runs cannot be
compared to each other -- let alone to stage 2, which is the whole point of the gate.

### GEPA is not the first thing to try here

Take the idea, not the algorithm. A home-grown harness comes first, and GEPA stays a
later question rather than the starting point.

There IS a population, and it is worth being accurate about this. Stage 2 leaves ten to
twenty-seven measured configurations -- the reverted ones as much as the kept ones, since
a technique that lost is evidence about the space -- and each stage-3 turn adds another.
`Result.frontier()` already computes the non-dominated set over all of them.

The constraint is not the population's existence but the rate it can grow. GEPA's power
is reflective mutation over a frontier sampled across many CHEAP rollouts; here an
evaluation is a model load, a concurrency sweep and a benchmark -- thirteen minutes and a
GPU. Sixteen turns adds sixteen members. Selection pressure over a population that grows
by one per quarter-hour is a different algorithm from selection pressure over thousands
of rollouts, and the parts of GEPA that assume the latter would be carried without doing
any work.

What transfers is the idea worth having: reflect on execution traces in natural language
and propose a targeted experiment, rather than perturbing a number. That is a prompt and
a loop. Built here it drops into the existing `Strategy` protocol as a fourth
implementation -- scored by the same `compare.py`, against the same baselines, on the
same axes as the walk, the screen and yolo -- and every part of it is ours to inspect
when it produces a result we do not believe, which on current evidence it will.

So: borrow the reflection, skip the genetics, and keep the gate. If the loop does not
beat the stage-2 DAG at matched budget, stage 2 ships alone and this section records why.

### Why there is no tabular benchmark

The obvious way to compare optimizers cheaply is the NAS-Bench pattern: enumerate the
configuration space once, measure every cell, then replay any optimizer against the table
in milliseconds. `replay.py` implements the replay half already -- domination, regret
against a known optimum, regret-vs-budget curves -- and runs today only against synthetic
spaces, which is why none of its numbers are evidence about this system.

The measured table is not coming, for a reason worth stating rather than leaving as an
unfinished task.

The real space is not enumerable. It is not six binary factors; it is those factors
crossed with their values, the operating point, the quantization rungs, and any
parallelism -- and the effect of each depends on what is already applied. That last part
makes it **the compiler phase-ordering problem**: the gain from a technique is a function
of the techniques already in place, so the space does not decompose. This project has
measured that directly. Three identical traversals of Qwen3-30B-A3B kept different
factors, and `chunked_prefill` was reverted at L=2 on runs that never learned what it
does at L=8, where speculative decoding had moved the operating point.

A table over six binary factors IS tractable -- sixty-four cells -- but it is tractable
precisely because it is a toy subspace, and there is no reason performance on it predicts
performance on the real one. The evidence already points the other way: screening beat
the walk by 1.13x on Qwen3-1.7B and the three methods landed within 4% of each other on
Qwen3-14B. A benchmark that gives exact regret on a space nobody deploys buys
false confidence, not insight.

So optimizers are compared the way everything else here is measured: on real hardware,
against real baselines, with the cost in launches reported beside the result.

### Other stage-3 candidates

- Custom kernel generation for an operator the profiler flags as dominant
- Admission control and queueing policy as a written program rather than a flag
- Deciding which candidates get promoted from simulation to real hardware
- Distillation recipe search — where decisions become genuinely semantic ("the 8B student
  is losing reasoning on long-context coding; try a 14B student at FP8 with stronger KV
  optimization")

---

## Appendix A — Facts that were expensive to learn

Measured on this hardware, this model, this stack. Re-derive them before trusting them
elsewhere — but do not re-derive them here.

| fact | value | consequence |
|---|---|---|
| Across-launch throughput spread | 1.91% worst | accept_band = 3.8%; a guessed 2% would reject real wins |
| Within vs across launch | 5x on one slice | within-launch spread alone understates the noise the gate faces |
| Per-token flip rate, greedy | 0.44% | whole-output equivalence has a 19% false-positive floor; first-11-tokens gives 3.5% |
| GSM8K across identical runs | +/-1.4pp (7 of 500) | a 0.5pp tolerance was 3x tighter than the noise — unpassable |
| NIAH across identical runs | +/-1pp (1 of 100) | tolerance equal to measurement granularity is a coin flip |
| ITL roofline, Qwen3.5-9B | 66 ms floor | measured 74-197 ms — never below, best case within 12% |
| Weight load from overlayfs | ~25 s per shard | ~3.5 min of every launch; a bind-mounted HF cache would cut it |
| torch.compile, 14B | ~30-40 min | keyed on shapes, so it does not amortize across a sweep |

### The equivalence taxonomy

"Lossless" is too strong a word, and conflating these classes was the single most
expensive error in this project.

| class | means | example |
|---|---|---|
| strictly equivalent | bit-identical output | rare in practice |
| algorithmically equivalent | distribution preserved, FP details differ | tensor parallelism, kernel swaps, LoRA merge |
| quality-equivalent | statistically indistinguishable | what stage 1 actually means |
| lossy | deliberate, measurable degradation | quantization, distillation |

**The failure this taxonomy prevents.** An earlier gate demanded token-identical 48-token
outputs (a *strict*-equivalence test) in an *algorithmically*-equivalent world, giving a
22.6% false-positive floor. Simultaneously the NIAH tolerance was set to 1%, exactly the
granularity of a 100-item set. Result: **the seed configuration failed its own accuracy
gate** and nothing could ever pass. Three separate defects, one root cause — a threshold
guessed rather than measured.

### Environment traps, all encountered

- **Relocation, three times.** A conda env bind-mounted at a different path breaks
  anything with a baked absolute path: console-script shebangs (`vllm`), ncurses terminfo
  lookup, and venv `pyvenv.cfg` + `bin/python` symlinks. Invoke `python -m module` rather
  than console scripts; use `pip install --target` + PYTHONPATH rather than a venv.
- **PATH inheritance.** vLLM JIT-builds CUDA extensions and shells out to `ninja`, which
  lives beside the interpreter. A subprocess inherits only PATH, so an absolute-path
  invocation dies deep in engine init with `FileNotFoundError: ninja`.
- **Version drift.** vLLM 0.26 dropped `--disable-log-requests`, `--swap-space`,
  `--cuda-graph-sizes`, `--tokenizer-pool-size`, and renamed every Prometheus metric.
  Read `--help=all` and live `/metrics`; never trust a remembered flag name.
- **Dependency conflicts are structural.** AIConfigurator pins `numpy~=1.26.4` against
  vLLM's `<2.4`. That can never share an environment — it runs as a subprocess with its
  own `--target` directory. Check a package's pins *before* installing.
- **Unified memory.** On GB10 `memory.total` reports `[N/A]` and
  `gpu_memory_utilization` is a fraction of *system* memory that the CPU also competes
  for. 0.90 left 1.6 GB of headroom on a 122 GB box.
- **Production shares the box.** The DGX runs live traffic. Never modify the `dextract`
  conda env, never touch `dextract-ocr.service` or OpenSearch. All experimental work goes
  in the `verticalinference` env.

---

## Appendix B — What exists, and what is still missing

| component | file | state |
|---|---|---|
| Fingerprint schema | `fingerprint.py` | built |
| User input + detection | `request.py` | built |
| Predictor + proxy map + roofline | `predictor.py` | built |
| Technique DAG | `dag/llm.json` | built |
| DAG validator | `validate_dag.py` | built |
| Predicate evaluator | `predicates.py` | built |
| Traversal engine | `traverse.py` | built |
| Real evaluator (vLLM) | `evaluator.py` | built |
| Calibration store | `calibration.py` | built |
| Quality benchmarks | `quality.py` | **partial** |
| Dry-run harness | `dryrun.py` | built |
| VLM / diffusion DAGs | `dag/vlm.json`, `dag/diffusion.json` | not built |
| Quantization conversion pipeline | — | not built |
| GEPA experiment selector | — | not built |
| GB10 kernel calibration | — | not built |

### Known gaps, in priority order

1. **HumanEval+ generates but refuses to score.** pass@1 requires executing model-written
   code; wiring that to a bare `exec()` would be the most dangerous line in the project.
   It needs a container/nsjail runner or the evalplus harness. MATH-500 and RULER cover
   reasoning and long-context recall meanwhile.
2. **Quantization uses published checkpoints only.** AWQ/GPTQ calibration is 20+ minutes
   per variant and does not fit the budget, so a customer's own model cannot be quantized
   in-pipeline. This is also what forces `lora_unmerge_for_weight_quant` to exist.
3. **NVFP4 is stubbed.** Wired into the spine as `status: "todo"` so enabling it is a
   one-word change; blocked on the same conversion pipeline.
4. **Per-modality DAGs.** The LLM spine assumes prefill + autoregressive decode. VLM
   reuses ~80% (add a vision subtree); diffusion reuses ~20% — no KV cache, no decode
   phase, no TTFT/ITL, and its step count is itself a lossy knob.
5. **GB10 has no calibrated kernel database.** The proxy + roofline is a workaround.
   Building the real thing is ~35-45 GPU-hours, and the go/no-go is whether a 140 W part
   can *sustain* the locked clocks the collector requires — it hit 84 C at 2.33 GHz under
   benchmark load.

---

*Written 27 August 2026, from a working session on GB10 / vLLM 0.26 / Qwen3-14B.
Every number in Appendix A was measured on that hardware; every correction marked
`[CORRECTION]` was a decision made, found wrong, and remade.*

*The design principle underneath all of it: **if a step claims to preserve behaviour,
that claim is a measurement, not an assumption.***
