# Decision log

Chronological record of what was decided, what it cost to learn, and what was
rejected. Kept because the reasoning is the expensive part — the code is
recoverable from it, and the reverse is not true.

Hardware throughout: NVIDIA GB10 (DGX Spark), 122 GB unified LPDDR5X, 273 GB/s.
Model: Qwen3-14B, 29.5 GB of weights. Backend: vLLM 0.26.0.

---

## The objective

> "Assuming I want to build an AI model inference optimizer where for a
> particular workload, it will auto-research various minutiae of configurations
> and techniques and propose the best inference serving configuration for that
> hardware."

And a correction that shaped everything after it:

> "This is not just about optimizing vLLM. You need to treat vLLM as one possible
> serving backend, not as the optimization target. The target is the whole
> inference stack: model representation, kernels/compiler, KV-cache/memory
> policy, parallelism, batching/scheduling, speculative decoding, serving
> runtime, and eventually lossy model transformations."

Three stages, in the user's own framing:

1. Fast deterministic estimation — pick a predictor, predict a config, validate
   with a small real run.
2. A deterministic DAG of techniques, lossless before lossy, walked by a state
   machine from stage 1's config, producing a Pareto frontier.
3. GEPA for focused per-node search, plus intelligence about which nodes to skip.

---

## Decisions

### GEPA is not the primary search tool

**Rejected** for stages 1 and 2. Inference optimization is hierarchical and each
level wants a different tool:

```
which idea should I try?      reasoning problem    -> an LLM (stage 3)
which implementation?         systems problem      -> a rule DAG (stage 2)
what should this number be?   black-box problem    -> enumerated sweeps (stage 2)
is this even possible?        arithmetic problem   -> a roofline (stage 1)
```

Asking an LLM to choose `max_num_batched_tokens = 12288` vs `16384` by
reflection is the mistake this architecture exists to avoid. GEPA's place is
stage 3, as an *experiment selector* reading diagnostics and deciding which
family of technique to try next — and it has to earn that in a bake-off against
the rule DAG at matched budget.

### Predict on a proxy, correct with physics

AIConfigurator has no calibrated kernel database for GB10. A config's *ranking*
survives a monotone rescaling, so the nearest family member (b200_sxm) picks the
shape; absolute numbers do not survive a 29x bandwidth gap, so the roofline
supplies the floor:

```
ITL >= active_weight_bytes / memory_bandwidth
    =  29.5 GB / 273 GB/s
    =  108 ms
```

Validated against measurement on Qwen3.5-9B: 66 ms floor, 74–197 ms measured,
never below, best case within 12%. Its error is one-directional, which is what
makes it safe to act on.

### Quantization is produced locally, never downloaded

**Reversed after being built the wrong way.** The weight-quantization nodes
originally pointed at published Hub checkpoints (`Qwen/Qwen3-14B-AWQ`).

> "I told you NOT to pull quantized checkpoints from the Hub. I want to simulate
> a real world scenario where a user brings their model and we run through the
> inferopt pipeline. Getting the said checkpoint from Hub defeats the purpose."

Correct, and the reasoning generalises: a customer's fine-tune has no published
quantization, and benchmarking someone else's `-AWQ` repo measures *their*
calibration job. `quantize.py` now produces variants locally via llmcompressor,
calibrated on the user's own trace rather than pileval — which is both more
faithful and free, since the trace is already loaded.

FP8 needs no producer at all: vLLM quantizes bf16 weights during model load.

### Quantization is layer-selective, from the fingerprint

The first recipe quantized every Linear identically, protecting only `lm_head`.
AWQ is selective *within* a layer — it scales outlier channels found from
calibration activations — but nothing was selective *across* layers.

Protected by default, both structural rather than empirical:

- **`lm_head`** projects to vocabulary; error perturbs the output distribution
  directly with no later layer to absorb it.
- **MoE router** (`mlp.gate`, anchored so it does not catch `gate_proj`) selects
  experts by top-k. Error near a decision boundary does not degrade the output
  slightly — it routes the token to a *different expert*, a discrete jump. A
  dense layer's error averages out; a router's does not.

**Deliberately not protected:** the first and last decoder blocks. Widely said to
be more sensitive, plausibly true here, but unmeasured — and this pipeline should
measure it as its own node rather than pay for the hedge on every model.

### prefix_caching runs first

**Reordered** after run three. On a workload with real prefix sharing it is what
makes the SLO reachable at all: TTFT 2368 → 630 ms, attainment 15% → 97%.

Until attainment is high, goodput is quantised by whole requests — at 15%
attainment one request flipping moves goodput ~11% against a 5% accept band — so
every node judged before it was decided on noise. The general principle: **the
SLO-enabling technique must come before the nodes measured against the SLO.**

### Concurrency is swept, not set

**The largest correction.** See `measurement-protocol.md` for the full reasoning.

The user's framing settled it:

> "Shouldn't concurrency be a function of what is possible from that hardware?
> User might have their own request. Based on their needs we provision n hosts to
> match the concurrency needs. Horizontal scaling."

Right. Demand is given, capacity is what we optimise, concurrency is `L = λW` and
falls out of both. The tool's output is **replicas needed** = demand ÷ capacity.

### Sweep the finalists, not every node

Also the user's:

> "Why not do this concurrency sweep to the final winning best configuration? Not
> to all nodes in the DAG? If a config is best at a particular concurrency, it is
> likely to be best across the different concurrency sweeps."

Correct for most nodes — curves rarely cross, so ranking at one concurrency ranks
them everywhere. **Except** `chunked_prefill` and the speculative-decoding family,
which cross by construction and in opposite directions. Those get two extra
levels; everything else is measured once at L\*.

### graph_capture runs last

`torch.compile` is keyed on shapes, so a cache warmed at one `max_num_seqs` is
invalidated by the next sweep value. Paying it per node would cost more than the
search. Everything earlier runs eager; compilation is tested once against the
accumulated config — which is also when it is worth most, on final shapes rather
than shapes about to change.

Measured at ~9 minutes on this hardware, not the ~40 the cost hint claimed.

### revert ≠ discard

A reverted config still enters the Pareto frontier. In run four,
`spec_decode_ngram` was reverted for scoring 3.1% below the incumbent — and it is
the **only** config that met the stated 500 ms p99 TTFT target, at 452 ms.
Everything the greedy walk kept sits at 600–608 ms and misses it.

A lineage-only frontier throws away exactly the operating points a user would
most want.

---

## Bugs that changed the design

Ordered by what they cost.

### The gate that could never pass

An accuracy gate demanded token-identical 48-token outputs — a *strict*
equivalence test — in a world where kernel selection makes runs only
*algorithmically* equivalent. At the measured 0.44%/token flip rate that is a
22.6% false-positive floor. Simultaneously the NIAH tolerance was 1%, exactly the
granularity of a 100-item set.

**The seed configuration failed its own gate.** Nothing could ever pass. Three
defects, one root cause: thresholds guessed rather than measured. Every threshold
in the project is now measured, and the four-way equivalence taxonomy exists to
stop the same conflation recurring.

### The drain bug

`_load` submitted at `qps` for `seconds`, then awaited **every** submitted task.
At 15.4 qps × 45 s that is ~690 requests; the semaphore let only ~60 start inside
the window, and `summarize` counts only in-window starts.

So ~630 requests contributed nothing to the measurement and took 20 minutes to
drain — a 45-second window measured in 21 minutes. Worse than the wall-clock:
pass 1's backlog was still executing when pass 2 began, so pass 2 measured a
server working through pass 1's queue.

Queued tasks are now cancelled at `t1`; only in-flight ones are awaited, which
also leaves the server idle between passes.

### `KeyError: 'prompt'` after nine launches

The context-length filter assumed every benchmark row has a `prompt` key.
MATH-500 rows carry `problem`. The prompt text was being built in two places —
inside each scorer and again in the filter — so they could disagree, and did.

It crashed at `lossless_complete`, and because `result.json` is only written
after `traverse()` returns, **all nine launches and ~75 minutes of GPU time were
discarded.** Trials are now journalled as they complete.

### The baseline existed only in scrollback

Stage 1.3 runs before `traverse()`, so it missed both the journal and
`result.json`. After run four finished, the number every percentage was computed
against was gone, and "+307%" had to be reconstructed from a previous run's
identical seed config.

> "We should abso-fucking-lutely capture baseline performance."

It is now written three times: a bordered console block, line 1 of the journal,
and a `result.json` key.

### Silent fallback produced a believable wrong number

Parameter count fell back to arithmetic over config fields when the checkpoint
index could not be read. A `NameError` was swallowed by a bare `except Exception`,
and the formula assumed MHA — reporting **16.45B for a 14.8B model**. An 11%
overestimate that looks entirely reasonable. A wrong value that looks fine is
worse than a crash.

### Hybrid attention: a 4x KV error

Only full-attention layers keep a cache that *grows* with context. Assuming every
layer caches gave 128 KB/token against 32 KB actual — at 256k context, 34.4 GB
predicted against 8.6 GB, the difference between "won't run" and "fits
comfortably".

### RULER prompts exceeded the served context

Generated at 16k–33k tokens, served under a right-sized `max_model_len` of 6144.
Every prompt would be rejected and the benchmark would score 0.0 — for *every*
config, reading as "quality unchanged" rather than "probe broken". Same shape as
the gate that could never pass. `run_benchmark` now filters and raises.

Shortening the contexts to fit then made RULER saturate at 1.00, which is its own
problem: it can still detect damage but has no headroom to rank.

---

## Environment constraints

- **The DGX runs live production.** Never modify the `dextract` conda env, never
  touch `dextract-ocr.service` or OpenSearch. Experimental work lives in
  `verticalinference`.
- **Two toolchains cannot share the serving env** and run as subprocesses with
  their own `--target` directory: aiconfigurator (`numpy~=1.26.4` vs vLLM's
  `<2.4`) and llmcompressor (`compressed-tensors==0.18.0` vs vLLM's `==0.17.0`).
- **A venv cannot be relocated** across a bind mount — console-script shebangs,
  the `bin/python` symlink and `pyvenv.cfg` all bake absolute paths.
  `pip install --target` plus PYTHONPATH has nothing to break.
- **PATH inheritance:** vLLM JIT-builds CUDA extensions and shells out to
  `ninja`, which lives beside the interpreter. A subprocess inherits only PATH,
  so a bare env dies deep in engine init with an error that reads like a model
  problem. Hit twice.
- **Unified memory:** `memory.total` reports `[N/A]` and `gpu_memory_utilization`
  is a fraction of *system* memory the CPU also competes for. 0.90 left 1.6 GB of
  headroom on a 122 GB box.

---

## Open

- **RULER saturates at 1.00** — needs contexts that fit `max_model_len` while
  still being hard.
- **Prefix caching does not shard cleanly.** It is per-replica, so round-robin
  load balancing across N replicas fragments the hit rate N ways. The measured
  4.24x is an upper bound on per-replica performance without prefix-aware
  routing.
- **HumanEval+ generates but refuses to score** — pass@1 needs a sandbox.
- **vLLM loading of locally-produced INT4/NVFP4 artifacts is unverified.**
  Production works; the load path has not been re-tested since the ninja fix.
- **Stage 3 is designed, not built.**
