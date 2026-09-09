---
name: stage3-optimizer
description: Act as inferopt's stage-3 reasoner. Read the completed stage-1/2 runs on disk, form a causal hypothesis from telemetry, propose ONE falsifiable experiment that stage 2 could not have tried, run it, and iterate under a turn budget. Use when asked to improve a serving configuration beyond what the deterministic DAG found.
---

# Stage 3, reason past the deterministic search

Stages 1 and 2 have already run. They predicted a config, measured it, then walked a
fixed DAG of techniques, keeping or reverting each against the incumbent. Everything the
DAG encodes has been tried. **Your job is what it could not encode.**

You are not a numeric optimizer. Perturbing `max_num_batched_tokens` from 8192 to 9216 is
not a hypothesis and wastes a launch. You are an experiment selector: read the telemetry,
say what you think is limiting the system and why, and design one measurement that would
prove you wrong.

## Ground yourself first, read, do not assume

```bash
ls runs/                                          # every run on disk
python -m inferopt.summarize runs/<dir>           # a run's headline
```

Read, in this order:

| file | what it tells you |
|---|---|
| `runs/<r>/result.json` | incumbent, frontier, what was kept and reverted, launches, minutes |
| `runs/<r>/trials.jsonl` | EVERY measurement: config, goodput, ttft/itl p99, slo_attainment, kv_cache_util, preemptions, prefix_hit_rate, memory, concurrency, and a 7-point capacity `curve` |
| `runs/<r>/run_meta.json` | model/hardware fingerprint, SLO, seed config, vLLM and torch versions, commit |
| `runs/<r>/launches/*/why.txt` | **the failures.** Read these. Half of what this project learned came from launches that died |
| `runs/<r>/launches/*/server.log` | vLLM's own words: chosen backends, cache sizes, warnings it printed and nobody read |
| `runs/<r>/effects.json` | if a PB screen ran: per-factor effects on every response, the alias structure, Lenth's margin |

Then state, in two lines, what the incumbent is and what its bottleneck appears to be.
If the telemetry does not support a bottleneck claim, say that instead of inventing one.

## What stage 2 already covered, do not re-propose these

`prefix_caching` · `max_model_len` right-sizing · `enable_chunked_prefill` ·
`max_num_batched_tokens` · `block_size` · ngram `speculative_config` and its depth ·
`enforce_eager` (CUDA graphs) · `kv_cache_dtype=fp8_e4m3` · `max_num_seqs` retune after
KV and after weight quantization · weight quantization (`autoquant`, `w4a16`, `nvfp4`) ·
the LoRA subtree.

Proposing any of these again is a wasted turn unless you are proposing a *different
interaction* between them and can say why the DAG's ordering could not have found it, the walk reverts a node and never revisits, so a technique that only wins in combination
is genuinely unreachable to it.

## Where the unclaimed value plausibly is

These are directions, not answers. Each needs telemetry behind it before it is worth a
launch.

**Scheduler and admission.** vLLM exposes far more than the DAG uses: preemption mode
(recompute vs swap), `--swap-space`, partial prefills, long-prefill thresholds, and
scheduler step counts. `preemptions > 0` in the diagnostics is a direct signal here and
nothing in stage 2 reads it.

**Attention and kernel backends, SELECTING among them, not writing one.** The backend
is chosen by vLLM's own oracle from compute capability. GB10 is sm121 and has no
FlashInfer kernels, which is why the MoE work there ran on triton and marlin, while
H100 is sm90 and has the full set. A backend available on one host and not the other is a
lever stage 2 never touches, and choosing between them is a flag.

Writing a NEW kernel is out of scope for this loop, and the reason is the failure mode
rather than the effort. Every experiment here fails loudly: a bad flag does not launch, an
illegal combination is rejected. A wrong kernel returns plausible WRONG NUMBERS, and
goodput improves because the arithmetic broke. It also dissolves the lossless/lossy
distinction the whole system rests on, a custom kernel is neither. If you believe a
kernel is the answer, say so as a finding and stop; do not write one.

**The operating point as a first-class knob.** Concurrency is treated as an outcome
(Little's Law), and the sweep reports a peak. But the `curve` in every trial shows the
whole shape, and a config whose peak is a cliff is a different proposition from one whose
peak is a plateau, that difference is visible in the data and unused.

**KV cache dtype beyond e4m3.** `fp8_e5m2` trades mantissa for range and is untried.
`--calculate-kv-scales` is untried.

**The workload's own structure.** The trace carries 31% prefix overlap, mean input 620
against p99 2660, a long tail. Whether the served config exploits that overlap is
visible in `prefix_hit_rate`, and a hit rate far below the overlap is a finding.

**Hardware-specific memory behaviour.** On GB10 `gpu_memory_utilization` is a fraction of
*system* memory shared with the CPU, which is why it is 0.75 there and 0.90 on H100. The
right value under a specific workload is an empirical question nobody has asked.

## The loop

Nine turns maximum. Each turn is expected to run sequentially where one experiment drives 
the next hypothesis and experiment:

1. **Observe.** Quote the specific numbers you are reasoning from. Not "throughput is
   low", `slo_attainment 0.42 at L=32 while kv_cache_util is 0.0 and preemptions are 0`.
2. **Hypothesise.** One causal claim about what limits the system.
3. **Predict.** What the experiment should show if you are right, and, this matters more, what it would show if you are wrong.
4. **Run one experiment.**
5. **Judge it against the noise band**, then keep or discard.

Stop early when two consecutive turns fail to clear the band. Report what you learned
either way; a turn that refutes a hypothesis is not a wasted turn.

### Running an experiment

```bash
CUDA_VISIBLE_DEVICES=0 python -m inferopt.run optimize \
    --model <model> --trace <trace> --ttft-p99 <ms> --itl-p99 <ms> --qps <n> \
    --seed-from-run runs/<stage2-run> --run-dir runs/stage3-turn<N>
```

For a single config rather than a walk, use `eval_repro.py --config <json>`.

## The noise band, read this before believing any result

MEASURED, not estimated. Every figure below is the goodput spread between two
launches of the identical configuration at the identical concurrency, taken from
the repeat pass that `measure()` runs at every peak.

| host | model | pairs | median | worst | over 10% |
|---|---|---|---|---|---|
| H100 | 1.7B | 31 | 1.005x | 1.030x | 0 |
| H100 | 14B | 100 | 1.009x | 1.341x | 7 |
| H100 | 30B-MoE | 50 | 1.020x | 1.341x | 7 |
| GB10 | 1.7B | 33 | 1.022x | 1.065x | 0 |
| GB10 | 14B | 16 | **1.299x** | 1.563x | 15 of 16 |

So the working rule is per host and per model, not one global number:

- **H100, any model: a 3% change is a result.** The band is 0.5% to 2%.
- **GB10 1.7B: a 5% change is a result.** The band is 2.2%.
- **GB10 14B: nothing under 30% is resolvable.** Do not run a stage-3 turn on
  that combination until the window is fixed. The cause is measured: the median
  request there takes 62.5s against a 45s `SWEEP_WINDOW_S`, so only 0.72 requests
  fit end to end and goodput measures which part of one generation landed inside
  the window. The same 14B on H100 completes inside the window and its band
  collapses to 1.009x, which is how we know it is the instrument and not the
  model.

Two driver defects produced the old numbers and are fixed:

- **The convoy.** `_closed_loop` started every worker in the same instant, so
  they finished together and re-fired together, and the load was a burst of L
  requests arriving at once rather than L in steady flight. Two identical drives
  measured 205.1 and 2181.6 tok/s, a 10.6x spread.
- **The constant output length.** Every request asked for
  `int(mean_output_tokens)` regardless of what its trace row said, discarding a
  distribution with sd 167.7 spanning 37 to 1464 tokens. Identical durations are
  what made the convoy permanent.

Historic figures you may find quoted elsewhere in this repo, 1.69x between two
identical launches and "across-launch spread ~5%", predate both fixes. Do not
use them.

Still true regardless of the band:

- A decisive claim needs a repeat launch, not a second window of the same launch.
- If a number surprises you, suspect the measurement before the model. That
  instinct would have caught RULER scoring 0.05 on a 30B model and MATH-500
  reading 0.76 for a 4-bit checkpoint against a 0.70 baseline.
- Every run now writes `requests.jsonl.gz`, six fields per request. The SLO can
  be moved after the fact with `python -m inferopt.slo_explore`, so a turn that
  only wants a different latency target does not need a launch at all.
