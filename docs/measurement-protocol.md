# Measurement protocol

How inferopt measures a serving configuration, and why each choice is what it is.
Every decision here was made after a specific measurement failure; the failures
are in `docs/decision-log.md`.

---

## The objective: goodput

**Goodput** is tokens per second counted only from requests that met their SLO.
A request that misses its TTFT or ITL target contributes zero, however many
tokens it produced.

Raw throughput rewards the wrong thing. A server maximises tokens/sec by queueing
deeply and serving everything slowly — a magnificent aggregate number while every
user stares at a blank screen. Three configs with identical 200 tok/s raw
throughput can score 200, 100 or 0 on goodput depending on SLO attainment.

Measured on this project: prefix caching moved throughput 179 → 197 tok/s (+10%)
and goodput 47.9 → 199.5 (+316%). The machine did not get faster. Answers started
arriving on time, so they started counting.

---

## Concurrency is an outcome, not an input

This was the single largest measurement error in the project's history.

Three quantities that are easy to conflate:

| | what it is | who sets it |
|---|---|---|
| demand | requests/sec the business needs | the user. Given. |
| capacity | requests/sec one replica sustains **at the SLO** | hardware + config. **What we optimise.** |
| concurrency | in-flight requests | neither — `L = λW`, Little's Law |

Concurrency was originally a fingerprint field computed as `max(1, int(qps * 2))`.
That does not typecheck: qps is 1/time, concurrency is dimensionless. You cannot
get one from the other without multiplying by a *time*, and which time is the
entire question.

It produced 30 for a workload where the server could hold ~110 and the
measurements showed ~22 sequences actually decoding. Every number in runs three
and four was taken at a concurrency nobody chose, on an unnamed point of a curve.

**Goodput as a function of concurrency has a peak.** Below it, adding concurrency
adds throughput. Above it, queueing pushes requests past the deadline faster than
throughput rises, so goodput falls.

```
 goodput                     peak = capacity at the SLO
 (tok/s)                   ↓
                      ___-***-___
                  __--           --__
              __--                   --__
          __--                           --__
      __--                                   --
    -
    └──────────────────────────────────────────────→  concurrency
    4    8   16   32   64  128  256
```

That peak is simultaneously the maximum goodput, the best concurrency, and the
sustainable req/s that divides into demand to size a fleet. Finding it is the
whole job.

---

## Two load drivers, for two different questions

### Closed loop — for characterising capacity

Hold exactly `L` requests in flight, replacing each as it completes. Bounded by
construction, so it converges and repeats.

### Open loop — for validating an operating point

Submit at the trace's arrival rate regardless of whether the server keeps up.
Matches production: users arrive when they arrive.

**Above capacity, open loop does not converge.** Queues grow without bound, so
TTFT depends on how long the window ran — a 45s window and a 90s window give
different answers. Runs three and four were both doing this at ~22x
oversubscription, which is why their TTFT numbers cannot be compared to anything.

So: **sweep closed, validate open.** Closed loop finds the peak; the open-loop
replay confirms it survives the trace's real burstiness, which closed loop
removes by construction.

---

## The sweep

Run on an **already-running server**. This is what makes capacity measurement
affordable: the ~4 minute model load is paid once and amortised across every
point. Relaunching per level would cost more than the entire traversal.

```
levels    4, 8, 16, 32, 64, 128, 256      geometric
each      20s settle + 45s window
stop      when goodput falls twice running
```

Early termination matters — past the peak, more concurrency only pushes requests
over the deadline, and those points cost time to learn nothing.

### Where the sweep runs

| stage | what | cost |
|---|---|---|
| 1.3b | seed config → finds **L\***, the operating point | ~10 min, once |
| traversal | every node measured at L\* | unchanged |
| crossing nodes | also at L\*/2 and 2·L\* | ~+1 min each |
| finalists | full sweep on the top frontier configs | ~15 min each |

### Why not sweep every node

Most goodput curves do not cross — they sit uniformly above or below each other,
so ranking at one concurrency ranks them everywhere. `prefix_caching` removes
prefill work at every L; `graph_capture` removes per-step CPU overhead at every L.
Sweeping those is waste.

**Two families do cross**, and they are marked `curve_crosses` in the DAG:

- **`chunked_prefill`** is *negative* at low concurrency — chunking a prefill
  that could run in one shot is pure overhead — and *positive* at high
  concurrency, where it stops a long prefill blocking every decode behind it.
- **Speculative decoding** is the mirror. It spends spare compute to amortise
  weight reads, so it wins at low concurrency and can go negative at high
  concurrency, where weight reads are already amortised across sequences and
  rejected drafts are pure waste.

Both were reverted in run four on measurements taken at L=30 — inside the noise
band, at an arbitrary concurrency. Those are exactly the decisions a single-point
measurement gets wrong.

---

## Windows, passes and the accept band

```
warmup     45s     fills the prefix cache; 15s did not
window     45s     fixed duration, not fixed request count
passes     2       combined with MIN, not mean or max
```

**Fixed duration, not fixed count.** A fixed request count on a fast config
measures mostly ramp-up; the parameters under search are invisible until the
scheduler is saturated.

**Warmup is 45s because 15s was not enough.** Before prefix caching was enabled,
the two passes were identical (47.9/47.9). After, they were 42% apart
(137.9/195.2) — pass 1 ran with a cold cache, pass 2 warm. Both passes must
measure the same steady state, which is also the state production runs in.

**Passes combine with `min`.** Two samples have no median, and the code that
claimed to take one (`sorted(passes)[len//2]`) took the larger. Combined with
`best = max(variants)`, a 2-variant node scored as the max of four draws — around
+3–6% above its true mean, against a 5% accept band. A node with no real effect
could clear the bar on noise alone and then raise the incumbent for everything
after it. The gate asks "is this reliably better", so the conservative estimate
is the honest one.

**The accept band is measured, not guessed.** 3.8%, from the worst across-launch
throughput spread (1.91%) doubled. A guessed 2% would reject real wins. It is
frozen for the duration of a run: a band that moved mid-traversal would leave
early and late decisions resting on different criteria, with one frontier built
from both.

---

## Quality: every point gets a coordinate

The frontier is not only goodput against latency. Someone deploying a reasoning
workload will trade throughput for accuracy without hesitating, so every measured
config carries a quality score.

| node class | how quality is obtained |
|---|---|
| baseline (stage 1.3) | **measured** — MATH-500 and RULER |
| lossless | **inherited** from the baseline, marked `~` |
| lossy | **measured** |

A lossless node cannot move quality — that is what makes it lossless — and the
equivalence probe is a *stronger* check than a 100-sample benchmark would be. If
the first K tokens match the reference, quality is identical by construction; a
benchmark could only give a noisier version of the same answer. Inheriting is
therefore both correct and free, and it is what makes the ~2 hour budget work.

Quality is scored as the **worst** benchmark, never the mean. A config that
preserves its average by collapsing on one task is not an acceptable operating
point.

---

## Equivalence, and what "lossless" actually means

Four classes, and conflating them was the most expensive error in the project:

| class | means | example |
|---|---|---|
| strictly equivalent | bit-identical output | rare in practice |
| algorithmically equivalent | distribution preserved, FP details differ | tensor parallelism, kernel swaps |
| quality-equivalent | statistically indistinguishable | what the lossless branch means |
| lossy | deliberate, measurable degradation | quantization |

Measured per-token flip rate under greedy decoding is 0.44%. Comparing whole
outputs therefore has a **19% false-positive floor** — a strict-equivalence test
applied to an algorithmically-equivalent world. Comparing the first 11 tokens
gives 3.5%. The prefix length comes from calibration, not from a guess.

---

## Outputs

Every run writes:

```
runs/<name>/trials.jsonl     line 1 is the baseline, then one line per measurement,
                             flushed as it completes so a crash cannot discard it
runs/<name>/result.json      baseline, capacity curve, operating concurrency,
                             demand, every trial, the frontier, finalist curves
runs/<name>/launches/*/      server logs per launch
```

and prints a `BASELINE` block, a `CAPACITY` block with the replica count, and a
frontier table carrying `vs base`, quality, TTFT, concurrency and replicas.

`python plot.py runs/<name>/result.json` renders the frontier twice: goodput
against TTFT for a latency-first reader, and goodput against accuracy for a
quality-first one.
