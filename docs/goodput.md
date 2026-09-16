# `goodput`: measuring what a serving endpoint actually delivers

A standalone package for putting load on an OpenAI-compatible endpoint and
scoring what comes back. It has no dependency on the rest of this project: no
DAG, no fingerprint, no quantizer, no vLLM import. It speaks HTTP and counts
tokens as they arrive.

It was split out of `inferopt.evaluator` because the measurement bugs it encodes
are not inferopt bugs. The convoy, the constant output lengths, and the drain
inside the window are load-generation bugs that any harness can make, and
NVIDIA's AIPerf makes two of them today. Anything that measures an LLM endpoint
needs this code, not just this project.

---

## Install

It ships in this repo under `src/goodput`, declared alongside `inferopt` in
`pyproject.toml`, so `pip install -e .` gets both. The only runtime dependency
is `httpx`.

```python
from goodput import Latency, closed_loop, open_loop, summarize, export
```

---

## Quickstart

```python
import asyncio
from goodput import Latency, closed_loop, summarize, export

prompts = ["Explain the CAP theorem.", "Write a haiku about queues."]
lengths = [256, 64]                      # per-request output budgets

reqs, t0, t1 = asyncio.run(closed_loop(
    "http://127.0.0.1:8000", "Qwen/Qwen3-1.7B",
    prompts, lengths, 128,               # 128 concurrent requests
    settle_s=20, window_s=45))

slo = Latency(ttft_p99_ms=500, itl_p99_ms=250)
m = summarize(reqs, t0, t1, slo)
print(f"{m['goodput']:.1f} tok/s at {m['slo_attainment']:.1%} attainment")

export.write_run("runs/demo/L128", reqs, t0, t1, slo, prompts=prompts)
```

`t0` and `t1` bound the measurement window. They are returned rather than
assumed because settle and window are one continuous run, so the caller cannot
compute them from wall clock. See *Settle flows into the window* below.

---

## Choosing a driver

Both are async and both return `(reqs, t0, t1)`. They differ in what they hold
fixed, and that decides which question they can answer.

```python
closed_loop(base_url, model, prompts, max_tokens, conc,
            settle_s, window_s, cursor=None, stagger_s=0.0)

open_loop(base_url, model, prompts, max_tokens, qps, conc, seconds, cursor=None)
```

| | `closed_loop` | `open_loop` |
|---|---|---|
| fixes | concurrency | arrival rate |
| emerges | rate | concurrency |
| above capacity | converges, bounded by construction | queues grow without bound |
| use it to | find where goodput peaks | validate an operating point |

`closed_loop` is the right instrument for a sweep: it is bounded, so it
converges and repeats. It is the wrong instrument for validating a chosen
config, because holding concurrency constant removes the burstiness that moves
the TTFT tail, and burstiness is what production has.

`open_loop` matches production arrivals but does not converge above capacity, so
a 45s window and a 90s window give different answers. Two runs of this project
were thrown away for measuring exactly that.

**Sweep closed, validate open.**

---

## `max_tokens`: pass a sequence, not a number

Both drivers accept either a scalar or a per-request sequence.

```python
closed_loop(url, model, prompts, 256, ...)          # every request asks for 256
closed_loop(url, model, prompts, lengths, ...)      # request i asks for lengths[i]
```

Prefer the sequence, and pair it with the prompt from the same trace row. A
4431-token prompt does not ask for a 102-token answer, and splitting them
replays a workload that never existed.

Giving every request the same output length is not a harmless simplification.
Identical durations mean workers that start together also finish together, so a
convoy never disperses. Measured on this project at concurrency 128, two
identical drives returned **205.1 and 2181.6 tok/s**, a 10.6x spread from
nothing but where the window landed against the convoy. With per-request lengths
the same pair returned 1891.0 and 1893.8.

---

## Settle flows into the window

This is the load-bearing detail, and it is the one most harnesses get wrong.

`settle_s` and `window_s` are **one continuous run**. The driver starts, lets
the pipeline fill for `settle_s`, then opens the measurement window without
pausing. Requests in flight when the window opens stay in flight.

The alternative, running settle as a separate call that drains before
measurement starts, means the window opens on an idle server with every worker
firing at once. Those first `conc` requests all queue behind each other for
prefill, and their TTFT is queue position, not latency.

NVIDIA's AIPerf does exactly this. Measured on the same server, same trace, same
per-row output lengths, at concurrency 128:

| | ours | AIPerf |
|---|---|---|
| arrivals in the first 0.5s of the window | 2 of 310 | **128 of 390** |
| peak/mean arrival rate | 7.3x | **146.7x** |
| ttft p99 | 375 ms | **2390 ms** |
| slo attainment | 1.000 | **0.723** |
| itl p99 | 77.8 ms | 76.5 ms |

ITL agrees because it is unaffected. The TTFT tail is a staircase by arrival
position within the burst: the first 16 arrivals averaged 399.7 ms, the last 16
averaged 2268.8 ms. Once the burst cleared, AIPerf's mean TTFT was 236.6 ms,
the same regime we measure throughout.

`stagger_s` spreads worker starts over the given interval. It is defence in
depth and not the thing that matters: with settle flowing into the window,
turning the stagger off changed nothing measurable (throughput 1760.7 vs 1761.3,
attainment 1.000 both).

---

## `cursor`: do not replay the same prompts

```python
cursor = [0]
for phase in range(3):
    reqs, t0, t1 = asyncio.run(closed_loop(url, model, prompts, lengths, 128,
                                           settle_s=20, window_s=45, cursor=cursor))
```

A mutable `[int]` carried across calls. Each phase starts where the last one
stopped, so successive phases serve different prompts.

Without it, every phase replays the same prompts into a warm prefix cache and
scores full-prompt hits that no production workload produces. This is not
theoretical: running one config three times against a single server drove the
mean prefill tokens actually computed from **348.2 down to 16.8**, a 97% cache
hit rate, making each run look faster than the last for no reason related to the
config.

The cursor wraps once total requests exceed `len(prompts)`, so give it a trace
longer than a full sweep will consume, or accept that late phases meet a partly
warm cache.

---

## The SLO

```python
from goodput import Latency

Latency(ttft_p99_ms=500, itl_p99_ms=250)   # both bounds
Latency(ttft_p99_ms=500)                   # itl unconstrained
Latency()                                  # goodput == throughput
```

`LatencyTarget` is a `Protocol`, so you can pass your own richer SLO object
instead as long as it exposes `ttft_p99_ms` and `itl_p99_ms`. That is how
inferopt passes its own pydantic `SLO` in unchanged, quality budgets and all.

Despite the `_p99` names, these are **per-request** bounds. `Req.meets(slo)` is
the predicate goodput is built on: a request counts only if its own TTFT and its
own mean ITL are both inside the targets. A 1-token response is exempt from the
ITL bound, since no interval exists to measure.

---

## What `summarize` returns

```python
m = summarize(reqs, t0, t1, slo)
```

| key | meaning |
|---|---|
| `goodput` | tok/s from SLO-conforming requests. **The objective.** |
| `goodput_req_s` | the same thing in requests/sec |
| `throughput` | tok/s from all requests |
| `throughput_req_s` | requests/sec, all requests |
| `slo_attainment` | fraction of started requests that met the SLO |
| `ttft_p99_ms`, `ttft_p95_ms` | TTFT percentiles |
| `itl_p99_ms`, `itl_p95_ms` | ITL percentiles |
| `ttft_n` | sample count both percentiles are drawn from |
| `completed`, `failed` | request counts |
| `failure_reasons` | the distinct reasons, most common first |
| `window_s` | window length |

Three things about these numbers that are easy to get wrong:

**`throughput = goodput + late tokens.`** Goodput counts tokens only from
requests that met the SLO. A config that serves everything slowly scores zero.

**Both goodput units are reported on purpose.** vLLM's own benchmark reports
requests/sec; this project optimises tokens/sec, which is the right objective
when responses vary in length. They differ by roughly mean output tokens, so
reporting only one invites a silent rescaling by a factor of a few hundred.

**`slo_attainment` divides by requests that STARTED, failures included.** A
config where a fifth of requests error out but the rest are quick would
otherwise report 100% attainment.

**Treat `ttft_p99_ms` with suspicion when `ttft_n` is small.** A 45s window at
concurrency 128 completes about 384 requests, so p99 is a maximum over the
slowest four, not a percentile. Across three identical launches of one config it
varied 6.63x at concurrency 64 while goodput over the same launches varied 1.06x.
`ttft_p95_ms` is reported beside it for that reason. Compare configs on goodput
and attainment; use p99 for the SLO it is written against, not for ranking.

---

## Export

```python
export.write_run(out_dir, reqs, t0, t1, slo, *,
                 config=None, run_info=None, server_metrics=None,
                 prompts=None, output_lengths=None) -> Path
```

Writes an artifact directory in the schema AIPerf uses, so anyone holding an
AIPerf run can diff it against ours with their own tools:

```
profile_export_aiperf.json    aggregate, percentiles, run config      always
profile_export.jsonl          one record per request, epoch timestamps always
profile_export_console.txt    the same numbers for a human            always
server_metrics_export.json    whatever you scraped from /metrics      if server_metrics=
inputs.json                   the prompts and lengths actually sent   if prompts=
```

The last two are written only when you pass the corresponding argument. They are
omitted rather than written empty, for the same reason unmeasured fields are
absent rather than zero: an empty file reads as "scraped, and there was nothing".

All 20 core AIPerf fields are emitted with identical stat-dict keys
(`avg/min/max/std/sum/count/p1..p99`).

Two deliberate choices. The files carry `producer: goodput/0.1.0` and **no**
`aiperf_version`: they are AIPerf-shaped, not AIPerf-produced, and the only
thing worse than an unreadable measurement is one that misrepresents who took
it. Fields we do not measure (GPU telemetry, HTTP transport timings) are
**absent rather than zero-filled**, because a zero reads as "measured, and it
was zero".

Our tokens/sec goodput lives under `goodput_extensions`, since AIPerf's
`goodput` field is requests/sec:

```python
agg = export.aggregate(reqs, t0, t1, slo)
agg["goodput"]["avg"]                                  # requests/sec, AIPerf's unit
agg["goodput_extensions"]["goodput_tokens_per_sec"]    # ours
```

`aggregate()` and `records()` return the same data in memory if you want to
route it somewhere else; `console(agg)` renders the text table.

---

## Using it outside inferopt

Nothing in the package imports inferopt, and `tests/test_goodput.py` asserts
that. To lift it into another project, copy `src/goodput/` and its test file.
The test imports only `goodput`, so it travels intact.

You supply the server. The package does not launch, configure, or tear down
anything; it measures whatever is already listening at `base_url`. Server
lifecycle stays with the caller, which is why `inferopt.evaluator` kept
`_serve`.

---

## Tests

```bash
python tests/test_goodput.py          # 83 checks
./run_tests.sh                        # the whole suite
```

The suite was checked by mutation rather than by coverage: six deliberate breaks
were introduced and all six failed the suite. Attainment denominator changed
from started to done, ITL dividing by `n_out` instead of `n_out - 1`, the
percentile convention drifting between `summarize` and `export`, a false
`aiperf_version`, tokens/sec written into the requests/sec field, and
`input_sequence_length` zero-filled when the server never reported it.

If you change the percentile convention, change it in both `metrics.summarize`
and `export._pct`. A test fails if they disagree, because otherwise the same run
reports two different p99s.
