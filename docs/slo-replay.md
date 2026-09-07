# Replaying a run at a different SLO

Every run writes `requests.jsonl.gz`: one row per request, for every request of
every measurement window. It exists so the SLO can be changed **after** the GPU
work is done, which is what makes an interactive TTFT/ITL slider possible.

Runs made before September 2026 do not have it. See *Which runs have it* below.

---

## The problem it solves

The SLO is a guess made before anything is measured. This project used
`ttft_p99 = 500ms, itl_p99 = 250ms` throughout, and those numbers were picked
out of the air. Everything downstream is conditional on them:

- **goodput** counts only requests that individually met the bound
- **SLO attainment** is the fraction that met it
- **replicas** = demand ÷ goodput_req_s
- **cost** = replicas × GPU price

Change the promise and all four move. Before this capture existed, changing the
promise meant re-running the GPU — a 41-hour proposition across the three model
families.

**And a p99 cannot substitute.** This is the part that surprises people. Goodput
is not derivable from `ttft_p99_ms`: it depends on *which* requests conformed,
and an aggregate has already thrown that away. Knowing that p99 TTFT was 274ms
tells you nothing about how much goodput survives a 200ms bound. The per-request
membership is the irreducible thing, so the per-request record is what gets kept.

---

## What is captured

Six numbers per request, written by `VllmEvaluator.dump_requests`:

| key | meaning | unit |
|---|---|---|
| `s` | start time, relative to the window opening | seconds |
| `t` | time to first token, `null` if the request failed | ms |
| `l` | end-to-end latency | ms |
| `n` | output tokens produced | count |
| `w` | tokens whose arrival fell **inside** the window | count |
| `k` | the request completed | bool |

Preceded by one meta row per measurement point:

```json
{"_":"meta","node":"kv_cache_fp8","L":128,"phase":"closed_loop",
 "win_s":45.0,"n":384,"slo":{"ttft_p99_ms":500.0,"itl_p99_ms":250.0}}
```

### Why exactly these six

`Req.meets()` — the predicate goodput rests on — reads

```python
ttft_ms <= bound  and  (latency_ms - ttft_ms) / (n_out - 1) <= itl_bound
```

so `t`, `l`, `n` and `k` reconstruct it at any threshold. `w` supplies the
goodput numerator, and `s` decides window membership. Nothing else is needed and
nothing else is stored.

**`token_times` is deliberately NOT stored.** It is ~260 floats per request —
roughly 40x the whole row — and buys only sub-request timing, which no SLO here
asks about. Note the consequence: the SLO uses *mean* inter-token latency, so a
future SLO phrased on *max* ITL or on a per-request ITL percentile could not be
replayed from this file. If that requirement ever appears, `token_times` has to
start being written, and old runs will not be able to answer.

**Rows that start before the window are kept**, with a negative `s`.
`summarize()` counts their tokens toward throughput while excluding them from the
percentiles and the started set, so a reader that dropped them would silently
disagree. This is asserted in the tests.

---

## How often it is written

Once per measurement point — every request, never sampled.

| call site | when | count |
|---|---|---|
| `VllmEvaluator._point` | each level of the closed-loop sweep | `SWEEP_LEVELS = (4,8,16,32,64,128,256)`, plus the repeat pass at the peak, plus a bracket extension to 512 when the peak lands at the top — **≈9 per launch** |
| `VllmEvaluator.serving_metrics` | each pass of the open-loop pinned path | `REPEATS = 2` |

Measured volume, on a launch with realistic 37–1464-token outputs:

```
one launch      9 points, 1932 request rows,  35 KB gzipped  (3.9 KB/point)
PB screen      20 launches  ->  0.7 MB
seqDAG walk    10 launches  ->  0.3 MB
full re-run   ~110 launches ->  4 MB
```

Four megabytes for forty-one GPU-hours. Appended as gzip members, so a killed run
keeps everything already written, and wrapped in a `try/except` that logs rather
than raises — telemetry must never be able to lose a launch.

---

## Reading it

```bash
# one threshold, every measurement point in the run
python -m inferopt.slo_explore runs/rerun-1.7b-pb --ttft 300 --itl 200 \
    --demand-qps 16 --gpu-hourly-usd 3.0

# the whole grid a slider indexes into, rather than recomputing per frame
python -m inferopt.slo_explore runs/rerun-1.7b-pb --sweep \
    --demand-qps 16 --gpu-hourly-usd 3.0 --out runs/rerun-1.7b-pb-grid.json
```

Or from Python:

```python
from inferopt.slo_explore import load, recompute

for meta, rows in load("runs/rerun-1.7b-pb"):
    m = recompute(meta, rows, ttft_ms=300, itl_ms=200,
                  demand_qps=16, gpu_hourly_usd=3.0)
    print(m["concurrency"], m["goodput"], m["slo_attainment"], m["replicas"])
```

---

## What a slider can and cannot move

**Moves with the SLO:**

- `goodput` (tok/s) and `goodput_req_s`
- `slo_attainment`
- `replicas` — `ceil(demand_qps / goodput_req_s)`; a count, so it steps rather
  than glides, and the steps are where the interesting cliffs are
- `usd_per_hour`, `usd_per_mtok`

**Does not move, correctly:**

- `throughput` — the server did the same work; only what *counts* changed.
  A slider where throughput moves with the SLO has a bug.
- `ttft_p99_ms`, `itl_p99_ms`, `ttft_p95_ms`, `itl_p95_ms` — these are
  properties of the measured distribution. The SLO is a line drawn across
  them, not an input to them. Plot the bound as a movable line *over* a fixed
  distribution; that is the honest visual.
- accuracy — a property of the configuration, not of the promise.

**Cannot be answered at all** without new GPU work: any change to the
configuration or the workload. Sliding `max_model_len`, swapping a quantization,
or changing the arrival rate all require re-measurement. The replay answers
"what if I had promised something different", never "what if I had run
something different".

### The GPU price is yours

No run captures a price, because a price is not a measurement — it depends on
your cloud, your commitment, your amortization. `--gpu-hourly-usd` is an input.
Without it, `slo_explore` reports replicas and stops.

---

## The exactness guarantee

The reconstruction is exact, not approximate. `test_slo_explore` in
`tests/test_dag_unit.py` asserts that at a run's **own** SLO, `recompute()`
reproduces the recorded `goodput`, `throughput`, `goodput_req_s`,
`throughput_req_s`, `slo_attainment`, both p99s, both p95s, `ttft_n` and
`completed` to within 1e-4 relative.

That check is the contract. If the stored fields ever become insufficient — a new
metric in `summarize()`, a change to `Req.meets()` — it fails loudly rather than
letting `slo_explore` return something plausible and wrong.

Mutation-checked: a reader using *max* instead of *mean* ITL fails 6 checks;
dropping the pre-window rows fails 2.

**If you change `Req.meets()`, change `slo_explore._meets()` in the same commit.**
They are two implementations of one predicate, which is exactly the shape of
defect this project has been bitten by before — see `docs/decision-log.md` on the
two implementations of prompt building.

---

## Which runs have it

Only runs made after the per-request capture landed. Everything in `runs/1.7b-*`,
`runs/14b-*`, `runs/moe-*` and `runs_h100/*` predates it and stores aggregates
only; `load()` raises `FileNotFoundError` on those, with the reason.

Those runs also predate two load-driver fixes — the worker stagger and
per-request replay lengths — so their numbers are not comparable with anything
produced after. `rerun_all.sh` regenerates the whole set against the corrected
driver, with capture on. See `docs/decision-log.md`.
