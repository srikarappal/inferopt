# Stage 3 on the 1.7B, GB10

Three turns, ten launches, about two and a half hours. Nothing cleared the noise
band. That is the result, and the way it failed is more useful than the number.

Incumbent going in: `max_model_len_rightsize`, 1968.8 tok/s at L=256, from the
lossless walk. Noise band on this host and model is 1.022x median, 1.065x worst,
so the bar was a 6.5% improvement.

## Turn 1: prefill head of line blocking. Refuted.

The curve is a cliff. L=256 gives ttft_p99 500ms at 0.99 attainment, L=512 gives
36832ms at 0.00, while throughput only falls 2016.9 to 1830.2. The server keeps
producing tokens; TTFT collapses. With `enable_chunked_prefill` off, one long
prompt in a step blocks every decode behind it, and this trace has p99 input
2660 against a mean of 620.

Bounding the per step token budget did not move it:

| arm | L=512 ttft_p99 | peak goodput |
|---|---|---|
| control | 39021ms | 1797.8 |
| chunked, 1024 token budget | 38508ms | 1832.8 |
| chunked, 2048 token budget | 38846ms | 1811.0 |

1.019x on the peak, and the cliff is untouched. If prefill scheduling caused it,
chunking would have bounded it.

A first attempt used `long_prefill_token_threshold` with
`max_num_partial_prefills`. All three flags parse, so `installed_flags()`
accepted them, and vLLM 0.26's V1 engine then refused at startup with
`NotImplementedError: Concurrent Partial Prefill is not supported`. Every config
now gets a two minute preflight launch before it can consume a measurement slot.

## Turn 2: residency cap. Mechanism confirmed, objective blind.

`max_num_seqs` is 256 and the cliff is at exactly L=512, twice that, with
kv_cache_util 0.184 and zero preemptions. At L=512 half the requests are not
slow, they are not resident. Their TTFT is a queue wait for a slot.

| max_num_seqs | L=512 ttft_p99 | L=512 kv | peak goodput |
|---|---|---|---|
| 256 | 37063ms | 0.181 | 1887.8 |
| 512 | 1675ms | 0.301 | 1913.7 |
| 1024 | 1162ms | 0.302 | 1943.5 at L=256 |

TTFT at L=512 fell **32x** and KV usage doubled exactly as predicted, so the
mechanism is not in doubt. Peak goodput moved 1.014x, inside the band, because
the peak stays at L=256 either way.

**This is a finding the objective cannot express.** A configuration that
survives a concurrency overshoot is not the same product as one that falls off a
37 second cliff, and peak goodput scores them identically. If the deployment can
ever exceed its planned concurrency, `max_num_seqs 512` is strictly better than
256 for free.

## Turn 3: the memory fraction. Refuted.

`gpu_memory_utilization` is 0.75 in every run this project has ever done on
GB10. It was chosen as a boot requirement, since 0.90 ran a 122GB box into the
OOM killer, and never swept as a performance knob. On unified memory it is a
fraction of SYSTEM memory the CPU shares, and the server reserves 91GB to touch
17GB of KV.

| fraction | peak goodput | vs control | L=256 kv |
|---|---|---|---|
| 0.45 | 1706.8 | 0.919x | 0.305 |
| 0.60 | 1865.7 | 1.005x | 0.224 |
| 0.75 | 1856.5 | 1.000x | 0.178 |
| 0.85 | 1771.8 | 0.954x | 0.155 |

No direction and no signal. 0.75 is not leaving anything on the table, and the
default is right for the wrong-sounding reason.

## What the three turns together establish

Peak goodput is pinned at roughly 1900 tok/s at L=256 by **ttft_p99 crossing the
500ms SLO**, measured at 494 to 569ms across every arm of every turn, while
itl_p99 sits at 134ms against a 250ms target with room to spare. It is not
pinned by capacity, by prefill scheduling, by residency or by the memory
reservation.

Ten launches found nothing worth 6.5%. The reasonable reading is that the stage
2 answer of 1968.8 is close to what serving configuration alone offers for this
model on this host under this SLO, and that the remaining headroom is not in
this search space.

Where it is: the lossy branch. The 14B lossy walk on this host reached 966.9
against 197.3 lossless, 4.9x. The MoE lossy walk reached 2148.0 against 90.3,
23.8x, at identical accuracy. **No lossy walk has ever been run on the 1.7B, on
either host.** That is the obvious next experiment and it is a stage 2 run, not
a stage 3 one.
