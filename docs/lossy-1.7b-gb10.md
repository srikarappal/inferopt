# The lossy walk on the 1.7B, GB10

First lossy walk ever run on this model, on either host. 17 launches, 238
minutes. The result is 2.53x at unchanged accuracy, and the reason it is 2.53x
rather than the 23.8x the MoE got is the useful part.

## What shipped

| node | goodput | math_500 | |
|---|---|---|---|
| stage_1_3 | 1300.8 | 0.58 | seed |
| lossless_complete | 1915.1 | 0.59 | everything lossless, combined |
| kv_cache_fp8 | 4039.8 | 0.59 | |
| retune_batching_after_kv | 4846.8 | 0.59 | **accepted, L=512** |
| weight_autoquantize | 5487.3 | **0.31** | rejected by the quality gate |

**1915.1 to 4846.8, 2.53x, at identical accuracy.** All of it from
`kv_cache_fp8` and the batching retune it unlocks.

## The rejected step, and why it matters

The walk's quality gate caught weight quantization:

```
quality gate: math_500 0.5900 -> 0.3100 (-0.2800) exceeds allow_loss 10.0%
```

Three of its four variants landed at 0.31 and one at 0.47, against 0.59 for
every lossless configuration. It was offering a further 9% of throughput for 28
accuracy points, and the gate refused it. Without that gate this run would have
reported 5487.3 tok/s and a model that answers roughly half as many questions
correctly.

## The prediction, and what it got wrong

Before the run: the gain should be much smaller than the 14B's 4.9x and the
MoE's 23.8x, because the 1.7B is 4.1GB of weights with `kv_cache_util` at 0.184,
so neither memory nor weight read bandwidth is close to binding. Possibly
nothing.

The direction held. "Possibly nothing" did not: 2.53x is not nothing.

> What I missed is that the mechanism changes with model size. On the MoE, weight
> quantization did the work because 60GB of weights dominated a 273 GB/s part.
> Here weights are negligible and the KV cache is the dominant consumer, so
> kv_cache_fp8 alone is worth 2.1x by halving KV bytes and letting residency
> reach L=512, while weight quantization adds only a further 9%.
>
> That is the more useful version of the finding: quantize what actually
> dominates the memory traffic for that model, and which one that is flips with
> size.

## Read against the other two models on this host

| model | lossless best | lossy best | gain | accuracy cost |
|---|---|---|---|---|
| 1.7B | 1915.1 | 4846.8 | 2.53x | none, 0.59 to 0.59 |
| 14B | 197.3 | 966.9 | 4.90x | none measured, 0.73 |
| 30B-MoE | 90.3 | 2148.0 | 23.8x | none, 0.74 to 0.74 |

The gain rises with model size and so does the affordability of quantizing
weights. At 30B, nvfp4 held accuracy at 0.74 while buying 16x the concurrency.
At 1.7B the same class of step costs 28 accuracy points. A small model has no
redundancy to spare, which is the other half of why the technique that dominates
one end of this table is useless at the other.

## Follow on

`kv_cache_fp8` is worth 2.1x here and was never reachable by the lossless walk,
which parks the entire lossy branch. Any deployment of a small model on this
hardware that has not tried it is leaving a factor of two on the floor, at no
measured accuracy cost, and it is one flag.
