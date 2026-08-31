# Serving configs for eval_repro

`eval_repro.py --config configs/<name>.json` serves the model with these flags.

| file | what it is |
|---|---|
| `stock.json` | vLLM defaults. `gpu_memory_utilization: 0.75` is a boot requirement on
  unified-memory parts, not a tuning choice -- 0.90 there leaves ~1.6GB of headroom on a
  122GB box. |
| `lossless.json` | the incumbent inferopt's LOSSLESS branch converged on in run nine
  (`runs/ninth/result.json`): prefix caching plus ngram speculative decode. Weights are
  untouched -- these are launch flags only, so quality cannot move by construction and any
  movement is measurement noise. On MATH-500 this was 4.0x goodput at zero accuracy cost
  (11.9 -> 47.7 tok/s, 0.7333 -> 0.7300). Frozen here as a file so a ladder does not depend
  on a run directory surviving. |
| `fp8.json` | stock plus `quantization: fp8`. FP8 is the one quantization with **no
  artifact**: vLLM quantizes the bf16 weights during model load, so there is no checkpoint
  to point `--model` at. Every other format writes a directory under `artifacts/`. |

The ladder is only readable if the lossless row is in it. `stock -> lossless` is what
costs nothing; `lossless -> quantized` is what trades quality for speed. Comparing a
quantized variant against `stock` alone silently credits weight quantization with the
speedup that flags already delivered for free.

An inferopt run's `result.json` also works directly:

    python eval_repro.py --model Qwen/Qwen3-14B --config runs/ninth/result.json
