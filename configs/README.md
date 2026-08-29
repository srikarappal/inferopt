# Serving configs for eval_repro

`eval_repro.py --config configs/<name>.json` serves the model with these flags.

| file | what it is |
|---|---|
| `stock.json` | vLLM defaults. `gpu_memory_utilization: 0.75` is a boot requirement on
  unified-memory parts, not a tuning choice -- 0.90 there leaves ~1.6GB of headroom on a
  122GB box. |
| `fp8.json` | stock plus `quantization: fp8`. FP8 is the one quantization with **no
  artifact**: vLLM quantizes the bf16 weights during model load, so there is no checkpoint
  to point `--model` at. Every other format writes a directory under `artifacts/`. |

An inferopt run's `result.json` also works directly:

    python eval_repro.py --model Qwen/Qwen3-14B --config runs/ninth/result.json
