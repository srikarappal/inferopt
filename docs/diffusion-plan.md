# Diffusion support: dLLMs and video, on SGLang

Decided 23 Sep 2026. Everything named here was read from SGLang `main` on that
day, not remembered; flag names are quoted as the source spells them.

## The decision

```
engine        SGLang for both. dLLMs on its LLM path (LLaDA 2, SDAR, DiffusionGemma
              are day 0), image and video on SGLang Diffusion (python/sglang/multimodal_gen).
              vLLM-Omni is bookmarked, not chosen: coverage is real, its flag surface
              was not verifiable.
DAGs          two files, not three. dag/llm.json stays one spine for autoregressive
              and diffusion decoding, dLLM nodes gated by a predicate the way the LoRA
              nodes are. dag/diffusion.json is new and holds image and video together;
              frames is an axis that is 1 for an image.
order         engine seam -> dLLM -> image -> video
```

## Why two shapes and not one

A dLLM is an LLM to everything inferopt measures: text in, tokens out, TTFT
and an inter token latency (per block), goodput in tokens a second, MATH-500
unchanged. What differs is the decode loop and its knobs. About 80% reuse.

Video has no tokens and no KV cache. The unit of work is one sample at
(height, width, frames, steps). The SLO is p99 seconds per sample, the
throughput is samples an hour a GPU, and there is no exact match to score.
About 20% reuse: the walk, the journal, the budget, the frontier maths.

## The engine seam

Today the evaluator is welded to vLLM in six places: `vllm_cmd()`,
`installed_flags()` (`vllm serve --help=all`), `to_cli()`, `_vllm_version()`,
the `/metrics` names (`vllm:kv_cache_usage_perc`, `vllm:num_preemptions`),
and `hardware_defaults()` (`moe_backend`). Those become one object:

```
Engine
  command()                argv prefix, resolved the way vllm_cmd() is today
  serve_argv(model, host, port, config)
  installed_flags()        from the engine's own --help
  version()
  translate(config)        the DAG's vocabulary -> this engine's flags
  metrics(text)            this engine's /metrics -> kv_cache_util, preemptions,
                           prefix hit rate, spec acceptance
  health_path, api         /health and how a request is shaped
  defaults(fp)             what this engine needs to run at all on this card
```

The DAG's vocabulary does not change. `max_model_len`, `enable_prefix_caching`
and the rest stay the canonical names because every node's rationale was
measured in them; the SGLang engine translates.

```
DAG key                    vLLM                          SGLang (srt)
max_model_len              --max-model-len               --context-length
enable_prefix_caching      --enable-prefix-caching       default on; False -> --disable-radix-cache
max_num_seqs               --max-num-seqs                --max-running-requests
max_num_batched_tokens     --max-num-batched-tokens      --chunked-prefill-size (chunked) / --max-prefill-tokens
enable_chunked_prefill     --enable-chunked-prefill      default on; False -> --chunked-prefill-size -1
block_size                 --block-size                  --page-size
kv_cache_dtype fp8         --kv-cache-dtype fp8          --kv-cache-dtype fp8_e4m3
quantization               --quantization                --quantization            [verify names per format]
enforce_eager              --enforce-eager               --disable-cuda-graph
gpu_memory_utilization     --gpu-memory-utilization      --mem-fraction-static
tensor_parallel_size       --tensor-parallel-size        --tp-size
speculative_config ngram   --speculative-config JSON     --speculative-algorithm + --speculative-num-draft-tokens
                                                         + --speculative-ngram-*   [verify the algorithm name]
metrics
  kv_cache_util            vllm:kv_cache_usage_perc      sglang:token_usage
  preemptions              vllm:num_preemptions_total    sglang:num_retracted_requests_total
  prefix hit rate          vllm:prefix_cache_hits/queries  sglang:cache_hit_rate
  spec acceptance          vllm:spec_decode_num_accepted_tokens / draft   sglang:spec_accept_rate
```

Which engine runs is decided once, from the fingerprint: `decoding ==
"diffusion"` or a diffusion architecture selects SGLang; otherwise vLLM,
unless `INFEROPT_ENGINE` says otherwise. A method comparison across engines
is a different study and is not this.

## dLLM: what joins llm.json

New fingerprint field `model.decoding`, `"autoregressive"` or `"diffusion"`,
read from the architecture in `config.json` (`LLaDA2MoeModelLM`,
`LLaDAModelLM`, `SDARForCausalLM`, `DiffusionGemma*`). New config keys, in the
DAG's vocabulary, that only the SGLang engine knows how to emit:

```
dllm_algorithm     LowConfidence | JointThreshold        --dllm-algorithm
dllm_block_size    tokens denoised together               --dllm-algorithm-config (yaml we write)
dllm_threshold     confidence to commit a token, 0.95     same yaml
dllm_fdfo          first done first out scheduling        --dllm-fdfo / --no-dllm-fdfo
```

Nodes, all `applicable_when: fingerprint.model.decoding == "diffusion"`, placed
after `graph_capture` and before `lossless_complete`:

```
dllm_threshold      lossy    sweep threshold 0.95 -> 0.9 -> 0.8. Fewer denoising passes per
                             block, more tokens committed at once; quality can move, so
                             math_500 is scored.
dllm_block_size     lossy    sweep the block. Larger blocks amortise the forward pass and
                             change what the model conditions on.
dllm_algorithm      lossy    JointThreshold against LowConfidence, same budget.
dllm_fdfo           lossless scheduling only; equivalence probe.
```

Which LLM nodes still apply to a dLLM is decided by measurement, not by
assumption: prefix caching, KV fp8, weight quantization and the batching
retunes have no reason not to. Speculative decoding does not apply and its
predicate says so.

The load driver is unchanged: the OpenAI completions stream reports tokens as
they are committed, so TTFT and inter token times are what a client sees.

## Video: dag/diffusion.json

Fingerprint: DiT parameters and dtype, text encoder and VAE sizes, the
resolution, frames and steps the workload asks for, read from the trace.
Workload: a prompt set with (height, width, frames, steps, guidance) per row,
arrival rate. SLO: p99 seconds per sample, and a quality budget.

Server knobs, from `multimodal_gen/runtime/server_args/server_args.py`:

```
attention_backend, enable_attention_backend_autotune
cache_dit_config
tp_size, sp_degree (ulysses_degree, ring_degree), enable_cfg_parallel, cfg_parallel_degree
quantization, component_quantizations, nunchaku_config, kv_cache_quant_config
dit_layerwise_offload, dit_cpu_offload, text_encoder_cpu_offload, vae_cpu_offload
enable_torch_compile, regional_compile, enable_breakable_cuda_graph
batching_mode, batching_max_size, batching_delay_ms
performance_mode, warmup_resolutions, warmup_num_frames, warmup_steps
```

Request knobs, from `configs/sample/sampling_params.py`:

```
num_inference_steps, guidance_scale, height, width, num_frames, fps, seed,
enable_teacache, enable_cache_dit, enable_spectrum, flow_shift, cfg_gate_step
```

API: `POST /v1/images/generations`; `POST /v1/videos` then `GET /v1/videos/{id}`
and `/content`, a job that is polled. `/health`; `/metrics` carries
`sglang:diffusion_request_latency_seconds`, `_num_running_reqs`,
`_generation_batch_size`, `_queue_time_seconds`.

Nodes, first cut, greedy walk as today:

```
attention_backend       lossless  sweep the backends the card has
torch_compile           lossless
cuda_graph              lossless  enable_breakable_cuda_graph
request_batching        lossless  batching_mode with max_size and delay swept
cfg_parallel            lossless  two cards or more
sequence_parallel       lossless  ulysses then ring, two cards or more
layerwise_offload       lossless  only when the DiT does not fit resident
lossless_complete       checkpoint, quality baseline measured here
weight_quantization     lossy     fp8 -> nvfp4 -> nunchaku, per component
cache_dit               lossy     Cache-DiT and TeaCache are approximate whatever the blog says;
                                  swept on their thresholds
steps                   lossy     num_inference_steps down from the customer's, the largest lever
guidance                lossy     guidance_scale, cfg_gate_step
frontier
```

Two quality probes, because the two classes of node need different questions:

```
equivalence (lossless)  same prompt, same seed -> same output. PSNR and LPIPS
                        against the baseline sample; a kernel or a compile that
                        moves a frame more than fp noise is not lossless.
quality (lossy)         LPIPS to the baseline sample on the customer's prompts,
                        and a preference model (ImageReward or HPSv2 for images,
                        VBench dimensions for video) so "different" and "worse"
                        are told apart. Both reported; the customer's budget
                        gates on the preference score.
```

Goodput is samples a second that met the latency target; the frontier axes
are seconds per sample against samples an hour a GPU. Stage 1.2 does not
exist: AIConfigurator is LLM only, so the plot is measured points alone.

## Answered by the first run (LLaDA2.0-mini on the GB10, 23 Sep 2026)

- SGLang 0.5.20 installs on aarch64 sm121 from wheels: it pins torch 2.13.0
  and flashinfer 0.6.18, both already in the image, and sgl-kernel ships an
  aarch64 wheel. Two traps: `cuda-tile` is an NVIDIA wheel stub that fetches
  from pypi.nvidia.com and fails its own hash check on a bad link (install it
  from that index directly), and the install drags `nvidia-cuda-runtime` back
  to 13.0 against nvcc 13.4, which breaks flashinfer's JIT the same way it
  once broke vLLM's; re-pin 13.4 after.
- A dLLM streams one BLOCK per chunk. The driver counted chunks as tokens and
  read a 40 tokens/s server as 1.3. Fixed by asking for
  `stream_options.continuous_usage_stats` and crediting each chunk with the
  tokens it carried (goodput `c361167`); exact for autoregressive streams too.
- Inter token latency means what it did: the driver's per request ITL is
  (latency minus TTFT) over completion tokens, a mean per token, so a block
  arriving at once does not distort it.
- CUDA graph capture overflows flashinfer's workspace on a dLLM at SGLang's
  default capture batch sizes: a dLLM decode step is a prefill-shaped batch
  of requests x block tokens. SGLang's own tests cap capture at
  `--cuda-graph-bs-decode 1 2 3 4`. The engine's diffusion defaults need a
  cap (`cuda_graph_max_bs_decode`), and graph_capture is then a real node.
- Every translated flag was in `sglang serve --help` (538 flags), the server
  accepted them all, and a launch that dies is recorded as goodput 0 and the
  walk moves on, as with vLLM.

## What has to be verified on a machine before it is believed

- SGLang wheels on the DGX (aarch64, sm121). The target is x86 H100 and
  B200 where `pip install sglang[all]` is routine; the test rig may not be.
- The `--quantization` names SGLang accepts for a modelopt fp8 or nvfp4
  checkpoint, and whether it reads `hf_quant_config.json` unprompted.
- The ngram speculative algorithm's name and the flags it takes.
- Whether the completions stream from a dLLM reports per block or per token,
  which decides what inter token latency means for it.
- The metric names above, against a running server's `/metrics`.

## Order of work

1. Engine seam in `evaluator.py`: `VllmEngine` carrying today's behaviour
   unchanged, `SglangEngine` for the LLM path with the table above, selection
   from the fingerprint. Unit tests on translation and metrics parsing; no
   GPU.
2. `model.decoding` in the fingerprint; dLLM nodes in `dag/llm.json`;
   `validate_dag` passes.
3. A dLLM run on the DGX against LLaDA2-mini, then the first datacenter card.
4. `dag/diffusion.json`, the diffusion fingerprint, the sample driver and the
   two probes. Image first (FLUX, Qwen-Image), video on the same code with
   frames as an axis (Wan 2.2).

## The profile every trial carries (for stage 3)

Every served trial, LLM or diffusion, ends with a torch profiler window at its
operating point: `--profiler-config` on vLLM, `/start_profile` on SGLang,
`profile: true` on one SGLang Diffusion request. All three write a Chrome
trace, and `inferopt/profile.py` reduces it to what a reader can act on:

```
diagnostics.profile
  total_gpu_ms, wall_ms, gpu_busy      how much of the span the GPU was executing anything
  families                             attention, gemm, moe, conv, norm, rope_embed, sampling,
                                       memory, elementwise, allreduce, other: ms and percent
  top_ops                              the 25 heaviest kernels by name, family, ms, percent, calls
  traces                               the files, beside the launch log, for the whole picture
```

It rides in the journal and in result.json, so stage 3 gets, for each point on
the frontier, not only what the configuration was but what the step was made
of on that card. A kernel effort aimed at an op that is 40% of a step on the
customer's hardware is a product; one aimed at what looked slow on the test
rig is a hobby, and this is how the two are told apart.
