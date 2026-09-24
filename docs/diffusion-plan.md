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

## Answered by the diffusion runs (Z-Image-Turbo and Wan2.1-T2V-1.3B on the GB10, 24 Sep 2026)

Both walks had the same shape: the server-side lossless branch is flat on
this card, every gain is in the request-side lossy branch.

Z-Image-Turbo, 1024x1024, 8 steps distilled, 13 launches:

| node | result |
|---|---|
| incumbent | 0.14 img/s, PickScore 0.2336 |
| attention_backend, torch_compile, cuda_graph, request_batching | reverted, all within noise; the server serialises samples |
| dit_fp8 | reverted, fp8 is slower than bf16 on the GB10 |
| cache_dit 0.40 | kept, +21% |
| steps 4 | kept, +59%; chosen 0.27 img/s at PickScore 0.2348 |

Wan2.1-T2V-1.3B, 832x480 x 33 frames, 30 steps, cfg 5.0, 14 launches:

| node | result |
|---|---|
| incumbent | 0.45 frames/s, 72 s per clip, PickScore 0.2097 |
| attention_backend | reverted: fa falls back to SDPA on sm121 (bit-identical, +0%); sage_attn 37.6 dB median against a 40 dB bar |
| torch_compile | reverted: 38.0 dB against the bar, and flat on speed |
| cuda_graph, request_batching | reverted, +0% and +2.2% |
| dit_fp8 | reverted at +4.4%, under the 5% band; PickScore 0.2109 |
| cache_dit 0.40 | kept, 1.13 frames/s, +151%; PickScore 0.2028 |
| steps 15 | kept, 1.68 frames/s, +49%; PickScore 0.2028 |
| guidance 3.5 | reverted, +0% (a scale costs the same compute) |

Chosen: 15 steps with Cache-DiT 0.40, 3.7x the seed, a clip in about 20 s.
Two candidates landed at 38 dB: the 40 dB bar may be too strict for a
30-step video sampler, where a reordering of floating point is amplified
through the trajectory; worth measuring against a perceptual metric before
it rejects a real win.

Two measurement faults found on the way, both fixed in the evaluator:

- A restarted runner left its server alive on port 8100; SGLang Diffusion
  moved every later server to another port ("Port 8100 was unavailable,
  using port 8142 instead") and the health check kept passing against the
  orphan. Seven launches measured one server. Now a bound port refuses the
  launch and names the holder, a launch whose log says it moved ports fails,
  and the server runs under PR_SET_PDEATHSIG so a dead runner takes it along
  (`d5b25ee`).
- Frames over the clock window quantised: a 74 s clip in a 297 s window
  completes 3 or 4 times by phase, and one server read 0.34 then 0.44
  frames/s. The rate is now over the span to the last completion (`7aced5a`).

SGLang exposes a request `quality` field that decides whether its own
kernel fusions (the Wan VAE RMSNorm+SiLU among them) may run; the DAG does
not sweep it yet.

## Custom kernels: what the bench said (`vi_kernels`, GB10, 24 Sep 2026)

Four Triton kernels, each compiled and verified against a reference on the
card, benchmarked on an idle GPU (`python -m vi_kernels.bench`):

| kernel | ours | what is already there | verdict |
|---|---|---|---|
| MoE small-M (LLaDA2 shape, M=1/4/8/16) | 0.25 / 0.92 / 1.66 / 2.84 ms | SGLang `fused_experts` 0.24 / 0.87 / 1.65 / 2.77 ms | no win, 1 to 5% slower; outputs agree |
| dLLM block attention, 32-row block | 0.092 ms full block, 0.023 ms at 2 active rows | SDPA 0.034 ms | slower unless under 8 rows are still active |
| DiT modulated LayerNorm and gated residual (Wan 480p) | 0.92 / 1.37 ms | eager 8.2 / 8.7 ms; SGLang has a CUDA fusion for widths that are multiples of 256 up to 8192 | upstream already; ours takes only the shapes theirs refuses |
| Wan VAE channel RMSNorm+SiLU (decoder mid stage) | 5.0 ms | eager 21.7 ms; SGLang has a Triton fusion gated on request `quality` and channels_last_3d | 4.3x on the op, under 1% of a clip; ours takes the eager path |

The adapter (`vi_kernels.sglang_adapter`) wraps `fused_experts`,
`_NormScaleShift.forward_native`, `FusedWanRMSNormSiLU`'s off path and
`WanRMS_norm`; its self-check drives each through SGLang's own entry point
and confirms ours ran. An SGLang server runs its kernels in a spawned
process, so the wrap is applied by an import hook that a `.pth` line arms
when `VI_KERNELS=1` is in the environment (`vi_kernels.autoload`), which is
also how a DAG node would switch it on per launch. Verified on live servers
(`bench/pickup.py`): a LLaDA2.0-mini server answered a chat request with
`moe_small_m` taking its first call in the scheduler process, and a Wan2.1
server decoded a clip with `channel_rmsnorm_silu` taking its first call in
its scheduler; the DiT wrap stayed idle there, as it should at width 1536
where SGLang's own CUDA fusion runs. Picked up, and by the bench, not worth
switching on for these models: the MoE and attention kernels are slower
than what SGLang ships and the two fusions are already upstream.

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
