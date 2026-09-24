"""Image and video diffusion on SGLang Diffusion: the 20% that is new.

The walk, the journal, the budget guard, resume and the frontier maths are the
LLM machinery unchanged. What a diffusion pipeline needs of its own is here:

  fingerprint   three components (text encoder, denoiser, VAE) read from the
                pipeline's model_index.json, and the sample the workload asks for
  driver        closed loop over prompts against /v1/images/generations or
                /v1/videos, timing each sample
  probes        equivalence: same prompt, same seed, PSNR against the baseline
                sample; quality: a preference model over the customer's prompts
  evaluator     launches the server through the engine, measures, tears down

FRAMES ARE THE TOKEN. An image is one frame; a clip is `num_frames`. Goodput
is frames a second whose sample met the latency target, the per user rate is
frames a second for one request, and the SLO is p99 seconds per sample. That
keeps Trial, the frontier and the plot axes as they are: per user rate against
per GPU throughput, in the unit the customer is buying.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import math
import statistics
import threading
import time
from pathlib import Path
from typing import Any

import httpx

from inferopt.engines import SglangDiffusionEngine
from inferopt.evaluator import HOST, VllmEvaluator, WARMUP_S
from inferopt.fingerprint import (SLO, Context, DiffusionShape, Fingerprint,
                                  LoraFingerprint, ModelFingerprint,
                                  WorkloadFingerprint)
from inferopt.traverse import Trial

SWEEP_LEVELS = (1, 2, 4, 8)
WINDOW_S = 120.0            # the floor; a window has to hold several samples, see measure()
SAMPLES_PER_WINDOW = 4      # at L=1. A 74 s clip in a 120 s window completed once per level and
                            # every level read the same number, which cannot be true
PROBE_SAMPLES = 8           # fixed seed prompts for equivalence and quality, images
PROBE_SAMPLES_VIDEO = 4     # clips take minutes each
EQUIVALENT_PSNR_DB = 40.0   # above this two renders differ by float noise, not by a kernel
POLL_S = 1.0

VIDEO_TRANSFORMERS = ("WanTransformer3DModel", "HunyuanVideoTransformer3DModel",
                      "LTXVideoTransformer3DModel", "CosmosTransformer3DModel",
                      "SanaVideoTransformer3DModel")


# ---------------------------------------------------------------- fingerprint

def _params_b(model: str, repo_files: list[dict], prefix: str) -> float:
    """Billions of parameters under one component, counted from each shard's
    safetensors header, which the hub serves without the shard. Exact, and
    indifferent to whether the checkpoint stores fp32 or bf16, which a size
    estimate is not: Wan's umT5 is 22.7 GB on disk and 5.7B parameters."""
    from huggingface_hub import parse_safetensors_file_metadata
    total = 0
    for f in repo_files:
        if f["path"].startswith(prefix + "/") and f["path"].endswith(".safetensors"):
            try:
                total += sum(parse_safetensors_file_metadata(model, f["path"]).parameter_count.values())
            except Exception:
                total += (f.get("size") or 0) // 2      # the estimate, when the header is unreadable
    return round(total / 1e9, 2)


def pipeline_shape(model: str, workload_row: dict) -> DiffusionShape | None:
    """What the pipeline is made of, or None when `model` is not one.

    Reads model_index.json and the component configs from the hub, never a
    shard. The sample size, frames, steps and guidance come from the workload:
    they are what the customer runs at, and the search varies them from there.
    """
    from huggingface_hub import HfApi, hf_hub_download
    try:
        index = json.load(open(hf_hub_download(model, "model_index.json")))
    except Exception:
        return None
    # Every component the index lists, by its directory name. The shape is
    # universal (encoders, denoiser, VAE); the count of each is not: FLUX has
    # two text encoders, SD3 three, Wan2.2-A14B two denoisers.
    components = {name: spec[1] for name, spec in index.items()
                  if isinstance(spec, list) and len(spec) == 2 and spec[1]
                  and name not in ("scheduler", "tokenizer") and not name.startswith("tokenizer")}
    encoders = [n for n in components if n.startswith("text_encoder")]
    denoisers = [n for n in components if n.startswith("transformer") or n.startswith("unet")]
    transformer_class = components.get("transformer") or (components[denoisers[0]] if denoisers else "")
    text_encoder = components.get("text_encoder") or (components[encoders[0]] if encoders else "")
    vae = components.get("vae", "")
    files = [{"path": s.path, "size": getattr(s, "size", 0)}
             for s in HfApi().list_repo_tree(model, recursive=True)]   # folders have no size
    total_bytes = sum((f["size"] or 0) for f in files if f["path"].endswith(".safetensors"))

    kind = "video" if transformer_class in VIDEO_TRANSFORMERS or "3D" in transformer_class else "image"
    guidance = float(workload_row.get("guidance_scale", 1.0))
    return DiffusionShape(
        kind=kind, transformer_class=transformer_class,
        transformer_params_b=round(sum(_params_b(model, files, n) for n in denoisers), 2),
        text_encoder_arch=text_encoder,
        text_encoder_params_b=round(sum(_params_b(model, files, n) for n in encoders), 2),
        vae_class=vae, weights_gb=round(total_bytes / 1e9, 1),
        components=components, n_text_encoders=len(encoders), n_denoisers=len(denoisers),
        width=int(workload_row["width"]), height=int(workload_row["height"]),
        frames=int(workload_row.get("num_frames", 1)) if kind == "video" else 1,
        fps=int(workload_row.get("fps", 24)),
        steps=int(workload_row["num_inference_steps"]),
        guidance_scale=guidance, distilled=guidance <= 1.0,
    )


def denoiser_fingerprint(model: str, shape: DiffusionShape) -> ModelFingerprint:
    """The denoiser in the LLM fingerprint's terms, so the walk's machinery
    (calibration store, hardware defaults, predicates) has a model to key on.
    The fields that mean nothing for a denoiser are filled with what keeps the
    schema valid and are not read by dag/diffusion.json."""
    from huggingface_hub import hf_hub_download
    cfg = json.load(open(hf_hub_download(model, "transformer/config.json")))
    layers = int(cfg.get("num_layers") or cfg.get("n_layers") or cfg.get("depth") or 1)
    heads = int(cfg.get("num_attention_heads") or cfg.get("num_heads") or cfg.get("n_heads") or 1)
    head_dim = int(cfg.get("attention_head_dim") or cfg.get("head_dim") or 0)
    hidden = int(cfg.get("dim") or cfg.get("hidden_size") or cfg.get("inner_dim") or (heads * head_dim) or 1)
    return ModelFingerprint(
        id=model, architecture=shape.transformer_class, decoding="denoising",
        is_dense=True, n_params_b=shape.transformer_params_b or 0.01,
        n_layers=layers, hidden_size=hidden, n_heads=heads, n_kv_heads=heads,
        attention_type="mha", native_dtype=str(cfg.get("torch_dtype") or "bfloat16"),
        bytes_per_param=2.0, max_model_len=shape.width * shape.height // 256 * max(1, shape.frames),
    )


def workload_fingerprint(rows: list[dict], qps: float, shape: DiffusionShape,
                         trace_ref: str) -> WorkloadFingerprint:
    """The prompt set as a workload. Output tokens are frames."""
    lengths = sorted(max(1, len(r.get("prompt", "")) // 4) for r in rows) or [1]
    pick = lambda q: lengths[min(len(lengths) - 1, int(q * (len(lengths) - 1)))]
    return WorkloadFingerprint(
        n_requests=len(rows), mean_input_tokens=float(statistics.mean(lengths)),
        p99_input_tokens=pick(0.99), p999_input_tokens=pick(0.999),
        mean_output_tokens=float(shape.frames), p99_output_tokens=shape.frames,
        request_rate_qps=qps, max_concurrency=max(1, int(math.ceil(qps * 4))),
        burstiness=1.0, prefix_overlap=0.0, prefix_overlap_per_adapter=0.0,
        multi_turn=False, greedy=True, temperature=0.0, top_p=1.0,
        structured_generation=0.0, trace_ref=trace_ref,
    )


# ------------------------------------------------------------------- driver

class Sample:
    """One generated sample and how long it took."""

    def __init__(self, prompt: str, seed: int):
        self.prompt, self.seed = prompt, seed
        self.start = time.perf_counter()
        self.done: float | None = None
        self.frames = 1
        self.error = ""
        self.image_png: bytes | None = None
        self.video_id: str | None = None

    @property
    def latency(self) -> float:
        return (self.done or time.perf_counter()) - self.start

    @property
    def ok(self) -> bool:
        return self.done is not None and not self.error


def request_body(shape: DiffusionShape, config: dict, prompt: str, seed: int) -> dict:
    """The per request half of a config, over the workload's defaults."""
    body = {"prompt": prompt, "n": 1, "seed": seed,
            **{k: config[k] for k in ("profile", "num_profiled_timesteps") if k in config},
            "width": int(config.get("width", shape.width)),
            "height": int(config.get("height", shape.height)),
            "num_inference_steps": int(config.get("num_inference_steps", shape.steps)),
            "guidance_scale": float(config.get("guidance_scale", shape.guidance_scale))}
    for key in ("enable_teacache", "flow_shift", "negative_prompt", "true_cfg_scale",
                "enable_cache_dit", "cache_dit_params"):
        if key in config:
            body[key] = config[key]
    if shape.kind == "video":
        body["num_frames"] = int(config.get("num_frames", shape.frames))
        body["fps"] = int(config.get("fps", shape.fps))
    else:
        body["response_format"] = "b64_json"
    return body


async def _one(client: httpx.AsyncClient, base_url: str, shape: DiffusionShape,
               config: dict, prompt: str, seed: int, want_content: bool) -> Sample:
    s = Sample(prompt, seed)
    body = request_body(shape, config, prompt, seed)
    try:
        if shape.kind == "image":
            r = await client.post(f"{base_url}/v1/images/generations", json=body, timeout=3600)
            if r.status_code != 200:
                s.error = f"HTTP {r.status_code}: {r.text[:200]}"
                return s
            s.done = time.perf_counter()
            data = (r.json().get("data") or [{}])[0]
            if want_content and data.get("b64_json"):
                s.image_png = base64.b64decode(data["b64_json"])
            return s
        # Video is a job: create, poll to completion, fetch content on request.
        r = await client.post(f"{base_url}/v1/videos", json=body, timeout=600)
        if r.status_code not in (200, 201, 202):
            s.error = f"HTTP {r.status_code}: {r.text[:200]}"
            return s
        s.video_id = r.json().get("id")
        s.frames = body["num_frames"]
        while True:
            await asyncio.sleep(POLL_S)
            j = (await client.get(f"{base_url}/v1/videos/{s.video_id}", timeout=60)).json()
            status = str(j.get("status", "")).lower()
            if status in ("completed", "succeeded", "done"):
                s.done = time.perf_counter()
                break
            if status in ("failed", "error", "cancelled"):
                s.error = f"job {status}: {str(j.get('error') or '')[:200]}"
                return s
        if want_content:
            c = await client.get(f"{base_url}/v1/videos/{s.video_id}/content", timeout=600)
            if c.status_code == 200:
                s.image_png = first_frame_png(c.content)
        return s
    except Exception as e:
        s.error = f"{type(e).__name__}: {str(e)[:200]}"
        return s


async def closed_loop(base_url: str, shape: DiffusionShape, config: dict, prompts: list[str],
                      concurrency: int, window_s: float, seed_base: int = 1000,
                      latency_target_ms: float | None = None) -> dict:
    """`concurrency` workers, each taking the next prompt, for `window_s`.

    Measured over samples that COMPLETE inside the window, so a slow sample
    started near the end does not count for or against; the window opens on
    a warm server and closes on the clock.
    """
    cursor = [0]
    finished: list[Sample] = []
    opened = time.perf_counter()
    stop = opened + window_s

    async def worker(client):
        while time.perf_counter() < stop:
            i = cursor[0]
            cursor[0] += 1
            s = await _one(client, base_url, shape, config, prompts[i % len(prompts)],
                           seed_base + i, want_content=False)
            if s.done is not None and s.done <= stop:
                finished.append(s)

    async with httpx.AsyncClient() as client:
        await asyncio.gather(*(worker(client) for _ in range(concurrency)))
    return summarise(finished, span_of(finished, opened, window_s), latency_target_ms)


def span_of(finished: list[Sample], opened: float, window_s: float) -> float:
    """The time the completed work took: window open to the last completion.

    Frames over the clock window quantises: a 74 s clip in a 297 s window
    completes 3 or 4 times depending on phase, and the same server read 0.34
    then 0.44 frames/s on consecutive launches, a 25% swing against a 5%
    accept band. Over the span to the last completion, 3 clips in 222 s and
    4 in 297 s are the same rate. The window still bounds how long we wait;
    it is no longer the denominator. Nothing completed: the window stands,
    and the rate is zero either way."""
    done = [s.done for s in finished if s.done is not None]
    return max(done) - opened if done else window_s


def summarise(samples: list[Sample], span_s: float,
              latency_target_ms: float | None = None) -> dict:
    """Goodput is frames of samples that met the target, over the span the
    completed work took (see span_of)."""
    ok = [s for s in samples if s.ok]
    lat = sorted(s.latency for s in ok)
    p = lambda q: (lat[min(len(lat) - 1, int(math.ceil(q * len(lat))) - 1)] if lat else float("inf"))
    met = [s for s in ok if latency_target_ms is None or s.latency * 1000 <= latency_target_ms]
    return {
        "completed": len(ok), "failed": len(samples) - len(ok),
        "latency_p50_s": p(0.5), "latency_p99_s": p(0.99),
        "span_s": round(span_s, 1),
        "throughput_frames_s": sum(s.frames for s in ok) / span_s,
        "goodput_frames_s": sum(s.frames for s in met) / span_s,
        "slo_attainment": (len(met) / len(ok)) if ok else 0.0,
        "failure_reasons": {s.error[:60]: 1 for s in samples if not s.ok},
    }


def first_frame_png(video_bytes: bytes) -> bytes | None:
    """The first frame of an mp4 as PNG, through PyAV when it is installed.
    None when it is not: an equivalence probe that cannot see the frame says
    so rather than passing."""
    try:
        import av
        from PIL import Image
    except ImportError:
        return None
    with av.open(io.BytesIO(video_bytes)) as container:
        for frame in container.decode(video=0):
            buf = io.BytesIO()
            frame.to_image().save(buf, format="PNG")
            return buf.getvalue()
    return None


# ------------------------------------------------------------------- probes

def psnr_db(a_png: bytes, b_png: bytes) -> float:
    from PIL import Image
    import numpy as np
    a = np.asarray(Image.open(io.BytesIO(a_png)).convert("RGB"), dtype=np.float64)
    b = np.asarray(Image.open(io.BytesIO(b_png)).convert("RGB"), dtype=np.float64)
    if a.shape != b.shape:
        return 0.0
    mse = float(((a - b) ** 2).mean())
    return float("inf") if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)


class PickScore:
    """A preference model over (prompt, image): what a person would pick.

    yuvalkirstain/PickScore_v1 through transformers alone, no extra package.
    Loaded once, on first use, on the CPU: it scores eight images a node and
    the GPU is busy serving. Absent transformers or the weights, scores are
    None, which the walk treats as unmeasured rather than as zero.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._model = self._processor = None
        self.last_error = ""

    def _load(self):
        if self._model is None:
            from transformers import AutoModel, AutoProcessor
            self._processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
            self._model = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1").eval().to(self.device)

    def score(self, prompts: list[str], pngs: list[bytes]) -> float | None:
        """None on any failure. A scorer that cannot answer must not end a
        run that has measured eight nodes, and None is read as unmeasured,
        which zero would not be."""
        try:
            self._load()
            import torch
            from PIL import Image
            images = [Image.open(io.BytesIO(p)).convert("RGB") for p in pngs]
            with torch.no_grad():
                im = self._processor(images=images, padding=True, truncation=True, max_length=77,
                                     return_tensors="pt").to(self.device)
                tx = self._processor(text=prompts, padding=True, truncation=True, max_length=77,
                                     return_tensors="pt").to(self.device)
                # transformers 5 hands back an output object; 4 handed back
                # the tensor. Take the pooled embedding either way.
                ie = _features(self._model.get_image_features(**im))
                te = _features(self._model.get_text_features(**tx))
                ie = ie / ie.norm(dim=-1, keepdim=True)
                te = te / te.norm(dim=-1, keepdim=True)
                scores = self._model.logit_scale.exp() * (te * ie).sum(dim=-1)
        except Exception as failed:
            self.last_error = f"{type(failed).__name__}: {str(failed)[:160]}"
            return None
        # Mean over the set, on PickScore's own scale (about 15 to 25); the
        # walk compares deltas against the budget, so the scale only has to be
        # the same on both sides.
        return round(float(scores.mean()) / 100.0, 4)


def _features(out):
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    if hasattr(out, "image_embeds") and out.image_embeds is not None:
        return out.image_embeds
    if hasattr(out, "text_embeds") and out.text_embeds is not None:
        return out.text_embeds
    return out


# ----------------------------------------------------------------- evaluator

class DiffusionEvaluator(VllmEvaluator):
    """Launch, measure, tear down, for a diffusers pipeline on SGLang Diffusion.

    Reuses the LLM evaluator's launch path (the progress following deadline,
    the flag check, the log capture) and replaces what it measures.
    """

    def __init__(self, fp: Fingerprint, slo: SLO, prompts: list[str], run_dir: str,
                 gpu: str = "0", port: int = 8100, log=print, engine=None):
        self.fp, self.slo, self.log = fp, slo, log
        self.shape = fp.diffusion
        self.engine = engine or SglangDiffusionEngine(model=fp.model.id)
        self.gpu, self.port, self.run_dir = gpu, port, Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = f"http://{HOST}:{port}"
        self.prompts = prompts
        self.trace_path = None
        self.replay = None
        self.scorer = PickScore()
        self.baseline_dir = self.run_dir / "baseline_samples"

    # what the LLM path calls that a pipeline has no answer for
    def replay_lengths(self):
        return []

    def _probe_prompts(self) -> list[str]:
        n = PROBE_SAMPLES_VIDEO if self.shape.kind == "video" else PROBE_SAMPLES
        return self.prompts[:n]

    def _first_latency(self, config: dict) -> float:
        """One sample, timed, when the warm-up window closed before any
        finished: a clip can take longer than the warm-up itself."""
        async def go():
            async with httpx.AsyncClient() as client:
                return await _one(client, self.base_url, self.shape, config, self.prompts[0], 1, False)
        return asyncio.run(go()).latency

    def _profile_one(self, config: dict, el=lambda: "") -> dict | None:
        """One sample with the engine's profiler on, reduced to where its
        denoising steps spend the GPU. Per request, because that is how
        SGLang Diffusion profiles; never fatal."""
        if not getattr(self, "profile", True):
            return None
        from inferopt.profile import summarise_traces
        fields = self.engine.profile_request_fields(steps=5)
        if not fields:
            return None
        try:
            async def go():
                async with httpx.AsyncClient() as client:
                    return await _one(client, self.base_url, self.shape, {**config, **fields},
                                      self.prompts[0], 3, False)
            asyncio.run(go())
            summary = None
            for _ in range(30):
                summary = summarise_traces(self._profile_dir)
                if summary:
                    break
                time.sleep(2)
            if summary:
                head = ", ".join(f"{f} {v['pct']:.0f}%" for f, v in list(summary["families"].items())[:4])
                self.log(f"        {el()} profile      gpu busy {summary['gpu_busy']}  {head}")
            return summary
        except Exception as failed:
            self.log(f"        {el()} profile      not taken ({type(failed).__name__}: {str(failed)[:80]})")
            return None

    def _render_probe_set(self, config: dict) -> list[Sample]:
        """The fixed seed samples every probe reads, generated at L=1."""
        async def go():
            async with httpx.AsyncClient() as client:
                return [await _one(client, self.base_url, self.shape, config, prompt, 7 + i, True)
                        for i, prompt in enumerate(self._probe_prompts())]
        return asyncio.run(go())

    def _equivalence_of(self, samples: list[Sample], tag: str) -> float | None:
        """Fraction of probe samples that do NOT match the baseline render at
        the same seed within the PSNR bar. None until a baseline exists.

        Every render and every PSNR is kept beside the launch log, because the
        bar is a number that has to be set from renders people have looked at,
        and a probe that keeps only its verdict cannot be argued with."""
        if not self.baseline_dir.exists():
            return None
        keep = self.run_dir / "launches" / tag / "probe"
        keep.mkdir(parents=True, exist_ok=True)
        psnrs: dict[str, float] = {}
        for i, s in enumerate(samples):
            ref = self.baseline_dir / f"{i}.png"
            if not s.image_png or not ref.exists():
                continue
            (keep / f"{i}.png").write_bytes(s.image_png)
            psnrs[str(i)] = round(psnr_db(ref.read_bytes(), s.image_png), 2)
        (keep / "psnr.json").write_text(json.dumps(psnrs, indent=1))
        if not psnrs:
            return None
        values = sorted(psnrs.values())
        self.log(f"        psnr vs baseline  min {values[0]:.1f}  median "
                 f"{values[len(values) // 2]:.1f}  max {values[-1]:.1f} dB  (bar {EQUIVALENT_PSNR_DB:.0f})")
        return sum(1 for v in values if v < EQUIVALENT_PSNR_DB) / len(values)

    def _keep_baseline(self, samples: list[Sample]) -> None:
        self.baseline_dir.mkdir(parents=True, exist_ok=True)
        for i, s in enumerate(samples):
            if s.image_png:
                (self.baseline_dir / f"{i}.png").write_bytes(s.image_png)

    def measure(self, config: dict[str, Any], *, probes: list[str], benchmarks: list[str],
                node_id: str, concurrency: int | None = None, levels=None,
                fixed_concurrency: int | None = None) -> Trial:
        replay = getattr(self, "replay", None)
        if replay is not None:
            from inferopt.resume import key, to_trial
            hit = replay.get(key(node_id, config))
            if hit is not None:
                t = to_trial(hit)
                t.diagnostics = {**(t.diagnostics or {}), "replayed": True}
                self.log(f"        replayed from journal, no launch spent ({t.goodput:.1f} frames/s)")
                return t

        tag = self._launch_tag(node_id, config)
        t_start = time.time()
        el = lambda: f"+{(time.time()-t_start)/60:4.1f}m"
        served = {k: v for k, v in config.items() if k != "model"}
        self.log(f"        {el()} launching  {json.dumps(served, default=str)[:100]}")
        target_ms = self.slo.ttft_p99_ms       # p99 seconds per sample, in ms
        from inferopt.evaluator import LaunchError
        try:
            with self._serve(config, tag) as model:
                self.log(f"        {el()} healthy, warming up")
                # Warm-up renders at least one sample and so knows how long one
                # takes; the measurement window is sized from that, so every
                # level completes several samples rather than one.
                warm = asyncio.run(closed_loop(self.base_url, self.shape, config, self.prompts, 1,
                                               min(WARMUP_S, 60.0)))
                one_sample_s = warm["latency_p50_s"] if warm["completed"] else self._first_latency(config)
                window_s = max(WINDOW_S, SAMPLES_PER_WINDOW * one_sample_s)
                self.log(f"        {el()} one sample {one_sample_s:.1f}s, window {window_s:.0f}s")
                pts = []
                for level in (levels or SWEEP_LEVELS):
                    med = asyncio.run(closed_loop(self.base_url, self.shape, config,
                                                  self.prompts, level, window_s,
                                                  latency_target_ms=target_ms))
                    med["concurrency"] = level
                    pts.append(med)
                    self.log(f"        {el()} L={level:<3d} goodput {med['goodput_frames_s']:7.2f} frames/s  "
                             f"p99 {med['latency_p99_s']:6.1f}s  slo {med['slo_attainment']:.0%}  "
                             f"({med['completed']} done)")
                    # Past the peak: more load cannot help. A server that does
                    # not batch samples serialises them, so latency grows with
                    # concurrency and goodput, once it falls, does not come
                    # back; the next level is fifteen minutes for the same
                    # answer.
                    if len(pts) >= 2 and med["goodput_frames_s"] < pts[-2]["goodput_frames_s"]:
                        break
                peak = max(pts, key=lambda m: m["goodput_frames_s"])

                samples = self._render_probe_set(config)
                if node_id == "incumbent" or not self.baseline_dir.exists():
                    self._keep_baseline(samples)
                div = self._equivalence_of(samples, tag) if "equivalence" in probes else None
                if div is not None:
                    self.log(f"        {el()} equivalence  {div:.0%} of fixed seed renders differ from the baseline")
                qual: dict = {}
                if "quality" in probes and benchmarks:
                    pngs = [s.image_png for s in samples if s.image_png]
                    prompts = [s.prompt for s in samples if s.image_png]
                    score = self.scorer.score(prompts, pngs) if pngs else None
                    qual = {b: score for b in benchmarks}
                    self.log(f"        {el()} quality      pickscore {score}"
                             + (f"  (unscored: {self.scorer.last_error})" if score is None else ""))
                profile = self._profile_one(config, el)
                mem = self._gpu_memory_gb()
                self.log(f"        {el()} done, tearing down")
        except LaunchError as e:
            self.log(f"        launch failed: {e}")
            wd = self.run_dir / "launches" / tag
            wd.mkdir(parents=True, exist_ok=True)
            (wd / "why.txt").write_text(f"{e}\n\n{getattr(e, 'stderr', '')}")
            return Trial(node_id=node_id, config=dict(config), goodput=0.0,
                         ttft_p99_ms=float("inf"), itl_p99_ms=float("inf"), memory_gb=0.0,
                         slo_ok=False, diagnostics={"launch_error": str(e),
                                                    "stderr_tail": (getattr(e, "stderr", "") or "")[-1200:]})

        frames = max(1, int(config.get("num_frames", self.shape.frames)))
        return Trial(
            node_id=node_id, config=dict(config),
            goodput=round(peak["goodput_frames_s"], 2),
            ttft_p99_ms=round(peak["latency_p99_s"] * 1000, 1),
            itl_p99_ms=round(peak["latency_p99_s"] * 1000 / frames, 2),
            memory_gb=mem, quality=qual, equivalence_divergence=div,
            concurrency=peak["concurrency"],
            curve=[{"concurrency": m["concurrency"],
                    "goodput": round(m["goodput_frames_s"], 2),
                    "throughput": round(m["throughput_frames_s"], 2),
                    "ttft_p99_ms": round(m["latency_p99_s"] * 1000, 1),
                    "itl_p99_ms": round(m["latency_p99_s"] * 1000 / frames, 2),
                    "slo_attainment": round(m["slo_attainment"], 3),
                    "completed": m["completed"], "failed": m["failed"]} for m in pts],
            diagnostics={"slo_attainment": round(peak["slo_attainment"], 3),
                         "throughput": round(peak["throughput_frames_s"], 2),
                         "completed": peak["completed"], "failed": peak["failed"],
                         "latency_p50_s": round(peak["latency_p50_s"], 2),
                         "failure_reasons": peak["failure_reasons"],
                         "unit": "frames", **({"profile": profile} if profile else {})},
            slo_ok=peak["goodput_frames_s"] > 0,
        )


# --------------------------------------------------------------------- entry

def is_pipeline(model: str) -> bool:
    """A diffusers pipeline has a model_index.json at its root; a language
    model has a config.json there instead."""
    from huggingface_hub import hf_hub_download
    try:
        hf_hub_download(model, "model_index.json")
        return True
    except Exception:
        return False


def build_context(model: str, rows: list[dict], slo: SLO, qps: float, trace_ref: str,
                  hw) -> tuple[Fingerprint, Context]:
    shape = pipeline_shape(model, rows[0])
    if shape is None:
        raise ValueError(f"{model} has no model_index.json, so it is not a diffusers pipeline")
    fp = Fingerprint(model=denoiser_fingerprint(model, shape), hw=hw,
                     workload=workload_fingerprint(rows, qps, shape, trace_ref),
                     lora=LoraFingerprint(), diffusion=shape)
    return fp, Context(fingerprint=fp, slo=slo)


def seed_config(shape: DiffusionShape) -> dict:
    """The customer's own sample, on the server's defaults. Every node moves
    one thing from here."""
    cfg = {"num_inference_steps": shape.steps, "guidance_scale": shape.guidance_scale,
           "width": shape.width, "height": shape.height}
    if shape.kind == "video":
        cfg.update({"num_frames": shape.frames, "fps": shape.fps})
    return cfg


def optimize_pipeline(*, model: str, rows: list[dict], latency_p99_ms: float,
                      qps: float = 1.0, allow_loss: float | None = None,
                      run_dir: str | None = None, gpu: str = "0", port: int = 8100,
                      max_launches: int | None = None, max_minutes: float | None = None,
                      dag: str | None = None, profile: bool = True, log=print):
    """Search a diffusers pipeline's serving configurations. The diffusion
    twin of api.optimize, and what it delegates to for a model with a
    model_index.json.

    `rows` is the workload: prompt, width, height, num_inference_steps,
    guidance_scale, and num_frames and fps for video. `latency_p99_ms` is the
    SLO: p99 seconds per sample, in ms.
    """
    from inferopt import resume
    from inferopt._paths import default_dag, runs as _runs
    from inferopt.api import Result
    from inferopt.provenance import trial_stamp
    from inferopt.request import InferOptRequest, detect_hardware
    from inferopt.traverse import traverse

    rd = Path(run_dir or _runs(f"{model.split('/')[-1].lower()}-diffusion"))
    rd.mkdir(parents=True, exist_ok=True)
    trace = rd / "trace.jsonl"
    trace.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    slo = SLO(ttft_p99_ms=latency_p99_ms, itl_p99_ms=None, quality_budget=allow_loss)
    hw = detect_hardware(InferOptRequest(model=model, trace=str(trace), ttft_p99_ms=latency_p99_ms))
    fp, ctx = build_context(model, rows, slo, qps, str(trace), hw)
    shape = fp.diffusion
    log(f"pipeline    {shape.kind}: {shape.transformer_class} {shape.transformer_params_b}B denoiser, "
        f"{shape.text_encoder_arch} {shape.text_encoder_params_b}B encoder, {shape.weights_gb} GB resident")
    log(f"sample      {shape.width}x{shape.height}"
        + (f" x {shape.frames} frames at {shape.fps} fps" if shape.kind == "video" else "")
        + f", {shape.steps} steps, guidance {shape.guidance_scale}"
        + (" (distilled)" if shape.distilled else ""))

    stamp = trial_stamp(fp, str(trace), slo)
    plan = resume.plan(rd, stamp)
    if plan.conflict:
        raise SystemExit(f"  {plan.reason}")
    log(resume.describe(plan))
    evaluator = DiffusionEvaluator(fp, slo, [r["prompt"] for r in rows], str(rd),
                                   gpu=gpu, port=port, log=log)
    evaluator.profile = profile
    if plan.resuming:
        evaluator.replay = plan.cache
    else:
        (rd / "trials.jsonl").write_text("")

    ctx.incumbent = seed_config(shape)
    dag_json = json.loads(Path(dag or default_dag("diffusion")).read_text())
    t0 = time.time()
    res = traverse(dag_json, ctx, evaluator, log=log, journal=rd / "trials.jsonl",
                   provenance=stamp, max_launches=max_launches, max_minutes=max_minutes)
    kept = [t for t in res.trials if t.kept]
    ok = [t for t in res.trials if t.goodput]
    out = Result(
        model=model, strategy="sequential", trials=list(res.trials), slo=slo,
        chosen=(kept[-1] if kept else (res.trials[0] if res.trials else None)),
        best_seen=(max(ok, key=lambda t: t.goodput) if ok else None),
        frontier=list(res.frontier()), launches=res.launches,
        minutes=res.minutes if res.minutes else (time.time() - t0) / 60,
        run_dir=str(rd),
        provenance={**stamp, "diffusion": shape.model_dump(),
                    "budget": {"max_launches": max_launches, "max_minutes": max_minutes}},
        extra={"incumbent": res.incumbent, "visited": res.visited, "skipped": res.skipped,
               "stopped_early": res.stopped_early, "unit": "frames"},
    )
    out.save()
    return out
