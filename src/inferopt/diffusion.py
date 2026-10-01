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
import re
import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import httpx

from inferopt.engines import SglangDiffusionEngine
from inferopt.evaluator import HOST, VllmEvaluator, WARMUP_S
from inferopt.finalists import sweep_finalists
from inferopt.fingerprint import (SLO, Context, DiffusionShape, Fingerprint,
                                  LoraFingerprint, ModelFingerprint,
                                  WorkloadFingerprint)
from inferopt.traverse import Trial

SWEEP_LEVELS = (1, 2, 4, 8)
# Memory kept free while the search raises load, the line earlyoom stops
# things at on the DGX. A step up whose predicted peak would cross it is not
# measured: running the box out of memory took its network down with it
# (MiniMax H3, 30 Sep 2026).
KEEP_FREE_SHARE = 0.15
MEMORY_POLL_S = 2.0
WINDOW_S = 120.0            # the floor; a window has to hold several samples, see measure()
SAMPLES_PER_WINDOW = 4      # at L=1. A 74 s clip in a 120 s window completed once per level and
                            # every level read the same number, which cannot be true
PROBE_SAMPLES = 8           # fixed seed prompts for equivalence and quality, images
PROBE_SAMPLES_VIDEO = 4     # clips take minutes each
# The seed's calibration: how many samples, one at a time, speak for whether
# the per-sample target is reachable before the sweep is spent on it.
CALIBRATION_SAMPLES = 4
CALIBRATION_SAMPLES_VIDEO = 3
EQUIVALENT_PSNR_DB = 40.0   # above this two renders differ by float noise, not by a kernel
POLL_S = 1.0
QUALITY_FRAMES = 4          # frames of a clip the preference model scores, first to last

VIDEO_TRANSFORMERS = ("WanTransformer3DModel", "HunyuanVideoTransformer3DModel",
                      "LTXVideoTransformer3DModel", "CosmosTransformer3DModel",
                      "SanaVideoTransformer3DModel", "MiniMaxH3Transformer3DModel",
                      "MiniMaxH3DiTModel")
# Components an index names that no server loads beside the others. MiniMax
# H3's root index lists the Ref2VA denoiser next to the FL2VA one and
# --model-variant picks one, so it is neither resident nor counted.
ALTERNATE_COMPONENTS = ("transformer_ref",)

# MiniMax H3 speaks its own contract on /v1/videos: a task, ordered conditions
# and a target of short edge, aspect ratio and seconds; the server resolves the
# aligned canvas and frame count at its one frame rate. Joint video and audio,
# CFG distilled, served by SGLang with --model-variant.
H3_FPS = 24
H3_ASPECT_RATIOS = {"21:9": 21 / 9, "16:9": 16 / 9, "4:3": 4 / 3, "1:1": 1.0, "3:4": 3 / 4, "9:16": 9 / 16}
H3_FLOW_SHIFT = 12.0          # FL2VA/model_index.json sigma_shift_scales.video
H3_AUDIO_FLOW_SHIFT = 3.0     # and .audio


def is_h3(shape: DiffusionShape) -> bool:
    return shape.transformer_class.startswith("MiniMaxH3")


def h3_aspect_ratio(width: int, height: int) -> str:
    """The nearest of the ratios H3 accepts for a text-to-video target."""
    ratio = width / max(1, height)
    return min(H3_ASPECT_RATIOS, key=lambda label: abs(H3_ASPECT_RATIOS[label] - ratio))


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
    # A spec is [library, class]; a modular index adds a loading spec as a
    # third element. Alternates are named but never loaded beside the rest.
    components = {name: spec[1] for name, spec in index.items()
                  if isinstance(spec, list) and len(spec) >= 2 and spec[1]
                  and not name.startswith("_") and name not in ALTERNATE_COMPONENTS
                  and "scheduler" not in name and not name.startswith("tokenizer")
                  and name != "processor"}
    encoders = [n for n in components if n.startswith("text_encoder")]
    denoisers = [n for n in components if n.startswith("transformer") or n.startswith("unet")]
    transformer_class = components.get("transformer") or (components[denoisers[0]] if denoisers else "")
    text_encoder = components.get("text_encoder") or (components[encoders[0]] if encoders else "")
    vae = components.get("vae", "")
    files = [{"path": s.path, "size": getattr(s, "size", 0)}
             for s in HfApi().list_repo_tree(model, recursive=True)]   # folders have no size
    # Only the components' own directories: a repository can hold more than
    # one server loads (MiniMax H3 ships its weights three times).
    total_bytes = sum((f["size"] or 0) for f in files
                      if f["path"].endswith(".safetensors") and f["path"].split("/", 1)[0] in components)

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
        # The clip itself, for a video: image_png is only its first frame,
        # decoded for the PSNR check; quality_frames spread from its first
        # frame to its last are what the preference model scores.
        self.video_mp4: bytes | None = None
        self.quality_frames: list[bytes] = []
        self.video_id: str | None = None

    @property
    def latency(self) -> float:
        return (self.done or time.perf_counter()) - self.start

    @property
    def ok(self) -> bool:
        return self.done is not None and not self.error


def h3_request_body(shape: DiffusionShape, config: dict, prompt: str, seed: int) -> dict:
    """MiniMax H3's canonical request, from the same sample every other
    pipeline states as a canvas and a frame count: the short edge and aspect
    ratio stand for width and height, seconds for frames over fps. Text to
    video and audio only; guidance is not a knob on a distilled model."""
    width = int(config.get("width", shape.width))
    height = int(config.get("height", shape.height))
    fps = int(config.get("fps", shape.fps)) or H3_FPS
    seconds = round(int(config.get("num_frames", shape.frames)) / fps, 3)
    return {"prompt": prompt, "seed": seed, "task": "t2va", "conditions": [],
            "target": {"short_edge": min(width, height), "aspect_ratio": h3_aspect_ratio(width, height),
                       "duration_seconds": seconds},
            "seconds": int(seconds) if seconds == int(seconds) else seconds,
            "num_inference_steps": int(config.get("num_inference_steps", shape.steps)),
            "flow_shift": float(config.get("flow_shift", H3_FLOW_SHIFT)),
            "audio_flow_shift": float(config.get("audio_flow_shift", H3_AUDIO_FLOW_SHIFT)),
            **{k: config[k] for k in ("profile", "num_profiled_timesteps", "negative_prompt",
                                      "enable_cache_dit", "cache_dit_params") if k in config}}


def frames_in(body: dict) -> int:
    """How many frames a video request asks for, whichever way it says so."""
    if body.get("num_frames"):
        return int(body["num_frames"])
    return int(round(float(body["target"]["duration_seconds"]) * H3_FPS))


def request_body(shape: DiffusionShape, config: dict, prompt: str, seed: int) -> dict:
    """The per request half of a config, over the workload's defaults."""
    if is_h3(shape):
        return h3_request_body(shape, config, prompt, seed)
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
        s.frames = frames_in(body)
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
                s.video_mp4 = c.content
                s.quality_frames = frames_png(c.content)
                s.image_png = s.quality_frames[0] if s.quality_frames else None
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


def frames_png(video_bytes: bytes, count: int = QUALITY_FRAMES) -> list[bytes]:
    """`count` frames of an mp4, spread from its first to its last, as PNG,
    through PyAV when it is installed. The first is always frame 0, which the
    equivalence probe compares. Empty without PyAV: a probe that cannot see
    the clip says so rather than passing.

    Several frames, not the first alone: a cache or fewer steps can leave the
    opening frame close to the baseline's and still blur or block up the
    motion after it, which a score of frame 0 never sees."""
    try:
        import av
    except ImportError:
        return []
    with av.open(io.BytesIO(video_bytes)) as container:
        images = [frame.to_image() for frame in container.decode(video=0)]
    picks = sorted({round(i * (len(images) - 1) / max(count - 1, 1)) for i in range(count)}) if images else []
    frames = []
    for index in picks:
        buf = io.BytesIO()
        images[index].save(buf, format="PNG")
        frames.append(buf.getvalue())
    return frames


_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_STAGE_DONE = re.compile(r"\[(TextEncodingStage|DenoisingStage|DecodingStage)\] finished in ([0-9.]+) seconds")
_PIXELS_DONE = re.compile(r"Pixel data generated successfully in ([0-9.]+)")
_STAGE_KEYS = {"TextEncodingStage": "text", "DenoisingStage": "denoise", "DecodingStage": "decode"}


def stage_timings(log_path: Path) -> dict | None:
    """Median seconds a request spent in each stage, from the server's own
    timers: SGLang Diffusion logs every stage it finishes and closes each
    request with "Pixel data generated" and its total. The rest of that total
    is writing the output plus any wait behind other requests, so the output
    step is the least waited requests' remainder (the 10th percentile).
    None when the log holds no complete request."""
    if not log_path.exists():
        return None
    current, requests = {}, []
    for line in _ANSI.sub("", log_path.read_text(errors="replace")).splitlines():
        done = _STAGE_DONE.search(line)
        if done:
            current[_STAGE_KEYS[done.group(1)]] = float(done.group(2))
            continue
        total = _PIXELS_DONE.search(line)
        if total and len(current) == 3:
            requests.append({**current, "total": float(total.group(1))})
            current = {}
    if not requests:
        return None
    rest = sorted(max(r["total"] - r["text"] - r["denoise"] - r["decode"], 0.0) for r in requests)
    return {**{key: round(statistics.median(r[key] for r in requests), 3) for key in ("text", "denoise", "decode")},
            "output": round(rest[len(rest) // 10], 3), "requests": len(requests)}


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
        # Mean over the set, on PickScore's own scale (about 15 to 25) over
        # 100; the walk holds a loss to the budget as a share of the
        # baseline's score, so the scale does not matter.
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

def unified_memory_in_use(meminfo: str | None = None) -> tuple[float, float] | None:
    """(used, total) GB of the whole box from /proc/meminfo: what an engine
    draws on where the card shares the host's memory, as the GB10 does."""
    try:
        text = meminfo if meminfo is not None else Path("/proc/meminfo").read_text()
    except OSError:
        return None
    fields = {}
    for line in text.splitlines():
        name, _, rest = line.partition(":")
        value = rest.split()
        if value and value[0].isdigit():
            fields[name] = int(value[0]) / 1024 / 1024
    if "MemTotal" not in fields or "MemAvailable" not in fields:
        return None
    return fields["MemTotal"] - fields["MemAvailable"], fields["MemTotal"]


def memory_in_use(gpu: str = "0") -> tuple[float, float] | None:
    """(used, total) GB of what the engine draws on: the card's own memory, or
    the box's where the card answers [N/A], as a unified memory part does."""
    try:
        out = subprocess.run(["nvidia-smi", "-i", gpu, "--query-gpu=memory.used,memory.total",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=15).stdout
        used, total = (float(part) for part in out.strip().split(","))
        return used / 1024, total / 1024
    except (OSError, ValueError, subprocess.SubprocessError):
        return unified_memory_in_use()


def fits_next_level(peak_gb: float, per_sample_gb: float, total_gb: float,
                    level: int, next_level: int) -> tuple[bool, float]:
    """Whether raising load from `level` to `next_level` samples in flight
    keeps KEEP_FREE_SHARE of memory free and the peak that predicts. A
    missing reading is not a reason to stop."""
    if not (peak_gb and per_sample_gb and total_gb):
        return True, 0.0
    predicted = peak_gb + per_sample_gb * (next_level - level)
    return predicted <= total_gb * (1 - KEEP_FREE_SHARE), predicted


def levels_under(levels, cap: int) -> list[int]:
    """The load levels to measure: all of them, or those at or under `cap`."""
    return [level for level in levels if not cap or level <= cap]


class MemoryWatch:
    """The highest memory in use while a block runs, sampled on its own
    thread. A VAE decode lasts seconds; a reading taken only before and after
    a level never sees it."""

    def __init__(self, gpu: str, read=memory_in_use):
        self.gpu, self.read = gpu, read
        self.peak_gb = 0.0
        self.total_gb = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self):
        self._sample()
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=MEMORY_POLL_S * 4)
        self._sample()
        return False

    def _run(self):
        while not self._stop.wait(MEMORY_POLL_S):
            self._sample()

    def _sample(self):
        reading = self.read(self.gpu)
        if reading:
            self.peak_gb = max(self.peak_gb, reading[0])
            self.total_gb = reading[1]


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
        self.read_memory = memory_in_use

    # what the LLM path calls that a pipeline has no answer for
    def replay_lengths(self):
        return []

    def _fits(self, memory: dict, peak_gb: float, level: int, next_level: int, el) -> bool:
        """Whether one more step up in load keeps KEEP_FREE_SHARE free; says so when not."""
        fits, predicted = fits_next_level(peak_gb, memory.get("per_sample_gb"), memory.get("total_gb"),
                                          level, next_level)
        if not fits:
            self.log(f"        {el()} L={next_level:<3d} not measured: about {predicted:.0f} of "
                     f"{memory['total_gb']:.0f} GB in use, past the {KEEP_FREE_SHARE:.0%} kept free")
        return fits

    def _cap_in_flight(self, fits: int | None):
        """What fits for the seed is the cap for every later node and the DAG
        reads it: batching needs more than one sample in flight."""
        if fits and (not self.shape.max_in_flight or fits < self.shape.max_in_flight):
            self.shape.max_in_flight = fits
            self.log(f"        at most {fits} in flight from here: that is what fits in memory")

    def _probe_prompts(self) -> list[str]:
        n = PROBE_SAMPLES_VIDEO if self.shape.kind == "video" else PROBE_SAMPLES
        return self.prompts[:n]

    def _calibrate(self, config: dict, el=lambda: "") -> dict:
        """A few samples one at a time on the seed, held to the per-sample
        target. Returns the p99 (the slowest, at this count), the count and
        whether the target was missed, which is what seed_misses_slo reads.
        The per-frame share rides on itl for a clip, as it does on a trial."""
        count = min(len(self.prompts), CALIBRATION_SAMPLES_VIDEO if self.shape.kind == "video"
                    else CALIBRATION_SAMPLES)
        prompts = self.prompts[:count]
        self.log(f"        {el()} calibrating on {count} {self.shape.kind} samples, one at a time")

        async def go():
            async with httpx.AsyncClient() as client:
                done = []
                for index, prompt in enumerate(prompts):
                    done.append(await _one(client, self.base_url, self.shape, config, prompt,
                                           5000 + index, False))
                return done

        samples = asyncio.run(go())
        ok = [s for s in samples if s.ok]
        latencies = sorted(s.latency for s in ok)
        frames = max(1, int(config.get("num_frames", self.shape.frames)))
        p99_s = latencies[-1] if latencies else float("inf")
        target_ms = self.slo.ttft_p99_ms
        summary = {
            "requests": count, "completed": len(ok), "failed": len(samples) - len(ok),
            "ttft_p99_ms": round(p99_s * 1000, 1),
            "itl_p99_ms": round(p99_s * 1000 / frames, 2),
            "latencies_s": [round(x, 1) for x in latencies],
            "failure_reasons": {s.error[:60]: 1 for s in samples if not s.ok},
            "misses": ["ttft"] if ok and target_ms and p99_s * 1000 > target_ms else [],
        }
        self.log(f"        {el()} calibration  p99 per sample {p99_s:.1f}s"
                 f"  ({len(ok)}/{count} served" + (f", target {target_ms / 1000:g}s" if target_ms else "") + ")")
        return summary

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

    def _keep_renders(self, samples: list[Sample], tag: str) -> Path:
        """Every probe render beside the launch log, whatever the node: the
        still, and the clip for a video. A lossy step (fewer steps, a cache,
        fp8) is the one whose outputs most need looking at, and a run that
        scored its renders and dropped them left nothing to look at."""
        keep = self.run_dir / "launches" / tag / "probe"
        keep.mkdir(parents=True, exist_ok=True)
        for i, s in enumerate(samples):
            if s.image_png:
                (keep / f"{i}.png").write_bytes(s.image_png)
            if s.video_mp4:
                (keep / f"{i}.mp4").write_bytes(s.video_mp4)
        return keep

    def _equivalence_of(self, samples: list[Sample], tag: str) -> float | None:
        """Fraction of probe samples that do NOT match the baseline render at
        the same seed within the PSNR bar. None until a baseline exists.

        Every PSNR is kept beside the renders (_keep_renders), because the bar
        is a number that has to be set from renders people have looked at, and
        a probe that keeps only its verdict cannot be argued with."""
        if not self.baseline_dir.exists():
            return None
        keep = self.run_dir / "launches" / tag / "probe"
        keep.mkdir(parents=True, exist_ok=True)
        psnrs: dict[str, float] = {}
        for i, s in enumerate(samples):
            ref = self.baseline_dir / f"{i}.png"
            if not s.image_png or not ref.exists():
                continue
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
            if s.video_mp4:
                (self.baseline_dir / f"{i}.mp4").write_bytes(s.video_mp4)

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
                if node_id == "incumbent":
                    self._cap_in_flight(((t.diagnostics or {}).get("memory") or {}).get("fits_in_flight"))
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
                idle = self.read_memory(self.gpu)
                # Warm-up renders at least one sample and so knows how long one
                # takes; the measurement window is sized from that, so every
                # level completes several samples rather than one.
                with MemoryWatch(self.gpu, self.read_memory) as warm_watch:
                    warm = asyncio.run(closed_loop(self.base_url, self.shape, config, self.prompts, 1,
                                                   min(WARMUP_S, 60.0)))
                one_sample_s = warm["latency_p50_s"] if warm["completed"] else self._first_latency(config)
                window_s = max(WINDOW_S, SAMPLES_PER_WINDOW * one_sample_s)
                self.log(f"        {el()} one sample {one_sample_s:.1f}s, window {window_s:.0f}s")
                # PROFILED BEFORE THE SWEEP, while the engine holds one sample's
                # memory. After a sweep its cache keeps the highest level's
                # memory and the profiler on top of that ran the DGX out
                # (MiniMax H3, 30 Sep 2026). The profile is never skipped, so
                # room is made for it instead.
                with MemoryWatch(self.gpu, self.read_memory) as profile_watch:
                    profile = self._profile_one(config, el)
                memory = {
                    "idle_gb": round(idle[0], 1) if idle else None,
                    "total_gb": round(idle[1], 1) if idle else None,
                    "per_sample_gb": (round(max(warm_watch.peak_gb - idle[0], 0.0), 1)
                                      if idle and warm_watch.peak_gb else None),
                    "profile_peak_gb": round(profile_watch.peak_gb, 1) or None,
                }
                if node_id == "incumbent":
                    # THE SEED SPEAKS FOR THE RUN, for a pipeline too. The
                    # sample is fixed by the job, so a request's cost barely
                    # varies with the prompt and there is no length grid to
                    # stratify over: the first few prompts, one at a time,
                    # held to the per-sample target. A miss returns before
                    # the sweep and the walk stops with the numbers.
                    calibration = self._calibrate(config, el)
                    if calibration.get("misses"):
                        return Trial(
                            node_id=node_id, config=dict(config), goodput=0.0,
                            ttft_p99_ms=calibration["ttft_p99_ms"],
                            itl_p99_ms=calibration["itl_p99_ms"], memory_gb=0.0,
                            slo_ok=False, concurrency=1,
                            diagnostics={"completed": calibration["completed"],
                                         "failed": calibration["failed"],
                                         "calibration": calibration, "unit": "frames"})
                pts, level_peak_gb = [], warm_watch.peak_gb
                for level in levels_under(levels or SWEEP_LEVELS, self.shape.max_in_flight):
                    if pts and not self._fits(memory, level_peak_gb, pts[-1]["concurrency"], level, el):
                        memory["fits_in_flight"] = pts[-1]["concurrency"]
                        break
                    with MemoryWatch(self.gpu, self.read_memory) as watch:
                        med = asyncio.run(closed_loop(self.base_url, self.shape, config,
                                                      self.prompts, level, window_s,
                                                      latency_target_ms=target_ms))
                    level_peak_gb = watch.peak_gb or level_peak_gb
                    med["concurrency"] = level
                    med["memory_peak_gb"] = round(watch.peak_gb, 1) or None
                    pts.append(med)
                    self.log(f"        {el()} L={level:<3d} goodput {med['goodput_frames_s']:7.2f} frames/s  "
                             f"p99 {med['latency_p99_s']:6.1f}s  slo {med['slo_attainment']:.0%}  "
                             f"({med['completed']} done)"
                             + (f"  {watch.peak_gb:.0f} GB" if watch.peak_gb else ""))
                    # Past the peak: more load cannot help. A server that does
                    # not batch samples serialises them, so latency grows with
                    # concurrency and goodput, once it falls, does not come
                    # back; the next level is fifteen minutes for the same
                    # answer.
                    if len(pts) >= 2 and med["goodput_frames_s"] < pts[-2]["goodput_frames_s"]:
                        break
                peak = max(pts, key=lambda m: m["goodput_frames_s"])
                memory["peak_gb"] = round(max([profile_watch.peak_gb, warm_watch.peak_gb]
                                              + [m["memory_peak_gb"] or 0.0 for m in pts]), 1) or None
                if node_id == "incumbent":
                    self._cap_in_flight(memory.get("fits_in_flight"))

                samples = self._render_probe_set(config)
                if node_id == "incumbent" or not self.baseline_dir.exists():
                    self._keep_baseline(samples)
                self._keep_renders(samples, tag)
                div = self._equivalence_of(samples, tag) if "equivalence" in probes else None
                if div is not None:
                    self.log(f"        {el()} equivalence  {div:.0%} of fixed seed renders differ from the baseline")
                qual: dict = {}
                if "quality" in probes and benchmarks:
                    # Every scored frame is paired with its sample's prompt:
                    # a clip's frames first to last, an image's one render.
                    pairs = [(s.prompt, frame) for s in samples
                             for frame in (s.quality_frames or ([s.image_png] if s.image_png else []))]
                    prompts = [prompt for prompt, _ in pairs]
                    pngs = [frame for _, frame in pairs]
                    score = self.scorer.score(prompts, pngs) if pngs else None
                    qual = {b: score for b in benchmarks}
                    why = ("" if score is not None else self.scorer.last_error if pngs
                           else "no frames to score: PyAV is missing or the clips did not decode")
                    self.log(f"        {el()} quality      pickscore {score}"
                             + (f"  (unscored: {why})" if score is None else ""))
                mem = memory["peak_gb"] or self._gpu_memory_gb()
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
        # Where a sample's time goes, from the log this launch just wrote.
        stages = stage_timings(self.run_dir / "launches" / tag / "server.log")
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
                         "unit": "frames", "memory": memory,
                         **({"profile": profile} if profile else {}),
                         **({"stages": stages} if stages else {})},
            slo_ok=peak["goodput_frames_s"] > 0,
        )


# --------------------------------------------------------------------- entry

def is_pipeline(model: str) -> bool:
    """A diffusers pipeline has a model_index.json at its root and no
    config.json. A language model has a config.json, and may ALSO ship a
    model_index.json: DiffusionGemma does, naming a transformers model and a
    block refinement scheduler, and it is tokens out on the LLM path."""
    from huggingface_hub import hf_hub_download
    local = Path(model)
    if local.is_dir():
        return (local / "model_index.json").exists() and not (local / "config.json").exists()
    try:
        hf_hub_download(model, "config.json")
        return False
    except Exception:
        pass
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
    if is_h3(shape):
        # A server flag, not a request key: which partition SGLang loads.
        cfg["model_variant"] = "fl2va"
    return cfg


def optimize_pipeline(*, model: str, rows: list[dict], latency_p99_ms: float,
                      qps: float = 1.0, allow_loss: float | None = None,
                      run_dir: str | None = None, gpu: str = "0", port: int = 8100,
                      max_launches: int | None = None, max_minutes: float | None = None,
                      dag: str | None = None, profile: bool = True, finalists: int = 3,
                      max_in_flight: int = 0, log=print):
    """Search a diffusers pipeline's serving configurations. The diffusion
    twin of api.optimize, and what it delegates to for a model with a
    model_index.json.

    `rows` is the workload: prompt, width, height, num_inference_steps,
    guidance_scale, and num_frames and fps for video. `latency_p99_ms` is the
    SLO: p99 seconds per sample, in ms. `max_in_flight` caps the samples in
    flight every node is measured at; 0 leaves it to what fits in memory.
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
    shape.max_in_flight = max_in_flight or 0
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
               "stopped_early": res.stopped_early, "suggested_slo": res.suggested_slo,
               "unit": "frames"},
    )
    # Stage 2.1, as for a language model: the finalists' dense curves, one
    # sample stream to six, land on their trials before anything is reported.
    out.extra["finalists"] = sweep_finalists(
        evaluator, fp, out.trials, out.frontier,
        out.chosen.node_id if out.chosen is not None else None,
        n=finalists, journal=rd / "trials.jsonl", log=log)
    out.save()
    return out
