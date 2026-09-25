"""Serve a model with any configuration and run inference against it.

    python -m inferopt.run_inference --run RUN_DIR --prompt "..."         # the walk's chosen config
    python -m inferopt.run_inference --model M --config cfg.json --prompt "..."
    python -m inferopt.run_inference --model M --config '{"num_inference_steps": 15}' --prompts p.txt
    python -m inferopt.run_inference --run RUN_DIR --serve-only          # leave it up, print the curl

A config is the same dict the walk records in result.json: server-side keys
become the engine's flags (max_model_len, dllm_block_size, quantization,
attention_backend, ...) and request-side keys ride on every request
(num_inference_steps, guidance_scale, enable_cache_dit, cache_dit_params, ...).
The engine is chosen the way the walk chose it: SGLang Diffusion for a
diffusers pipeline, SGLang for a masked diffusion LM, vLLM otherwise, unless
--engine says. Outputs land in --out: images as PNG, videos as MP4, text on
stdout with tokens per second.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import httpx

from inferopt import request as req
from inferopt.diffusion import is_pipeline, pipeline_shape, request_body
from inferopt.engines import ENGINES, SglangDiffusionEngine
from inferopt.evaluator import HOST, child_env, port_holder, with_parent_death_signal

DEFAULT_PROMPT = "a red kite over a green field, photograph, natural light"
DEFAULT_LLM_PROMPT = "What is 17 * 23? Show your work briefly."


def load_config(args) -> tuple[str, dict]:
    """(model, config) from --run or --model/--config."""
    if args.run:
        result = json.loads(Path(args.run, "result.json").read_text())
        chosen = result.get("chosen") or {}
        model = args.model or result["model"]
        config = dict(chosen.get("config") or {})
    else:
        if not args.model:
            sys.exit("--model is required without --run")
        model, config = args.model, {}
    if args.config:
        text = Path(args.config).read_text() if Path(args.config).exists() else args.config
        config.update(json.loads(text))
    return model, config


def pick_engine(model: str, name: str | None):
    arch = ""
    if not is_pipeline(model):
        arch = (req._hf_config(model).get("architectures") or ["unknown"])[0]
    if name:
        kind = name
    elif not arch:
        kind = "sglang-diffusion"
    else:
        kind = req.serving_engine_of(arch)
    engine = SglangDiffusionEngine(model=model) if kind == "sglang-diffusion" else ENGINES[kind]()
    return kind, engine, arch


def engine_defaults(engine, model: str, arch: str) -> dict:
    """What the walk would put under a config on this card: the memory
    fraction a unified-memory box survives, the MoE backend sm12x needs, a
    dLLM's batch cap. Best effort; a bare `vllm serve` is what you get when
    the card cannot be read."""
    try:
        # The request validates that its trace exists; the hardware probe
        # never reads it, so an empty file satisfies it.
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
            trace = fh.name
        hw = req.detect_hardware(req.InferOptRequest(model=model, trace=trace))
        dense = True
        if arch:
            config = req._hf_config(model)
            text = config.get("text_config") if isinstance(config.get("text_config"), dict) else {}
            dense = not any(k in config or k in text
                            for k in ("num_experts", "num_local_experts", "n_routed_experts"))
        fp = SimpleNamespace(
            hw=hw, diffusion=None,
            model=SimpleNamespace(is_dense=dense, decoding=req.decoding_of(arch) if arch else "denoising"))
        return dict(engine.defaults(fp))
    except Exception as why:
        print(f"no engine defaults applied ({type(why).__name__}: {why})", flush=True)
        return {}


def launch(engine, model: str, port: int, config: dict, log_path: Path) -> subprocess.Popen:
    holder = port_holder(port)
    if holder is not None:
        sys.exit(f"port {port} is already bound by {holder}; stop it or pass --port")
    cmd = engine.serve_argv(model, HOST, port, config, workdir=log_path.parent)
    print(f"launching: {' '.join(cmd)}", flush=True)
    fh = open(log_path, "wb")
    return subprocess.Popen(with_parent_death_signal(cmd), stdout=fh, stderr=subprocess.STDOUT,
                            env=child_env(CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "0")),
                            start_new_session=True)


def wait_healthy(proc, base_url: str, health_path: str, log_path: Path, timeout_s: float) -> None:
    started = time.monotonic()
    last_note = started
    while True:
        if proc.poll() is not None:
            sys.exit(f"server exited {proc.returncode} during startup; see {log_path}")
        try:
            if httpx.get(f"{base_url}{health_path}", timeout=2).status_code == 200:
                print(f"healthy after {time.monotonic() - started:.0f}s", flush=True)
                return
        except httpx.HTTPError:
            pass
        if time.monotonic() - started > timeout_s:
            sys.exit(f"not healthy in {timeout_s:.0f}s; see {log_path}")
        if time.monotonic() - last_note > 60:
            last_note = time.monotonic()
            print(f"  still starting ({(last_note - started) / 60:.0f} min)", flush=True)
        time.sleep(2)


def stop(proc) -> None:
    if proc.poll() is None:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=30)


def run_llm(base_url: str, model: str, prompts: list[str], max_tokens: int, sampling: dict) -> None:
    for prompt in prompts:
        started = time.perf_counter()
        first = None
        text, tokens = [], 0
        with httpx.stream("POST", f"{base_url}/v1/chat/completions", timeout=3600, json={
                "model": model, "max_tokens": max_tokens, **sampling, "stream": True,
                "stream_options": {"include_usage": True, "continuous_usage_stats": True},
                "messages": [{"role": "user", "content": prompt}]}) as r:
            if r.status_code != 200:
                r.read()
                print(f"\n>>> {prompt}\nHTTP {r.status_code}: {r.text[:400]}", flush=True)
                continue
            for line in r.iter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                for choice in chunk.get("choices") or []:
                    piece = (choice.get("delta") or {}).get("content")
                    if piece:
                        first = first or time.perf_counter()
                        text.append(piece)
                usage = chunk.get("usage") or {}
                tokens = max(tokens, usage.get("completion_tokens") or 0)
        elapsed = time.perf_counter() - started
        ttft = (first - started) if first else elapsed
        # Over the whole request, not after the first token: a diffusion LM
        # commits a whole canvas at once, so nearly all of its tokens arrive
        # with the first chunk and a decode-only rate would read as infinite.
        rate = tokens / elapsed if tokens and elapsed else 0.0
        print(f"\n>>> {prompt}\n{''.join(text).strip()}\n"
              f"[{tokens} tokens in {elapsed:.1f} s, TTFT {ttft * 1000:.0f} ms, "
              f"{rate:.1f} tok/s end to end]", flush=True)


def run_diffusion(base_url: str, model: str, config: dict, prompts: list[str], out: Path, seed: int) -> None:
    shape = pipeline_shape(model, config)
    if shape is None:
        sys.exit(f"{model} has no model_index.json")
    out.mkdir(parents=True, exist_ok=True)
    for i, prompt in enumerate(prompts):
        body = request_body(shape, config, prompt, seed + i)
        started = time.perf_counter()
        if shape.kind == "image":
            r = httpx.post(f"{base_url}/v1/images/generations", json=body, timeout=3600)
            r.raise_for_status()
            path = out / f"{i:03d}.png"
            path.write_bytes(base64.b64decode(r.json()["data"][0]["b64_json"]))
        else:
            r = httpx.post(f"{base_url}/v1/videos", json=body, timeout=600)
            r.raise_for_status()
            video_id = r.json()["id"]
            while True:
                time.sleep(1)
                job = httpx.get(f"{base_url}/v1/videos/{video_id}", timeout=60).json()
                status = str(job.get("status", "")).lower()
                if status in ("completed", "succeeded", "done"):
                    break
                if status in ("failed", "error", "cancelled"):
                    sys.exit(f"video job {status}: {job.get('error')}")
            content = httpx.get(f"{base_url}/v1/videos/{video_id}/content", timeout=600)
            content.raise_for_status()
            path = out / f"{i:03d}.mp4"
            path.write_bytes(content.content)
        elapsed = time.perf_counter() - started
        unit = f"{body.get('num_frames', 1) / elapsed:.2f} frames/s" if shape.kind == "video" else f"{1 / elapsed:.2f} img/s"
        print(f">>> {prompt}\n    {path}  {elapsed:.1f}s  {unit}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="a walk's run dir; takes model and chosen config from its result.json")
    ap.add_argument("--model", help="HF model id (required without --run; overrides the run's)")
    ap.add_argument("--config", help="JSON file or inline JSON, merged over the run's chosen config")
    ap.add_argument("--engine", choices=sorted(ENGINES), help="override the engine choice")
    ap.add_argument("--prompt", action="append", help="a prompt; repeatable")
    ap.add_argument("--prompts", help="a file with one prompt per line, or a trace .jsonl with a prompt field")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="inference-out", help="where images and videos are written")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--launch-timeout", type=float, default=1800)
    ap.add_argument("--serve-only", action="store_true", help="launch, print how to call it, and keep serving")
    ap.add_argument("--keep", action="store_true", help="leave the server running after the prompts")
    args = ap.parse_args(argv)

    model, config = load_config(args)
    kind, engine, arch = pick_engine(model, args.engine)
    # The walk's defaults for this card go UNDER the config: an explicit key
    # wins, and a bare --model still launches the way a trial would.
    config = {**engine_defaults(engine, model, arch), **config}
    prompts = list(args.prompt or [])
    if args.prompts:
        for line in Path(args.prompts).read_text().splitlines():
            if not line.strip():
                continue
            prompts.append(json.loads(line)["prompt"] if line.lstrip().startswith("{") else line)
    if not prompts:
        prompts = [DEFAULT_PROMPT if kind == "sglang-diffusion" else DEFAULT_LLM_PROMPT]
    print(f"model   {model}\nengine  {kind}\nconfig  {json.dumps(config)}", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base_url = f"http://{HOST}:{args.port}"
    proc = launch(engine, model, args.port, config, out / "server.log")
    try:
        wait_healthy(proc, base_url, engine.health_path, out / "server.log", args.launch_timeout)
        if kind == "sglang-diffusion":
            shape = pipeline_shape(model, config)
            example = request_body(shape, config, prompts[0], args.seed) if shape else {}
            endpoint = "/v1/videos" if shape and shape.kind == "video" else "/v1/images/generations"
        else:
            example = {"model": model, "messages": [{"role": "user", "content": prompts[0]}], "max_tokens": args.max_tokens}
            endpoint = "/v1/chat/completions"
        print(f"\ncurl -s {base_url}{endpoint} -H 'content-type: application/json' -d '{json.dumps(example)}'\n", flush=True)
        if args.serve_only:
            print("serving; Ctrl-C to stop", flush=True)
            proc.wait()
            return 0
        if kind == "sglang-diffusion":
            run_diffusion(base_url, model, config, prompts, out, args.seed)
        else:
            # Greedy where the server takes it; vLLM refuses temperature and
            # seed on a diffusion LM.
            greedy = not (kind == "vllm" and arch and req.decoding_of(arch) == "diffusion")
            run_llm(base_url, model, prompts, args.max_tokens, {"temperature": 0} if greedy else {})
        if args.keep:
            print("\nserver left running; Ctrl-C to stop", flush=True)
            proc.wait()
    except KeyboardInterrupt:
        pass
    finally:
        stop(proc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
