"""The serving engine behind a measurement, as one object.

The evaluator was welded to vLLM in six places: how to invoke it, how to read
its flag list, how a config becomes flags, its version, its /metrics names and
what it needs to run at all on a given card. Each is a method here, so a second
engine is a second class rather than six more branches.

THE DAG'S VOCABULARY DOES NOT CHANGE. `max_model_len`, `enable_prefix_caching`
and the rest stay the canonical names, because every node's rationale was
measured in them; an engine that spells them differently translates. A config
key an engine cannot express is passed through as a flag of the same name, so
the flag check against the engine's own --help catches it before a launch is
spent, exactly as today.

Which engine runs is decided once, from the fingerprint. A masked diffusion LM
decodes by denoising a block of tokens, which SGLang serves and vLLM does not;
everything else stays on vLLM. INFEROPT_ENGINE overrides, for a run that wants
to measure an autoregressive model on SGLang deliberately.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


class Engine:
    """What the evaluator asks of a server it launches. Subclasses fill it in."""

    name = ""
    health_path = "/health"

    # /metrics names this engine emits, each with the reduction its type
    # requires when several series exist: a gauge takes the worst, a counter
    # adds up.
    REDUCE: dict[str, str] = {}

    # Flags that take an int. A float here is not a rounding preference, it is
    # a launch failure: argparse rejects "512.0" and the server exits during
    # argument parsing, before it reads a single weight.
    INT_FLAGS: frozenset[str] = frozenset()

    def __init__(self):
        self._command: list[str] | None = None
        self._flags: set[str] | None = None

    # --- invocation -------------------------------------------------------

    def command(self) -> list[str]:
        raise NotImplementedError

    def help_argv(self) -> list[str]:
        raise NotImplementedError

    def serve_argv(self, model: str, host: str, port: int, config: dict,
                   workdir: Path | None = None) -> list[str]:
        raise NotImplementedError

    def version(self) -> str:
        raise NotImplementedError

    def installed_flags(self, env: dict | None = None) -> set[str]:
        """Every --flag this engine's own help mentions, underscored.

        Cached for the process. Empty when the help could not be read, which
        the caller treats as "cannot validate" rather than "nothing to
        validate": see the launch path.
        """
        if self._flags is None:
            try:
                out = subprocess.run(self.help_argv(), env=env, capture_output=True,
                                     text=True, timeout=180).stdout
                self._flags = {m.group(1).replace("-", "_")
                               for m in re.finditer(r"--([a-z0-9][a-z0-9-]*)", out)}
            except Exception:
                self._flags = set()
        return self._flags

    def flag_names(self, config: dict) -> set[str]:
        """The engine flags this config would set, for checking against
        installed_flags(). Translation may rename or drop keys, so the check
        has to look at what would actually be passed."""
        argv = self.translate(config, workdir=None)
        return {arg[2:].replace("-", "_").removeprefix("no_")
                for arg in argv if arg.startswith("--")}

    # --- config -----------------------------------------------------------

    def translate(self, config: dict, workdir: Path | None = None) -> list[str]:
        """The DAG's config as this engine's flags. `workdir` is where a
        translation may write a file the engine reads."""
        raise NotImplementedError

    def generic_flags(self, items) -> list[str]:
        """The default spelling: --key-with-dashes value, booleans as lone
        flags with a --no- form, dicts as JSON, ints kept whole."""
        args: list[str] = []
        for k, v in items:
            flag = "--" + k.replace("_", "-")
            if isinstance(v, bool):
                args.append(flag if v else "--no-" + k.replace("_", "-"))
            elif isinstance(v, dict):
                args += [flag, json.dumps(v)]
            elif isinstance(v, float) and k in self.INT_FLAGS:
                args += [flag, str(round(v))]
            else:
                args += [flag, str(v)]
        return args

    def defaults(self, fp) -> dict:
        """Flags this (model, card) pair REQUIRES on this engine to run at all.
        Merged UNDER the caller's config, so an explicit value always wins."""
        return {}

    # --- metrics ----------------------------------------------------------

    def derive(self, series: dict, reduce) -> dict:
        """inferopt's scalars from this engine's /metrics series.

        `reduce(name)` collapses one metric's series by the rule in REDUCE.
        Returns only what was present; a scalar the engine does not emit is
        absent rather than zero, because zero preemptions and no preemption
        counter are different facts.
        """
        raise NotImplementedError


def _resolve(env_override: str, script: str, module: str) -> list[str]:
    """Find a console script, most explicit first.

      1. the override variable, if the caller wants to name it exactly
      2. the console script on PATH
      3. the console script under sys.prefix, for a venv whose bin is not on PATH
      4. python -m <module>, using THIS interpreter

    Four always works when the package imports: the console script is a shim
    around exactly that module, and it cannot pick up a different environment
    than the one already imported.
    """
    override = os.environ.get(env_override)
    if override:
        return override.split()
    bindir = str(Path(sys.executable).parent)
    path = bindir + os.pathsep + os.environ.get("PATH", "")
    found = shutil.which(script, path=path)
    if found:
        return [found]
    under_prefix = Path(sys.prefix) / "bin" / script
    if under_prefix.exists():
        return [str(under_prefix)]
    import importlib.util
    if importlib.util.find_spec(module.split(".")[0]) is not None:
        return [sys.executable, "-m", module]
    return []


class VllmEngine(Engine):
    """vLLM, exactly as the evaluator has always driven it."""

    name = "vllm"
    REDUCE = {
        "vllm:kv_cache_usage_perc": "max",          # a fraction: worst pressure
        "vllm:num_preemptions": "sum",
        "vllm:num_preemptions_total": "sum",
        "vllm:prefix_cache_hits": "sum",
        "vllm:prefix_cache_hits_total": "sum",
        "vllm:prefix_cache_queries": "sum",
        "vllm:prefix_cache_queries_total": "sum",
        "vllm:spec_decode_num_accepted_tokens": "sum",
        "vllm:spec_decode_num_draft_tokens": "sum",
    }
    INT_FLAGS = frozenset({
        "max_num_seqs", "max_num_batched_tokens", "max_model_len", "block_size",
        "max_loras", "max_cpu_loras", "max_lora_rank", "num_speculative_tokens",
        "prompt_lookup_max", "prompt_lookup_min", "tensor_parallel_size",
        "pipeline_parallel_size", "max_num_partial_prefills", "swap_space",
        "max_seq_len_to_capture", "seed",
    })

    def command(self) -> list[str]:
        if self._command is None:
            self._command = _resolve("INFEROPT_VLLM_CMD", "vllm", "vllm.entrypoints.cli.main")
        return self._command

    def help_argv(self) -> list[str]:
        return [*self.command(), "serve", "--help=all"]

    def serve_argv(self, model, host, port, config, workdir=None) -> list[str]:
        return [*self.command(), "serve", model, "--host", host, "--port", str(port),
                *self.translate(config, workdir)]

    def version(self) -> str:
        try:
            import vllm
            return getattr(vllm, "__version__", "unknown")
        except Exception:
            return "unknown"

    def translate(self, config, workdir=None) -> list[str]:
        return self.generic_flags((k, v) for k, v in config.items() if k != "model")

    def defaults(self, fp) -> dict:
        # gpu_memory_utilization: 0.75 on unified memory, where the fraction is
        # of SYSTEM memory the CPU also competes for and 0.90 runs a 122 GB box
        # into the OOM killer. 0.90 on a dedicated GPU, where 0.75 strands 20 GB.
        #
        # moe_backend=triton: MoE on sm12x. vLLM defaults to FlashInfer CUTLASS,
        # no prebuilt sm120 kernels ship, and the JIT build does not finish in
        # any reasonable time: sm120/121 has 99 KiB shared memory per block
        # against sm100's 228 KiB, so tile configs written for datacenter
        # Blackwell cannot fit.
        out = {"gpu_memory_utilization": 0.75 if fp.hw.unified_memory else 0.90}
        if not fp.model.is_dense and fp.hw.sm_major == 12:
            if "moe_backend" in self.installed_flags():
                out["moe_backend"] = "triton"
        return out

    def derive(self, series, reduce) -> dict:
        g = lambda *ks: next((reduce(k) for k in ks if k in series), None)
        hits = g("vllm:prefix_cache_hits", "vllm:prefix_cache_hits_total")
        queries = g("vllm:prefix_cache_queries", "vllm:prefix_cache_queries_total")
        accepted = g("vllm:spec_decode_num_accepted_tokens")
        drafts = g("vllm:spec_decode_num_draft_tokens")
        return {k: v for k, v in {
            "kv_cache_util": g("vllm:kv_cache_usage_perc"),
            "preemptions": g("vllm:num_preemptions", "vllm:num_preemptions_total"),
            "prefix_hit_rate": (hits / queries) if hits is not None and queries else None,
            "spec_acceptance_rate": (accepted / drafts) if accepted is not None and drafts else None,
        }.items() if v is not None}


class SglangEngine(Engine):
    """SGLang's LLM path, which is also where a masked diffusion LM is served.

    Flag names below were read from sglang main, python/sglang/srt/arg_groups,
    on 23 Sep 2026. Anything marked verify has a shape that was confirmed and
    a value that was not; a launch says which.
    """

    name = "sglang"
    REDUCE = {
        "sglang:token_usage": "max",                    # KV in use, a fraction
        "sglang:num_retracted_requests_total": "sum",   # a retraction is a preemption
        "sglang:num_retracted_reqs": "max",
        "sglang:cache_hit_rate": "max",
        "sglang:spec_accept_rate": "max",
    }
    INT_FLAGS = frozenset({
        "context_length", "max_running_requests", "chunked_prefill_size",
        "max_prefill_tokens", "page_size", "tp_size", "pp_size",
        "speculative_num_draft_tokens", "speculative_num_steps", "cuda_graph_max_bs",
    })

    def command(self) -> list[str]:
        if self._command is None:
            self._command = _resolve("INFEROPT_SGLANG_CMD", "sglang", "sglang.launch_server")
        return self._command

    def _serve_prefix(self) -> list[str]:
        # `sglang serve ...` through the console script, `python -m
        # sglang.launch_server ...` through the module: same arguments after.
        cmd = self.command()
        return cmd if cmd and cmd[-1] == "sglang.launch_server" else [*cmd, "serve"]

    def help_argv(self) -> list[str]:
        return [*self._serve_prefix(), "--help"]

    def serve_argv(self, model, host, port, config, workdir=None) -> list[str]:
        # --enable-metrics: SGLang does not expose /metrics unless asked, and
        # the KV gauge is sampled during every window.
        return [*self._serve_prefix(), "--model-path", model, "--host", host,
                "--port", str(port), "--enable-metrics", *self.translate(config, workdir)]

    def version(self) -> str:
        try:
            import sglang
            return getattr(sglang, "__version__", "unknown")
        except Exception:
            return "unknown"

    # DAG key -> SGLang flag, for the keys that are a plain rename.
    RENAMES = {
        "max_model_len": "context_length",
        "max_num_seqs": "max_running_requests",
        "block_size": "page_size",
        "gpu_memory_utilization": "mem_fraction_static",
        "tensor_parallel_size": "tp_size",
        "pipeline_parallel_size": "pp_size",
        "moe_backend": "moe_runner_backend",
        "quantization": "quantization",            # names per format: verify
    }

    def translate(self, config, workdir=None) -> list[str]:
        cfg = {k: v for k, v in config.items() if k != "model"}
        out: list[tuple[str, object]] = []

        # Prefix caching and chunked prefill are ON by default in SGLang, so
        # the DAG's True is silence and its False is the flag.
        if cfg.pop("enable_prefix_caching", True) is False:
            out.append(("disable_radix_cache", True))
        chunked = cfg.pop("enable_chunked_prefill", True)
        budget = cfg.pop("max_num_batched_tokens", None)
        if chunked is False:
            out.append(("chunked_prefill_size", -1))
            if budget is not None:
                out.append(("max_prefill_tokens", budget))
        elif budget is not None:
            out.append(("chunked_prefill_size", budget))

        if cfg.pop("enforce_eager", False):
            out.append(("disable_cuda_graph", True))

        kv_dtype = cfg.pop("kv_cache_dtype", None)
        if kv_dtype is not None:
            # vLLM's "fp8" is e4m3 by default; SGLang wants it spelled out.
            out.append(("kv_cache_dtype", "fp8_e4m3" if kv_dtype == "fp8" else kv_dtype))

        spec = cfg.pop("speculative_config", None)
        if isinstance(spec, dict):
            out += self._speculative(spec)

        out += self._dllm(cfg, workdir)

        for key, value in cfg.items():
            out.append((self.RENAMES.get(key, key), value))
        return self.generic_flags(out)

    @staticmethod
    def _speculative(spec: dict) -> list[tuple[str, object]]:
        """vLLM's speculative_config JSON as SGLang's flat flags.

        The ngram algorithm's spelling is the one thing here not read from a
        launch that worked: sglang main has speculative_ngram_* flags and an
        algorithm choice list this code has not seen. A wrong name fails at
        the flag check, not mid measurement."""
        method = str(spec.get("method", "")).lower()
        out: list[tuple[str, object]] = [("speculative_algorithm", method.upper() or "NGRAM")]
        if spec.get("num_speculative_tokens") is not None:
            out.append(("speculative_num_draft_tokens", spec["num_speculative_tokens"]))
        if spec.get("model"):
            out.append(("speculative_draft_model_path", spec["model"]))
        return out

    @staticmethod
    def _dllm(cfg: dict, workdir: Path | None) -> list[tuple[str, object]]:
        """The dLLM keys. Algorithm and scheduling are flags; block size and
        threshold live in a YAML the server reads, so one is written per
        launch, beside its log, and the path is what travels."""
        algorithm = cfg.pop("dllm_algorithm", None)
        block_size = cfg.pop("dllm_block_size", None)
        threshold = cfg.pop("dllm_threshold", None)
        fdfo = cfg.pop("dllm_fdfo", None)
        out: list[tuple[str, object]] = []
        if algorithm is not None:
            out.append(("dllm_algorithm", algorithm))
        settings = {k: v for k, v in (("block_size", block_size), ("threshold", threshold))
                    if v is not None}
        if settings:
            target = (workdir or Path(".")) / "dllm.yaml"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("".join(f"{k}: {v}\n" for k, v in settings.items()))
            out.append(("dllm_algorithm_config", str(target)))
        if fdfo is not None:
            out.append(("dllm_fdfo", bool(fdfo)))
        return out

    def defaults(self, fp) -> dict:
        out = {"gpu_memory_utilization": 0.75 if fp.hw.unified_memory else 0.90}
        if not fp.model.is_dense and fp.hw.sm_major == 12:
            if "moe_runner_backend" in self.installed_flags():
                out["moe_backend"] = "triton"
        if fp.model.decoding == "diffusion":
            # What SGLang's own LLaDA 2 tests launch with
            # (test/registered/dllm/test_dllm_batching_fdfo.py): the model's
            # config and tokenizer are remote code, the algorithm is not
            # defaulted by the server, and flashinfer is the backend they test.
            out.update({"trust_remote_code": True, "dllm_algorithm": "LowConfidence",
                        "attention_backend": "flashinfer"})
        return out

    def derive(self, series, reduce) -> dict:
        g = lambda *ks: next((reduce(k) for k in ks if k in series), None)
        return {k: v for k, v in {
            "kv_cache_util": g("sglang:token_usage"),
            "preemptions": g("sglang:num_retracted_requests_total", "sglang:num_retracted_reqs"),
            "prefix_hit_rate": g("sglang:cache_hit_rate"),
            "spec_acceptance_rate": g("sglang:spec_accept_rate"),
        }.items() if v is not None}


class SglangDiffusionEngine(SglangEngine):
    """SGLang Diffusion, python/sglang/multimodal_gen, for image and video.

    Same console script, `sglang serve --model-path`, which routes a diffusers
    pipeline to the diffusion runtime by itself. Two kinds of key live in a
    diffusion config: server flags, translated here, and per request sampling
    parameters (steps, guidance, size, frames, seed), which are not flags at
    all; the sample driver sends them with every request, so translate drops
    them and the flag check never sees them.
    """

    name = "sglang-diffusion"
    REQUEST_KEYS = frozenset({
        "num_inference_steps", "guidance_scale", "true_cfg_scale", "width", "height",
        "num_frames", "fps", "seed", "negative_prompt", "enable_teacache", "flow_shift",
    })
    REDUCE = {
        "sglang:diffusion_num_running_reqs": "max",
        "sglang:diffusion_num_queue_reqs": "max",
        "sglang:diffusion_generation_batch_size": "max",
        "sglang:diffusion_requests_total": "sum",
    }
    INT_FLAGS = frozenset({"tp_size", "sp_degree", "ulysses_degree", "ring_degree",
                           "cfg_parallel_degree", "batching_max_size", "batching_delay_ms",
                           "num_gpus"})

    def translate(self, config, workdir=None) -> list[str]:
        out: list[tuple[str, object]] = []
        for key, value in config.items():
            if key == "model" or key in self.REQUEST_KEYS:
                continue
            # cache-dit and component quantisation take a JSON document; the
            # generic spelling already writes a dict as JSON.
            out.append((key, value))
        return self.generic_flags(out)

    def defaults(self, fp) -> dict:
        return {}

    def derive(self, series, reduce) -> dict:
        g = lambda *ks: next((reduce(k) for k in ks if k in series), None)
        return {k: v for k, v in {
            "batch_size_peak": g("sglang:diffusion_generation_batch_size"),
            "queued_peak": g("sglang:diffusion_num_queue_reqs"),
            "running_peak": g("sglang:diffusion_num_running_reqs"),
        }.items() if v is not None}


ENGINES = {"vllm": VllmEngine, "sglang": SglangEngine, "sglang-diffusion": SglangDiffusionEngine}


def engine_for(fp=None, name: str | None = None) -> Engine:
    """The engine a fingerprint needs, unless the caller or INFEROPT_ENGINE says.

    A masked diffusion LM is served by SGLang and not by vLLM, so that is not
    a preference. Everything else stays where every number in this repo was
    measured.
    """
    chosen = name or os.environ.get("INFEROPT_ENGINE")
    if not chosen:
        decoding = getattr(getattr(fp, "model", None), "decoding", "autoregressive")
        if getattr(fp, "diffusion", None) is not None or decoding == "denoising":
            chosen = "sglang-diffusion"
        else:
            chosen = "sglang" if decoding == "diffusion" else "vllm"
    if chosen not in ENGINES:
        raise ValueError(f"unknown engine {chosen!r}; have {', '.join(sorted(ENGINES))}")
    return ENGINES[chosen]()
