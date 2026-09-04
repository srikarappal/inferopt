"""The real evaluator: launch a config on a GPU and measure it.

    ev = VllmEvaluator(fingerprint, slo, trace_path, run_dir)
    trial = ev.measure(config, probes=["goodput","equivalence"], benchmarks=[], node_id="x")

Implements traverse.Evaluator. The measurement details here are not arbitrary --
each one is a bug that was paid for once already:

  goodput, not throughput   raw tok/s rewards a config that serves everything
                            slowly. Goodput counts only requests that met the
                            SLO, so blowing the latency target scores zero.

  fixed-duration windows    a fixed request count on a fast config measures
                            mostly ramp-up; the parameters under search are
                            invisible until the scheduler is saturated.

  token-arrival timestamps  a request straddling the window edge contributes
                            exactly the tokens it produced inside it. Discarding
                            in-flight work penalises long generations; counting
                            the drain penalises slow configs twice.

  prefix-K equivalence      comparing whole outputs has a 19% false-positive
                            floor at the measured 0.44%/token flip rate. K comes
                            from calibration, not from a guess.

  flags checked first       vLLM's CLI surface moves between releases. An
                            unknown key costs a multi-minute failed launch, so
                            it is rejected here in microseconds instead.

HISTORY -- measurement bugs, which are the expensive kind

  THE DRAIN BUG. _load submitted at qps for `seconds`, then awaited EVERY
  submitted task. At 15.4qps x 45s that is ~693 requests; the semaphore let only
  ~60 start inside the window, and summarize counts only in-window starts. So
  633 requests contributed nothing to the measurement and took 20 minutes to
  drain -- a 45-second window measured in 21 minutes. Worse than the wall-clock:
  pass 1's backlog was still executing when pass 2 began, so pass 2 measured a
  server working through pass 1's queue. Now queued tasks are cancelled at t1
  and only in-flight ones are awaited, which also leaves the server idle between
  passes.

  Prometheus metrics were renamed silently. gpu_cache_usage_perc became
  kv_cache_usage_perc and the rate gauges became raw counters, so the scrape
  returned {} and the search degraded to blind hill-climbing with no error
  anywhere. Names are read from live /metrics now, never remembered.

  vLLM 0.26 dropped flags that used to exist: --disable-log-requests,
  --swap-space, --cuda-graph-sizes, --tokenizer-pool-size. installed_flags()
  parses `--help=all` at startup and reconciles the allowlist, so an unknown key
  fails before a launch instead of during one.

  ninja, twice. vLLM JIT-builds CUDA extensions and shells out to `ninja`, which
  lives beside the interpreter. A subprocess inherits only PATH, so a bare env
  dies deep in engine init with FileNotFoundError -- an error that reads like a
  model problem and is not. child_env() exists for this; quantize.py's load
  probe hit the identical bug later by not using it.

  A launch failure was reported as an SLO failure. Reporting inf TTFT as "does
  not satisfy the SLO" sends you to tune a threshold when the server never
  started. LaunchError is now a distinct path with the server log tail.

  Port 8000 collides with production. The OCR server sits on 8813 and vLLM
  defaults to 8000; a fixed port is a collision waiting to happen.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from calibration import STORE
from fingerprint import SLO, Fingerprint
from traverse import Trial

_VLLM_CMD: list[str] | None = None


def vllm_cmd() -> list[str]:
    """How to invoke vLLM, resolved rather than assumed.

    This was the bare string "vllm", which works only when the console script
    happens to sit on PATH or beside sys.executable. On a host where it does
    not -- the usual cause being that `python run.py` picked up a different
    interpreter from the one vLLM is installed under -- subprocess raises
    FileNotFoundError: 'vllm', which says nothing about the actual problem.

    Resolution order, most explicit first:

      1. INFEROPT_VLLM_CMD, if the caller wants to name it exactly
      2. the console script on PATH (child_env has already prepended the
         interpreter's own bin directory)
      3. the console script under sys.prefix, for a venv whose bin is not on PATH
      4. python -m vllm.entrypoints.cli.main, using THIS interpreter

    Four is the one that always works: the console script is a two-line shim
    around exactly that entry point, so if `import vllm` succeeds the module
    form succeeds too, and it cannot pick up a different environment than the
    one already imported. `python -m vllm` does NOT work -- the package has no
    __main__ -- which is why the specific entry point is named.
    """
    global _VLLM_CMD
    if _VLLM_CMD is not None:
        return _VLLM_CMD

    override = os.environ.get("INFEROPT_VLLM_CMD")
    if override:
        _VLLM_CMD = override.split()
        return _VLLM_CMD

    found = shutil.which("vllm", path=child_env().get("PATH"))
    if found:
        _VLLM_CMD = [found]
        return _VLLM_CMD

    cand = Path(sys.prefix) / "bin" / "vllm"
    if cand.exists():
        _VLLM_CMD = [str(cand)]
        return _VLLM_CMD

    try:
        import importlib.util
        if importlib.util.find_spec("vllm.entrypoints.cli.main") is not None:
            _VLLM_CMD = [sys.executable, "-m", "vllm.entrypoints.cli.main"]
            return _VLLM_CMD
    except Exception:
        pass

    raise LaunchError(
        f"cannot find vLLM.\n"
        f"  interpreter : {sys.executable}\n"
        f"  sys.prefix  : {sys.prefix}\n"
        f"  'vllm' importable: {_vllm_importable()}\n"
        f"Looked for a 'vllm' console script on PATH and under sys.prefix/bin, "
        f"and for the vllm.entrypoints.cli.main module.\n"
        f"The usual cause is running this with a different interpreter than the "
        f"one vLLM is installed under -- check that `{sys.executable} -c 'import "
        f"vllm'` works. Override explicitly with INFEROPT_VLLM_CMD if vLLM lives "
        f"somewhere unusual.")


def _artifact_quant_algo(path: str) -> str | None:
    """quant_algo an artifact declares, or None if it is not one of ours.

    hf_quant_config.json is written last by the producer, so its presence also
    means the export finished."""
    try:
        f = Path(path) / "hf_quant_config.json"
        if not f.exists():
            return None
        return (json.loads(f.read_text()).get("quantization") or {}).get("quant_algo")
    except Exception:
        return None


def moe_expert_state(path: str) -> str:
    """Whether the MoE EXPERT layers in an artifact are quantized: full/none/mixed.

    The declared quant_algo does not answer this, and assuming it does is what
    broke autoquant@6.0. That checkpoint declares MIXED_PRECISION, exactly like
    autoquant@5.0, and the two behave differently: at a 5.0-bit budget all 48
    expert layers are quantized, at 6.0 only 45 are, because the search had bits
    to spare and left three MoE layers in bf16. vLLM then rejects marlin with
    "not supported for unquantized MoE" -- while still rejecting triton for the
    other 45. A partly-quantized MoE is a THIRD state, and reading one top-level
    string cannot distinguish it.

    The evidence used is quantized_layers, which MIXED_PRECISION checkpoints
    carry: a layer present in that map but with no `.experts` entry has had its
    attention quantized and its experts left alone. Single-format checkpoints
    (NVFP4, W4A16_NVFP4) carry no such map, and for them the declared algorithm
    IS the whole truth.
    """
    try:
        f = Path(path) / "hf_quant_config.json"
        if not f.exists():
            return "unknown"
        q = (json.loads(f.read_text()).get("quantization") or {})
        algo = (q.get("quant_algo") or "").upper()
        ql = q.get("quantized_layers") or {}
        if not ql:
            return "full" if "NVFP4" in algo else "none"
        import re as _re
        seen, with_experts = set(), set()
        for k in ql:
            m = _re.match(r"model\.layers\.(\d+)\.", k)
            if not m:
                continue
            seen.add(m.group(1))
            if ".experts" in k:
                with_experts.add(m.group(1))
        if not seen:
            return "unknown"
        if not with_experts:
            return "none"
        return "full" if with_experts == seen else "mixed"
    except Exception:
        return "unknown"


def _moe_backends(kind: str) -> set[str]:
    """Backends vLLM ITSELF accepts, read from its source rather than copied.

    A copy goes stale between releases in exactly the way that produces a 3am
    ValueError. `kind` is "nvfp4" or "unquantized"."""
    try:
        import importlib
        import inspect
        import re
        # importlib, not attribute access on the package: oracle/__init__ does
        # not import every submodule, so `oracle.nvfp4` raised AttributeError
        # and this returned an empty set -- which the caller reads as "no
        # opinion" and skips the correction entirely. A silent empty set here
        # reinstates the exact bug this function exists to prevent.
        mod = importlib.import_module(
            f"vllm.model_executor.layers.fused_moe.oracle.{kind}")
        fn = getattr(mod, f"map_{kind}_backend")
        return set(re.findall(r'"(\w+)":', inspect.getsource(fn)))
    except Exception:
        return set()


def _nvfp4_moe_backends() -> set[str]:
    return _moe_backends("nvfp4")


def reconcile_moe_backend(config: dict, *, log=print) -> dict:
    """Correct a moe_backend that is invalid for the artifact being served.

    A pure function on the config, called by _serve, so it can be tested without
    a GPU -- the reason this exists as a function and not four lines inline.

    hardware_defaults sets moe_backend=triton for a MoE on sm12x, because
    FlashInfer's sm120 CUTLASS path JIT-compiles for hours on an UNQUANTIZED
    MoE. vLLM then refuses triton for a QUANTIZED NvFP4 MoE:

        ValueError: moe_backend='triton' is not supported for NvFP4 MoE.

    hardware_defaults cannot resolve that: it derives from the fingerprint and
    cannot know a quantized artifact will be loaded. This is the first point
    that knows both. Cost of not doing it: an 18-hour run built a 23 GB artifact
    it could not load, with three more conversions queued to fail the same way.

    THERE ARE THREE STATES, NOT TWO. The first version of this function assumed
    two -- quantized or not -- and switched anything NVFP4-family to marlin.
    That fixed nvfp4 and w4a16 and BROKE autoquant@6.0, whose experts are only
    partly quantized: marlin refuses the 3 unquantized expert layers and triton
    refuses the other 45. The accepted sets for the two cases overlap in exactly
    two backends, and a mixed checkpoint must use one of them:

        quantized NVFP4   cutlass, flashinfer_*, marlin, humming, emulation
        unquantized       aiter, flashinfer_cutlass, flashinfer_trtllm, triton
        BOTH              flashinfer_cutlass, flashinfer_trtllm

    flashinfer_trtllm is preferred over flashinfer_cutlass for the mixed case
    for the same reason triton was chosen originally: the sm120 CUTLASS path
    JIT-compiles for hours. marlin stays the choice for a fully quantized
    checkpoint -- it is what vLLM itself falls back to for W4A16_NVFP4, and it
    does not JIT on sm120.
    """
    mb, model = config.get("moe_backend"), config.get("model")
    if not mb or not model:
        return config
    algo = _artifact_quant_algo(model)
    if not algo:
        return config
    if "NVFP4" not in algo.upper() and algo != "MIXED_PRECISION":
        return config

    state = moe_expert_state(model)
    if state in ("none", "unknown"):
        # Experts are bf16, so the backend chosen for an unquantized MoE is
        # already right. Switching here is what broke autoquant@6.0.
        return config

    if state == "mixed":
        ok = _moe_backends("nvfp4") & _moe_backends("unquantized")
        pref = ("flashinfer_trtllm", "flashinfer_cutlass")
    else:
        ok = _moe_backends("nvfp4")
        pref = ("marlin",)
    if not ok:
        # Could not read vLLM's own accepted list. Say so: staying silent here
        # looks identical to "the backend is fine", and the launch then dies
        # minutes later with a ValueError that this function was written to
        # prevent.
        log(f"        WARNING: cannot read vLLM's accepted MoE backends; "
            f"leaving moe_backend={mb!r} unchecked against a {algo} artifact")
        return config
    if mb in ok:
        return config
    choice = next((c for c in pref if c in ok), None) or sorted(ok)[0]
    config["moe_backend"] = choice
    log(f"        moe_backend {mb!r} is invalid for a {algo} MoE whose experts "
        f"are {state}-quantized; using {choice!r}")
    return config


def hardware_defaults(fp) -> dict:
    """Flags this (model, GPU) pair REQUIRES to run at all, not tuning choices.

    Lives here, in one place, because it has to apply to every path that launches
    a server. It did not: moe_backend was set only in run.py's seed_config, so
    the traversal got it and eval_repro did not. A stock-baseline run on
    Qwen3-30B-A3B then went down FlashInfer's sm120 CUTLASS path and hung for 30
    minutes JIT-compiling kernels, exactly the failure the flag exists to
    prevent -- while the traversal beside it ran fine. A hardware fact expressed
    in one caller is a bug waiting for the second caller.

      gpu_memory_utilization  0.75 on unified memory, where the fraction is of
                              SYSTEM memory the CPU also competes for and 0.90
                              runs a 122GB box into the OOM killer. 0.90 on a
                              dedicated GPU, where 0.75 strands ~20GB.

      moe_backend=triton      MoE on sm12x. vLLM defaults to FlashInfer CUTLASS,
                              no prebuilt sm120 kernels ship, and the JIT build
                              does not finish in any reasonable time: sm120/121
                              has 99 KiB shared memory per block against sm100's
                              228 KiB, so tile configs written for datacenter
                              Blackwell cannot fit. Dense models never select a
                              MoE kernel; sm100 has the memory CUTLASS expects.

    Callers merge these UNDER their own settings, so an explicit value always
    wins -- these are defaults, not overrides.
    """
    out: dict[str, Any] = {
        "gpu_memory_utilization": 0.75 if fp.hw.unified_memory else 0.90,
    }
    if not fp.model.is_dense and fp.hw.sm_major == 12:
        if "moe_backend" in installed_flags():
            out["moe_backend"] = "triton"
    return out


def _vllm_version() -> str:
    try:
        import vllm
        return getattr(vllm, "__version__", "unknown")
    except Exception:
        return "unknown"


def _vllm_importable() -> bool:
    try:
        import importlib.util
        return importlib.util.find_spec("vllm") is not None
    except Exception:
        return False


HOST = "127.0.0.1"
LAUNCH_TIMEOUT_S = float(os.environ.get("INFEROPT_LAUNCH_TIMEOUT_S", "1800"))
# How long a launch may produce NO log output before it is declared hung. The
# deadline above is extended whenever the server writes anything, so a slow but
# talking startup is allowed to finish while a silent one fails fast.
STALL_S = float(os.environ.get("INFEROPT_LAUNCH_STALL_S", "600"))
# Backstop, so a server that logs in a loop forever cannot hold the run.
LAUNCH_HARD_CAP_S = float(os.environ.get("INFEROPT_LAUNCH_HARD_CAP_S", "10800"))


def _last_line(path: Path) -> str:
    """The last non-empty line of a log, for a failure message.

    A launch that hangs is identified by the last thing it said. Reporting only
    "not healthy in 7200s" sent two hours of investigation in the wrong
    direction when the answer -- gen_cutlass_fused_moe_sm120_module -- was
    sitting on the final line the whole time.
    """
    try:
        lines = [l.strip() for l in path.read_text(errors="replace").splitlines() if l.strip()]
        return lines[-1] if lines else "(log is empty)"
    except Exception:
        return "(log unreadable)"
# 45s, not 15s. The prefix cache does not fill in 15s: passes were identical
# (47.9/47.9) before prefix caching was enabled and 42% apart after (137.9/195.2),
# because pass 1 ran cold and pass 2 warm. Both passes must measure the same
# steady state, which is also the state production runs in.
WARMUP_S = 45.0
SETTLE_S = 20.0        # floor; the real value is derived per model, see below
SETTLE_MAX_S = 75.0    # ceiling, so a slow model cannot make the sweep unbounded
SWEEP_WINDOW_S = 45.0
SWEEP_LEVELS = (4, 8, 16, 32, 64, 128, 256)
WINDOW_S = 45.0        # open-loop window, used by --fixed-concurrency
REPEATS = 2

# Which direction is "worse" for each metric. Aggregating passes with a blanket
# min() is conservative for goodput and OPTIMISTIC for latency -- it reports the
# better of two TTFT samples, which is exactly backwards for a gate that decides
# whether an SLO was met.
LOWER_IS_BETTER = {"ttft_p99_ms", "itl_p99_ms", "failed", "window_s"}


def aggregate(passes: list[dict]) -> dict:
    """Worst value across passes, per metric direction.

    The gate asks "is this reliably better", so every metric is taken at its
    least flattering observed value: min for goodput and throughput, max for
    TTFT and ITL. `concurrency` is excluded -- the smaller of two identical L
    values is meaningless and it is carried on the Trial separately.
    """
    out = {}
    for k, v in passes[0].items():
        if k == "concurrency" or not isinstance(v, (int, float)):
            continue
        vals = [p[k] for p in passes if k in p]
        out[k] = max(vals) if k in LOWER_IS_BETTER else min(vals)
    return out


class LaunchError(RuntimeError):
    def __init__(self, msg: str, stderr: str = ""):
        super().__init__(msg)
        self.stderr = stderr


def child_env(**extra: str) -> dict[str, str]:
    """vLLM JIT-builds CUDA extensions and shells out to `ninja`, which lives
    beside the interpreter. A subprocess inherits only PATH, so an absolute-path
    invocation dies deep in engine init with FileNotFoundError."""
    bindir = str(Path(sys.executable).parent)
    path = os.environ.get("PATH", "")
    if bindir not in path.split(os.pathsep):
        path = bindir + os.pathsep + path
    return {**os.environ, "PATH": path, **extra}


_FLAGS: set[str] | None = None


def installed_flags() -> set[str]:
    global _FLAGS
    if _FLAGS is None:
        try:
            out = subprocess.run([*vllm_cmd(), "serve", "--help=all"], env=child_env(),
                                 capture_output=True, text=True, timeout=180).stdout
            _FLAGS = {m.group(1).replace("-", "_") for m in re.finditer(r"--([a-z0-9][a-z0-9-]*)", out)}
        except Exception:
            _FLAGS = set()
    return _FLAGS


# vLLM flags that take an int. A float here is not a rounding preference, it is
# a launch failure: argparse rejects "512.0" for an int-typed argument and the
# server exits during argument parsing, before it reads a single weight.
#
# This is not hypothetical. dag/llm.json computed max_num_seqs as
# `incumbent.max_num_seqs * 1.5`, which is 384.0, and BOTH retune_batching
# nodes died on every run they were ever applicable to -- four launches per
# lossy run, contributing nothing, for as long as the nodes have existed. The
# failure was invisible because a dead launch is recorded as goodput 0.0 and
# reads like a configuration that simply did not help.
#
# The expressions are fixed at the source. This exists so the next one cannot
# cost a launch: a count that arrives as a float is rounded, because a
# fractional number of sequences has no meaning to round WRONG.
_INT_FLAGS = frozenset({
    "max_num_seqs", "max_num_batched_tokens", "max_model_len", "block_size",
    "max_loras", "max_cpu_loras", "max_lora_rank", "num_speculative_tokens",
    "prompt_lookup_max", "prompt_lookup_min", "tensor_parallel_size",
    "pipeline_parallel_size", "max_num_partial_prefills", "swap_space",
    "max_seq_len_to_capture", "seed",
})


def to_cli(cfg: dict) -> list[str]:
    args: list[str] = []
    for k, v in cfg.items():
        if k == "model":
            continue
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            args.append(flag if v else "--no-" + k.replace("_", "-"))
        elif isinstance(v, dict):
            args += [flag, json.dumps(v)]
        elif isinstance(v, float) and k in _INT_FLAGS:
            args += [flag, str(round(v))]
        else:
            args += [flag, str(v)]
    return args


# --------------------------------------------------------------------------
# request-level measurement
# --------------------------------------------------------------------------

@dataclass
class Req:
    start: float = 0.0
    ttft: float | None = None
    latency: float = 0.0
    n_out: int = 0
    ok: bool = False
    text: str = ""
    token_times: list[float] = field(default_factory=list)

    def meets(self, slo: SLO) -> bool:
        """Per-request SLO satisfaction -- the definition goodput rests on."""
        if not self.ok or self.ttft is None:
            return False
        if slo.ttft_p99_ms and self.ttft * 1e3 > slo.ttft_p99_ms:
            return False
        if slo.itl_p99_ms and self.n_out > 1:
            itl = (self.latency - self.ttft) / (self.n_out - 1) * 1e3
            if itl > slo.itl_p99_ms:
                return False
        return True


async def _one(client, base_url, model, prompt, max_tokens, stream=True) -> Req:
    r = Req(start=time.perf_counter())
    payload = {"model": model, "prompt": prompt, "max_tokens": max_tokens,
               "temperature": 0.0, "seed": 0, "stream": stream}
    if stream:
        payload["stream_options"] = {"include_usage": True}
    try:
        if not stream:
            resp = await client.post(f"{base_url}/v1/completions", json=payload, timeout=900.0)
            resp.raise_for_status()
            j = resp.json()
            r.text = j["choices"][0]["text"]
            r.n_out = (j.get("usage") or {}).get("completion_tokens", 0)
            r.latency = time.perf_counter() - r.start
            r.ttft, r.ok = r.latency, True
            return r
        async with client.stream("POST", f"{base_url}/v1/completions",
                                 json=payload, timeout=900.0) as resp:
            if resp.status_code != 200:
                return r
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                body = line[6:].strip()
                if body == "[DONE]":
                    break
                ch = json.loads(body)
                if ch.get("choices") and ch["choices"][0].get("text"):
                    now = time.perf_counter()
                    if r.ttft is None:
                        r.ttft = now - r.start
                    r.n_out += 1
                    r.text += ch["choices"][0]["text"]
                    r.token_times.append(now)
                if ch.get("usage"):
                    r.n_out = ch["usage"].get("completion_tokens") or r.n_out
        r.latency = time.perf_counter() - r.start
        r.ok = r.ttft is not None
    except Exception:
        pass
    return r


async def _load(base_url, model, prompts, max_tokens, qps, conc, seconds,
                cursor: list[int] | None = None):
    """Offer load for `seconds`, then drain only the requests that started.

    The window is open-loop: requests are submitted at `qps` whether or not the
    server keeps up, and a semaphore caps in-flight work at `conc`. When offered
    load exceeds capacity the excess queues behind that semaphore.

    Queued requests are invisible to `summarize`, which counts only requests
    whose start falls inside [t0, t1). Waiting for them buys nothing: on GB10 at
    15.4qps, 693 requests were submitted, 60 started in-window, and draining the
    other 633 turned a 45s window into 21 minutes.

    So at t1, tasks still blocked on the semaphore are cancelled -- they never
    issued a request, so cancellation is clean. Tasks past the semaphore are
    mid-generation and are awaited: ITL needs a complete generation, and
    truncating in flight would bias the tail toward whatever finished fastest.

    Returns the offered/started counts so the caller can report overload.
    """
    sem = asyncio.Semaphore(conc)
    out: list[Req] = []
    live: set[asyncio.Task] = set()
    interval = 1.0 / qps if qps else 0.0
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=conc + 16)) as c:
        async def go(p):
            async with sem:
                live.add(asyncio.current_task())
                out.append(await _one(c, base_url, model, p, max_tokens))
        # The prompt index ADVANCES ACROSS PHASES. It used to restart at 0 on
        # every call, so warmup, pass 1 and pass 2 all served prompts 0..59 --
        # identical full prompts, which a warm prefix cache hits completely.
        # That inflated the measured hit rate to 72% where real prefix sharing
        # in this trace accounts for 20%, and it was most of the prefix_caching
        # win. A cursor shared across one measure() call keeps the phases
        # disjoint; it resets per node, so every config still sees the same
        # prompt sequence and comparisons stay paired.
        base = cursor[0] if cursor is not None else 0
        tasks, i = [], 0
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < seconds:
            tasks.append(asyncio.create_task(go(prompts[(base + i) % len(prompts)])))
            i += 1
            await asyncio.sleep(interval) if interval else await asyncio.sleep(0)
        t1 = time.perf_counter()
        for t in tasks:
            if t not in live and not t.done():
                t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    if cursor is not None:
        cursor[0] = base + len(live)      # advance by what actually ran
    return out, t0, t1, len(tasks), len(live)


async def _closed_loop(base_url, model, prompts, max_tokens, conc,
                       settle_s: float, window_s: float,
                       cursor: list[int] | None = None):
    """Hold exactly `conc` requests in flight, replacing each as it completes.

    The open-loop driver above fixes the ARRIVAL RATE and lets concurrency
    emerge. That matches production -- users arrive when they arrive -- but
    above capacity it does not converge: queues grow without bound, so TTFT
    depends on how long the window ran and a 45s window and a 90s window give
    different answers. Run three and run four were both measuring that.

    Closed loop is bounded by construction, so it converges and repeats. It is
    the right instrument for CHARACTERISING capacity -- sweeping L to find where
    goodput peaks. It is the wrong instrument for validating an operating point,
    because holding L constant removes burstiness, and burstiness is what moves
    the TTFT tail. Hence both drivers: sweep closed, validate open.

    SETTLE AND WINDOW ARE ONE CONTINUOUS RUN, and only the window is reported.
    Two things go wrong otherwise, both biasing the result low:

    Running settle as a separate call drains the pipeline completely before the
    measurement starts, so the window opens with `conc` requests firing
    simultaneously from an idle server -- a cold start, not the steady state the
    number is supposed to describe. Settling only helps if it flows into the
    measurement.

    Returning `time.perf_counter()` after the gather puts the DRAIN inside the
    window: workers stop launching at the deadline, but the last in-flight
    requests keep running, and at 45s windows with ~22s requests that is a third
    of the elapsed time spent at declining concurrency. Both numerator and
    denominator grow, so throughput comes out understated by roughly that
    fraction. The window is therefore the nominal interval, and tokens produced
    after it simply fall outside.
    """
    out: list[Req] = []
    issued = [0]
    base = cursor[0] if cursor is not None else 0
    start = time.perf_counter()
    t0 = start + settle_s                 # measurement window opens
    t1 = t0 + window_s                    # and closes
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=conc + 16)) as c:
        async def worker(slot: int):
            i = slot
            while time.perf_counter() < t1:
                # Same reason as _load: phases must not replay the same prompts,
                # or a warm prefix cache scores full-prompt hits that no
                # production workload would produce.
                issued[0] += 1
                out.append(await _one(c, base_url, model,
                                      prompts[(base + i) % len(prompts)], max_tokens))
                i += conc
        await asyncio.gather(*[asyncio.create_task(worker(k)) for k in range(conc)],
                             return_exceptions=True)
    if cursor is not None:
        cursor[0] = base + issued[0]
    return out, t0, t1


def summarize(reqs: list[Req], t0: float, t1: float, slo: SLO) -> dict:
    win = max(1e-9, t1 - t0)
    in_win = lambda tt: t0 <= tt < t1
    all_tok = sum(1 for r in reqs for tt in r.token_times if in_win(tt))
    good_tok = sum(1 for r in reqs if r.meets(slo) for tt in r.token_times if in_win(tt))
    started = [r for r in reqs if t0 <= r.start < t1]
    done = [r for r in started if r.ok]
    ttfts = sorted(r.ttft for r in done if r.ttft is not None)
    itls = sorted((r.latency - r.ttft) / (r.n_out - 1)
                  for r in done if r.ttft is not None and r.n_out > 1)
    pct = lambda xs, q: (xs[min(len(xs) - 1, int(q * (len(xs) - 1)))] if xs else float("nan"))
    # Two units, because the field uses both and they are not interchangeable.
    #
    # vLLM's own benchmark (vllm/benchmarks/serve.py) reports
    # `request_goodput = good_completed / dur_s` -- REQUESTS per second.
    # We optimise tokens/sec, which is the right objective when responses vary in
    # length (a config that finishes only short requests on time should not score
    # the same as one that finishes long ones). But reporting only tok/s makes
    # every number here ~mean_output_tokens x larger than the vLLM figure someone
    # would compare it against, so both are emitted and both are labelled.
    #
    # req/s is also what divides into demand for the replica count.
    good_reqs = sum(1 for r in done if r.meets(slo))
    return {
        "goodput": good_tok / win,
        "goodput_req_s": good_reqs / win,
        "throughput": all_tok / win,
        "throughput_req_s": len(done) / win,
        "slo_attainment": (sum(r.meets(slo) for r in done) / len(done)) if done else 0.0,
        "ttft_p99_ms": pct(ttfts, 0.99) * 1e3,
        "itl_p99_ms": pct(itls, 0.99) * 1e3,
        "completed": len(done), "failed": len(started) - len(done), "window_s": win,
    }


# --------------------------------------------------------------------------
# evaluator
# --------------------------------------------------------------------------

class VllmEvaluator:
    def __init__(self, fp: Fingerprint, slo: SLO, trace_path: str, run_dir: str,
                 gpu: str = "0", port: int = 8000, log=print):
        self.fp, self.slo, self.log = fp, slo, log
        self.gpu, self.port, self.run_dir = gpu, port, Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.trace_path = trace_path       # AWQ/NVFP4 calibrate on this workload
        rows = [json.loads(l) for l in open(trace_path) if l.strip()]
        self.prompts = [r["prompt"] for r in rows if r.get("prompt")]
        if not self.prompts:
            raise ValueError(
                f"{trace_path} has no 'prompt' field. The fingerprint can be built from "
                f"token counts alone, but the benchmark needs the actual text to replay."
            )
        self.max_tokens = int(fp.workload.mean_output_tokens)

        # Settle for at least ONE request duration. A closed-loop window opens
        # with all L workers firing simultaneously; until the first cohort has
        # completed and been replaced, the window is observing a cold start
        # rather than steady state. At 259 tokens and a 108ms roofline ITL a
        # request takes ~28s here, so the 20s floor was never long enough -- and
        # the error grows with L, because more simultaneous prefills queue behind
        # each other. That is what produced ttft_p99 7134ms at L=60 on a server
        # that was concurrently sustaining 221 tok/s.
        itl_floor_s = fp.model.active_weight_gb / max(1e-9, fp.hw.memory_bandwidth_gb_s)
        one_request_s = fp.workload.mean_output_tokens * itl_floor_s
        self.settle_s = min(SETTLE_MAX_S, max(SETTLE_S, one_request_s * 1.2))
        self.qps = fp.workload.request_rate_qps
        self.conc = fp.workload.max_concurrency
        cal = STORE.get(fp)
        self.equiv_k = cal.equivalence_prefix_tokens() if cal else 8
        self.equiv_ref: list[str] | None = None
        self.base_url = f"http://{HOST}:{port}"

    # --- server lifecycle ---
    @contextmanager
    def _serve(self, config: dict, tag: str):
        # `quantize: <kind>` is an instruction to PRODUCE a variant of the served
        # model, not a vLLM flag. It resolves to a local artifact path before the
        # launch, and never to a downloaded checkpoint -- see quantize.py.
        config = dict(config)
        kind = config.pop("quantize", None)
        # effective_bits travels as its own key: "autoquant@6.0" in a sweep value
        # gets parsed as a predicate expression by validate_dag, where @ is
        # MatMult. The producer still wants them joined.
        bits = config.pop("quantize_bits", None)
        if kind == "autoquant":
            if bits is None:
                raise LaunchError("quantize=autoquant requires quantize_bits")
            kind = f"autoquant@{float(bits)}"
        if kind:
            from quantize import ensure_variant
            path = ensure_variant(self.fp, kind, self.trace_path, log=self.log)
            if path:
                config["model"] = path
            else:
                config["quantization"] = kind      # fp8: a load-time flag

        reconcile_moe_backend(config, log=self.log)

        model = config.get("model") or self.fp.model.id
        # THE FLAG CHECK MUST NOT DISABLE ITSELF. It used to fall back to "[]"
        # whenever installed_flags() came back empty, which is precisely the case
        # where it is most needed: an old vLLM whose `serve --help=all` is not
        # recognised returns nothing, validation silently switches off, and every
        # config is passed blind to a binary that may not accept it. The launch
        # then dies with "exited 1 during startup" and no indication that a flag
        # was the reason.
        # Only demand the flag list when there is something to check against it.
        # A config with no keys beyond `model` cannot contain an unknown flag, so
        # refusing to launch it because `--help=all` was unreadable would block a
        # bare `vllm serve <model>` for no reason.
        to_check = [k for k in config if k != "model"]
        flags = installed_flags() if to_check else None
        if to_check and not flags and os.environ.get("INFEROPT_SKIP_FLAG_CHECK"):
            flags = None                        # explicit opt-out, launch blind
        elif to_check and not flags:
            raise LaunchError(
                f"could not read this vLLM's flag list, so no config can be "
                f"validated before launching.\n"
                f"`{' '.join(vllm_cmd())} serve --help=all` produced nothing "
                f"parseable -- older vLLM builds do not accept --help=all.\n"
                f"Set INFEROPT_SKIP_FLAG_CHECK=1 to launch anyway and let vLLM "
                f"reject bad flags itself, at the cost of a failed launch per "
                f"bad key instead of an instant error.")
        unknown = [k for k in config if k != "model" and k not in flags] if flags else []
        if unknown:
            raise LaunchError(
                f"config keys this vLLM ({_vllm_version()}) does not accept: "
                f"{unknown}. Flags move between releases -- 0.26 removed "
                f"--disable-log-requests, --swap-space and --cuda-graph-sizes, "
                f"and older builds take speculative decoding as flat flags "
                f"rather than --speculative-config JSON.")

        d = self.run_dir / "launches" / tag
        d.mkdir(parents=True, exist_ok=True)
        err = d / "server.log"
        cmd = [*vllm_cmd(), "serve", model, "--host", HOST, "--port", str(self.port), *to_cli(config)]
        env = child_env(CUDA_VISIBLE_DEVICES=self.gpu,
                        VLLM_CACHE_ROOT=str(self.run_dir / ".vllm_cache" / f"gpu{self.gpu}"))
        with open(err, "wb") as fh:
            proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                    env=env, start_new_session=True)
        try:
            # THE DEADLINE FOLLOWS PROGRESS, NOT WALL CLOCK.
            #
            # A single timeout covering Popen -> /health lumps together four
            # unrelated things: weight download, weight load, kernel JIT, and KV
            # profiling. Only the last is a property of the config under test.
            # A Qwen3-30B-A3B run spent 40 min downloading and 88 min inside
            # FlashInfer's sm120 CUTLASS MoE compile, then got SIGINT'd at 7200s
            # having measured nothing -- and the console said only "not healthy
            # in 7200s", which reads as a slow launch rather than a stuck one.
            #
            # So: while the server is still WRITING to its log it is making
            # progress and the deadline is pushed out. When it goes silent for
            # STALL_S it is hung, and that fails immediately with the last line
            # it managed to write -- which is the one thing that identifies the
            # cause. A hard cap stops a pathological loop running forever.
            #
            # This does not rescue a phase that is silently slow (FlashInfer's
            # compile shells out and prints nothing for the whole 88 minutes).
            # It converts that from a two-hour blind wait into a ten-minute
            # failure that names the last thing the server said.
            start = time.monotonic()
            hard_cap = start + LAUNCH_HARD_CAP_S
            deadline = start + LAUNCH_TIMEOUT_S
            last_size, last_note = -1, start
            while True:
                now = time.monotonic()
                if proc.poll() is not None:
                    raise LaunchError(f"exited {proc.returncode} during startup",
                                      err.read_text()[-3000:])
                try:
                    if httpx.get(f"{self.base_url}/health", timeout=2).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass

                size = err.stat().st_size if err.exists() else 0
                if size != last_size:                 # still talking -> still working
                    last_size = size
                    deadline = max(deadline, now + STALL_S)

                if now > deadline or now > hard_cap:
                    why = ("stopped producing output" if now > deadline
                           else f"exceeded the hard cap of {LAUNCH_HARD_CAP_S/60:.0f} min")
                    raise LaunchError(
                        f"launch {why} after {(now-start)/60:.0f} min.\n"
                        f"        last line: {_last_line(err)}",
                        err.read_text()[-3000:])

                # A long launch should not look like a frozen one.
                if now - last_note > 120:
                    last_note = now
                    self.log(f"        still starting ({(now-start)/60:.0f} min) "
                             f"-- {_last_line(err)[:110]}")
                time.sleep(2)
            yield model
        finally:
            if proc.poll() is None:
                os.killpg(os.getpgid(proc.pid), signal.SIGINT)
                try:
                    proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    proc.wait(timeout=30)

    def _metrics(self) -> dict:
        try:
            text = httpx.get(f"{self.base_url}/metrics", timeout=10).text
        except httpx.HTTPError:
            return {}
        raw: dict[str, float] = {}
        for line in text.splitlines():
            if line and not line.startswith("#"):
                name = line.split("{")[0].split(" ")[0]
                try:
                    raw[name] = raw.get(name, 0.0) + float(line.rsplit(" ", 1)[1])
                except (ValueError, IndexError):
                    pass
        g = lambda *ks: next((raw[k] for k in ks if k in raw), None)
        hits, qs = g("vllm:prefix_cache_hits", "vllm:prefix_cache_hits_total"), \
                   g("vllm:prefix_cache_queries", "vllm:prefix_cache_queries_total")
        acc, drafts = g("vllm:spec_decode_num_accepted_tokens"), g("vllm:spec_decode_num_draft_tokens")
        return {k: v for k, v in {
            "kv_cache_util": g("vllm:kv_cache_usage_perc"),
            "preemptions": g("vllm:num_preemptions", "vllm:num_preemptions_total"),
            "prefix_hit_rate": (hits / qs) if hits is not None and qs else None,
            "spec_acceptance_rate": (acc / drafts) if acc is not None and drafts else None,
        }.items() if v is not None}

    # --- probes ---
    def _equivalence(self, model: str) -> float | None:
        """Compare only the first K tokens. Whole-output comparison has a 19%
        false-positive floor at the measured flip rate; K is derived from it."""
        probe = self.prompts[:84]
        reqs = asyncio.run(self._greedy(model, probe, self.equiv_k))
        got = [r.text.strip()[:self.equiv_k * 4] for r in reqs]
        if self.equiv_ref is None:
            self.equiv_ref = got
            return 0.0
        n = min(len(got), len(self.equiv_ref))
        return sum(1 for a, b in zip(self.equiv_ref[:n], got[:n]) if a != b) / max(1, n)

    async def _greedy(self, model, prompts, max_tokens) -> list[Req]:
        sem = asyncio.Semaphore(32)
        async with httpx.AsyncClient(timeout=900.0) as c:
            async def go(p):
                async with sem:
                    return await _one(c, self.base_url, model, p, max_tokens, stream=False)
            return list(await asyncio.gather(*[go(p) for p in prompts]))

    # --- capacity ---
    def serving_metrics(self, model, concurrency: int, *, el=lambda: "",
                        cursor: list[int] | None = None, warmup: bool = True):
        """THE serving measurement. One implementation, used by everything.

        Open loop at the trace's arrival rate with a semaphore cap of
        `concurrency`, a fixed WINDOW_S window, REPEATS passes, aggregated
        direction-aware -- min for goodput and throughput, MAX for TTFT and ITL,
        so every metric is reported at its least flattering observed value.

        Extracted because eval_repro.py had a second, DIFFERENT measurement
        bolted on: a single closed-loop point with no warmup, no repeats and no
        direction-aware aggregation, described in its own comment as "directly
        comparable" to a traversal. It was not. Two implementations of the same
        measurement diverge -- that is how the prompt-building bug got in, and
        how a number gets reported against a claim it does not support.

        Returns (aggregated, per-pass) so a caller can report the worst case and
        still see the spread behind it.
        """
        cursor = [0] if cursor is None else cursor
        if warmup:
            self.log(f"        {el()} warming up {WARMUP_S:.0f}s")
            asyncio.run(_load(self.base_url, model, self.prompts, self.max_tokens,
                              self.qps, self.conc, WARMUP_S, cursor=cursor))
        passes = []
        for i in range(REPEATS):
            reqs, t0, t1, offered, started = asyncio.run(
                _load(self.base_url, model, self.prompts, self.max_tokens,
                      self.qps, concurrency, WINDOW_S, cursor=cursor))
            m = summarize(reqs, t0, t1, self.slo)
            m["concurrency"] = concurrency
            m["offered"], m["started"] = offered, started
            passes.append(m)
            self.log(f"        {el()} pass {i+1}/{REPEATS}  "
                     f"goodput {m['goodput']:7.1f} tok/s "
                     f"({m['goodput_req_s']:.2f} req/s)  "
                     f"thru {m['throughput']:7.1f}  "
                     f"ttft_p99 {m['ttft_p99_ms']:6.0f}ms  "
                     f"itl_p99 {m['itl_p99_ms']:6.1f}ms  "
                     f"slo {m['slo_attainment']:.0%}  "
                     f"({m['completed']} ok/{m['failed']} fail, "
                     f"{started}/{offered} started)")
        med = aggregate(passes)
        med["concurrency"] = concurrency
        spread = ((max(p["goodput"] for p in passes) - min(p["goodput"] for p in passes))
                  / max(1e-9, abs(min(p["goodput"] for p in passes))))
        med["pass_spread"] = spread
        if spread > 0.10:
            self.log(f"        {el()} NOTE pass-to-pass goodput spread {spread:.0%} "
                     f"at L={concurrency} -- larger than the accept band, so a "
                     f"keep/revert decision here is not resolvable at this sample size")
        return med, passes

    def _point(self, model, L: int, el, label: str = "",
               cursor: list[int] | None = None) -> dict:
        """One closed-loop measurement at concurrency L on a live server.

        Settle and window are a single continuous run so the window observes a
        pipeline that is already full -- see _closed_loop.
        """
        reqs, t0, t1 = asyncio.run(_closed_loop(
            self.base_url, model, self.prompts, self.max_tokens, L,
            getattr(self, "settle_s", SETTLE_S), SWEEP_WINDOW_S, cursor=cursor))
        m = summarize(reqs, t0, t1, self.slo)
        m["concurrency"] = L
        self.log(f"        {el()} L={L:<4d} goodput {m['goodput']:7.1f} tok/s "
                 f"({m['goodput_req_s']:.2f} req/s)  thru {m['throughput']:7.1f}  "
                 f"ttft_p99 {m['ttft_p99_ms']:6.0f}ms  slo {m['slo_attainment']:.0%}  "
                 f"({m['completed']} done){'  ' + label if label else ''}")
        return m

    @staticmethod
    def peak(curve: list[dict]) -> dict:
        """The operating point: highest goodput on the curve.

        Goodput already encodes the SLO -- requests that miss the deadline
        contribute nothing -- so its maximum IS the SLO-constrained capacity.
        There is no separate 'find where TTFT crosses 500ms' step; that crossing
        is what bends the curve over.
        """
        return max(curve, key=lambda m: m["goodput"])

    def capacity(self, config: dict, tag: str) -> tuple[list[dict], dict]:
        """Launch `config` and sweep it across the full range.

        Used for the frontier finalists, where the winner's own curve is what
        sets capacity and therefore the replica count. Delegates to measure() so
        the sweep, the peak selection and the extension logic have exactly one
        implementation -- there used to be a second inline copy here, which is
        how two independently-derived operating points could disagree.
        """
        t = self.measure(config, probes=["goodput"], benchmarks=[],
                         node_id=tag, levels=SWEEP_LEVELS)
        curve = t.curve or []
        pk = max(curve, key=lambda m: m["goodput"]) if curve else {
            "concurrency": t.concurrency, "goodput": t.goodput}
        return curve, pk

    # --- the Evaluator protocol ---
    def measure(self, config: dict[str, Any], *, probes: list[str],
                benchmarks: list[str], node_id: str,
                concurrency: int | None = None,
                levels: tuple[int, ...] | list[int] | None = None,
                fixed_concurrency: int | None = None) -> Trial:
        """Measure one config.

        `concurrency` is the operating point found by the stage 1.3 sweep. Every
        node is measured there rather than at an arbitrary offered load -- run
        four judged everything at L=30, a number produced by `int(qps*2)`, and
        the middle of its frontier was uninterpretable as a result.

        Every node spans a bracket and is scored on its PEAK, which also handles
        the families whose curves cross rather than sitting uniformly above or
        below the incumbent's -- chunked_prefill is negative at low L (chunking a
        prefill that could run in one shot is overhead) and positive at high L
        (it stops a long prefill blocking every decode behind it); speculative
        decoding is the mirror. Neither needs a special case once the peak is
        what gets compared.
        """
        tag = f"{node_id}-{abs(hash(json.dumps(config, sort_keys=True, default=str))) % 10**8:08d}"
        t_start = time.time()
        el = lambda: f"+{(time.time()-t_start)/60:4.1f}m"
        changed = {k: v for k, v in config.items() if k != "model"}
        self.log(f"        {el()} launching  {json.dumps(changed, default=str)[:88]}")
        # Reset per node, advanced across the phases WITHIN a node. Paired
        # comparison between configs is preserved; replay within a node is not.
        cursor = [0]
        try:
            with self._serve(config, tag) as model:
                self.log(f"        {el()} healthy, warming up {WARMUP_S:.0f}s")
                asyncio.run(_load(self.base_url, model, self.prompts,
                                  self.max_tokens, self.qps, self.conc, WARMUP_S,
                                  cursor=cursor))

                if fixed_concurrency:
                    # RUN-FOUR REPRODUCTION MODE.
                    #
                    # The exact instrument run four used: open loop at the
                    # trace's arrival rate with a semaphore cap, a fixed 45s
                    # window, REPEATS passes, no bracket and no sweep. Kept as a
                    # first-class mode rather than a historical curiosity --
                    # when the production arrival rate IS known, measuring at it
                    # is the right thing to do, and reproducing a previous
                    # measurement is how you tell a code change from a real one.
                    #
                    # Under saturation the semaphore is always full, so this is
                    # close to closed loop at the same L; the difference is that
                    # the window opens on a drained server rather than a settled
                    # one.
                    conc = fixed_concurrency
                    pts = []
                    med, passes = self.serving_metrics(
                        model, conc, el=el, cursor=cursor, warmup=False)
                    diag = self._metrics()
                    div = self._equivalence(model) if "equivalence" in probes else None
                    if div is not None:
                        self.log(f"        {el()} equivalence  {div:.1%} of "
                                 f"first-{self.equiv_k}-token prefixes differ")
                    qual = {}
                    if "quality" in probes and benchmarks:
                        from quality import resolution, run_benchmark
                        for b in benchmarks:
                            qual[b] = run_benchmark(
                                b, lambda ps, mt: asyncio.run(self._greedy(model, ps, mt)),
                                max_input_tokens=config.get("max_model_len"),
                                model=self.fp.model.id)
                            self.log(f"        {el()} {b:20s} {qual[b]:.4f}  "
                                     f"(+/- {resolution(b):.1%} resolution)")
                    mem = self._gpu_memory_gb()
                    self.log(f"        {el()} done, tearing down")
                    return Trial(
                        node_id=node_id, config=dict(config),
                        goodput=round(med["goodput"], 1),
                        ttft_p99_ms=round(med["ttft_p99_ms"], 1),
                        itl_p99_ms=round(med["itl_p99_ms"], 2),
                        memory_gb=mem, quality=qual, equivalence_divergence=div,
                        concurrency=conc, curve=[],
                        diagnostics={**diag,
                                     "slo_attainment": round(med["slo_attainment"], 3),
                                     "throughput": round(med["throughput"], 1),
                                     "goodput_req_s": round(med.get("goodput_req_s", 0.0), 3),
                                     "throughput_req_s": round(med.get("throughput_req_s", 0.0), 3),
                                     "completed": med["completed"], "failed": med["failed"],
                                     "mode": "fixed_concurrency_open_loop"},
                        slo_ok=med["goodput"] > 0)

                # BRACKET THE PEAK, do not measure at a fixed L.
                #
                # Measuring every node at one concurrency is what run five got
                # wrong. L* was found by sweeping the SEED -- a config with
                # chunked_prefill off and 2598-token prompts, so concurrent
                # prefills block decode, TTFT crosses the SLO as soon as L rises,
                # and the sweep terminated at L=4. Every later node was then
                # measured at a 4 x 9.2 = 37 tok/s ceiling and came back 30.9,
                # 30.8, 30.8 -- indistinguishable, and 6x below what the same
                # configs measured at L=30 in run four.
                #
                # The error is conceptual, not arithmetic. What these techniques
                # DO is raise the concurrency the server can sustain. Pinning
                # them all at the worst config's peak guarantees none of them can
                # show it. So each node is measured across a bracket and scored
                # on its PEAK: a config that sustains more concurrency wins by
                # reaching a higher point, which is exactly the property being
                # optimised.
                # `levels` overrides the bracket with an explicit sweep. Stage
                # 1.3 uses it to characterise the seed across the full geometric
                # range in the SAME launch that measures the baseline -- there
                # used to be a second launch of the identical config just to
                # sweep it, which cost ~9 minutes and produced a second, possibly
                # disagreeing answer for the operating point.
                base_L = concurrency or self.conc
                use = list(levels) if levels else sorted(
                    {max(2, base_L // 2), base_L, base_L * 2})
                pts: list[dict] = []
                for L in use:
                    pts.append(self._point(model, L, el, cursor=cursor))

                # If the best sits at an endpoint the bracket did not contain the
                # peak; walk outward rather than reporting a boundary as a
                # maximum. Two steps is enough to cross an octave in each
                # direction and bounds the cost.
                for _ in range(3):
                    best_i = max(range(len(pts)), key=lambda i: pts[i]["goodput"])
                    if best_i == len(pts) - 1:
                        nxt = pts[-1]["concurrency"] * 2
                        if nxt > 1024:
                            break
                        self.log(f"        {el()} peak at the top of the bracket, extending to L={nxt}")
                        pts.append(self._point(model, nxt, el, cursor=cursor))
                    elif best_i == 0 and pts[0]["concurrency"] > 2:
                        nxt = max(2, pts[0]["concurrency"] // 2)
                        self.log(f"        {el()} peak at the bottom of the bracket, extending to L={nxt}")
                        pts.insert(0, self._point(model, nxt, el, cursor=cursor))
                    else:
                        break

                peak = max(pts, key=lambda m: m["goodput"])
                conc = peak["concurrency"]
                self.log(f"        {el()} peak goodput {peak['goodput']:.1f} tok/s at L={conc}")

                # A second pass at the peak only. The bracket points establish
                # WHERE the peak is; the repeat establishes how noisy it is, and
                # only the peak's noise matters for the keep/revert gate.
                passes = [peak, self._point(model, conc, el, label="repeat", cursor=cursor)]
                cand = max(pts, key=lambda m: m["goodput"])
                if cand["concurrency"] != conc:
                    conc = cand["concurrency"]
                    passes = [cand, self._point(model, conc, el, label="repeat", cursor=cursor)]

                # MIN across the peak's passes, not median and definitely not max.
                #
                # This was `sorted(...)[len(passes)//2]`, named `med` for median
                # -- but two samples have no median, and index 1 of two is the
                # LARGER. Combined with `best = max(variants)` in traverse, a
                # 2-variant node scored as the max of four draws, sitting ~1.5-2
                # sigma above its true mean. At the 2-4% spread measured on this
                # rig that is +3-6%, against a 5% accept band: a node with no real
                # effect could clear the bar on noise alone, and then raise the
                # incumbent for everything after it.
                #
                # The gate asks "is this reliably better", so the conservative
                # estimate is the honest one. A config that wins on its worst pass
                # has actually won. `concurrency` is excluded -- taking the min of
                # two identical L values is meaningless and it is carried on the
                # Trial separately.
                med = aggregate(passes)
                spread = ((max(p["goodput"] for p in passes) - min(p["goodput"] for p in passes))
                          / max(1e-9, abs(min(p["goodput"] for p in passes))))

                # A node whose two passes at the SAME concurrency disagree by more
                # than the accept band cannot be decided by that band: the
                # difference being tested is smaller than the difference between
                # two runs of the identical config.
                if spread > 0.10:
                    self.log(f"        {el()} NOTE pass-to-pass goodput spread {spread:.0%} "
                             f"at L={conc} -- larger than the accept band, so a "
                             f"keep/revert decision here is not resolvable at this "
                             f"sample size")

                diag = self._metrics()
                div = None
                if "equivalence" in probes:
                    div = self._equivalence(model)
                    self.log(f"        {el()} equivalence  {div:.1%} of first-{self.equiv_k}-token "
                             f"prefixes differ from the reference")
                qual = {}
                if "quality" in probes and benchmarks:
                    from quality import resolution, run_benchmark
                    for b in benchmarks:
                        qual[b] = run_benchmark(
                            b, lambda ps, mt: asyncio.run(self._greedy(model, ps, mt)),
                            max_input_tokens=config.get("max_model_len"),
                                model=self.fp.model.id)
                        self.log(f"        {el()} {b:20s} {qual[b]:.4f}  "
                                 f"(+/- {resolution(b):.1%} resolution at this sample size)")
                mem = self._gpu_memory_gb()
                self.log(f"        {el()} done, tearing down")
        except LaunchError as e:
            self.log(f"        launch failed: {e}")
            (self.run_dir / "launches" / tag / "why.txt").write_text(f"{e}\n\n{e.stderr}")
            return Trial(node_id=node_id, config=dict(config), goodput=0.0,
                         ttft_p99_ms=float("inf"), itl_p99_ms=float("inf"),
                         memory_gb=0.0, slo_ok=False,
                         diagnostics={"launch_error": str(e), "stderr_tail": e.stderr[-1200:]})

        return Trial(node_id=node_id, config=dict(config),
                     goodput=round(med["goodput"], 1),
                     ttft_p99_ms=round(med["ttft_p99_ms"], 1),
                     itl_p99_ms=round(med["itl_p99_ms"], 2),
                     memory_gb=mem, quality=qual, equivalence_divergence=div,
                     # Both of these were declared on the dataclass, documented,
                     # and never assigned -- so every trial recorded null and run
                     # five's operating point had to be back-computed from
                     # throughput instead of read off the record.
                     concurrency=conc,
                     curve=[{k: v for k, v in m.items()
                             if isinstance(v, (int, float))} for m in pts],
                     diagnostics={**diag, "slo_attainment": round(med["slo_attainment"], 3),
                                  "throughput": round(med["throughput"], 1),
                                  "goodput_req_s": round(med.get("goodput_req_s", 0.0), 3),
                                  "throughput_req_s": round(med.get("throughput_req_s", 0.0), 3),
                                  "completed": med["completed"], "failed": med["failed"]},
                     slo_ok=med["goodput"] > 0)

    def _gpu_memory_gb(self) -> float:
        try:
            out = subprocess.run(["nvidia-smi", "-i", self.gpu,
                                  "--query-compute-apps=used_memory",
                                  "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=15).stdout
            return round(sum(float(x) for x in out.split() if x.strip().isdigit()) / 1024, 1)
        except Exception:
            return 0.0
