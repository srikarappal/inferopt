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
import gzip
import hashlib
import json
import os
import re
import signal
import shutil
import subprocess
import threading
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from inferopt.calibration import STORE
from inferopt import leaderboard
from inferopt.engines import VllmEngine, engine_for
from inferopt.fingerprint import SLO, Fingerprint
from inferopt.legality import repair
from inferopt.quality import context_needed
from goodput.driver import Req, _closed_loop, _load, _mt, _one
from goodput.metrics import _reasons, summarize
from inferopt.traverse import Trial

_VLLM_CMD: list[str] | None = None


# The vLLM engine, for the callers that still ask this module directly. The
# resolution order that used to live here (INFEROPT_VLLM_CMD, the console
# script on PATH, then under sys.prefix, then python -m the entry point) is
# engines._resolve now, shared with SGLang.
vllm_engine = VllmEngine()


def vllm_cmd() -> list[str]:
    """How to invoke vLLM, resolved rather than assumed. See engines._resolve."""
    found = vllm_engine.command()
    if found:
        return found
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


def hardware_defaults(fp, engine=None) -> dict:
    """Flags this (model, GPU) pair REQUIRES to run at all, not tuning choices.

    Lives in one place because it has to apply to every path that launches a
    server. It did not: moe_backend was set only in run.py's seed_config, so
    the traversal got it and eval_repro did not, and a stock-baseline run on
    Qwen3-30B-A3B went down FlashInfer's sm120 CUTLASS path and hung for 30
    minutes JIT-compiling kernels. A hardware fact expressed in one caller is
    a bug waiting for the second caller.

    What the flags are is the engine's to say (engines.py); which engine is the
    fingerprint's. Callers merge these UNDER their own settings, so an explicit
    value always wins.
    """
    return (engine or engine_for(fp)).defaults(fp)


def _vllm_version() -> str:
    return vllm_engine.version()


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
# Slack left between prompt + generation and max_model_len. Tokenizer counts in
# a trace are not always the server's counts (chat templates, added specials),
# and being a few tokens over is a 400, not a truncation.
CONTEXT_MARGIN_TOKENS = 32
SETTLE_MAX_S = 75.0    # ceiling, so a slow model cannot make the sweep unbounded
SWEEP_WINDOW_S = 45.0
SWEEP_LEVELS = (4, 8, 16, 32, 64, 128, 256)
WINDOW_S = 45.0        # open-loop window, used by --fixed-concurrency
REPEATS = 2

# Which direction is "worse" for each metric. Aggregating passes with a blanket
# min() is conservative for goodput and OPTIMISTIC for latency -- it reports the
# better of two TTFT samples, which is exactly backwards for a gate that decides
# whether an SLO was met.
LOWER_IS_BETTER = {"ttft_p99_ms", "itl_p99_ms", "ttft_p95_ms", "itl_p95_ms",
                   "failed", "window_s"}
# ttft_n is a SAMPLE COUNT, not a metric -- but aggregate() takes the least
# flattering value of everything numeric, and fewer samples is the weaker
# claim, so it belongs on the min side with goodput. That is where the
# default already puts it; noted so nobody "fixes" it into the set above.


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


def installed_flags() -> set[str]:
    """vLLM's flag list, from its own --help=all. See Engine.installed_flags."""
    return vllm_engine.installed_flags(env=child_env())


def to_cli(cfg: dict) -> list[str]:
    """A config as vLLM flags. See Engine.generic_flags for the rules, and
    VllmEngine.INT_FLAGS for why a count that arrives as a float is rounded:
    dag/llm.json once computed max_num_seqs as 384.0 and both retune_batching
    nodes died on every run they were applicable to, invisibly, because a dead
    launch reads as a configuration that did not help."""
    return vllm_engine.translate(cfg)


# --------------------------------------------------------------------------
# request-level measurement
#
# MOVED TO THE `goodput` PACKAGE, imported at the top of this file.
#
# The load generators and the goodput maths depend on nothing in this project:
# not the DAG, the fingerprint, the quantizer, or vLLM. Keeping them here made
# them unusable by anything else, and the measurement bugs this project paid
# for -- the convoy, the constant output lengths, the drain inside the window --
# are not inferopt bugs, they are load-generation bugs that any harness can
# make. They are re-exported below so eval_repro and the tests keep importing
# them from this module.
# --------------------------------------------------------------------------



# --------------------------------------------------------------------------
# evaluator
# --------------------------------------------------------------------------

class VllmEvaluator:
    """Launches a server per config, measures it, tears it down.

    Named for the engine it was written against. The engine is a parameter
    now (engines.py) and defaults to what the fingerprint needs, so a masked
    diffusion LM lands on SGLang without the caller knowing the difference.
    """

    def __init__(self, fp: Fingerprint, slo: SLO, trace_path: str, run_dir: str,
                 gpu: str = "0", port: int = 8000, log=print, engine=None):
        self.fp, self.slo, self.log = fp, slo, log
        self.engine = engine or engine_for(fp)
        self.gpu, self.port, self.run_dir = gpu, port, Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.trace_path = trace_path       # AWQ/NVFP4 calibrate on this workload
        rows = [json.loads(l) for l in open(trace_path) if l.strip()]
        replay = [r for r in rows if r.get("prompt")]
        self.prompts = [r["prompt"] for r in replay]
        if not self.prompts:
            raise ValueError(
                f"{trace_path} has no 'prompt' field. The fingerprint can be built from "
                f"token counts alone, but the benchmark needs the actual text to replay."
            )
        # The MEAN is still the right number for sizing -- settle length, the
        # equivalence probe, anything that needs one figure. It was the wrong
        # number to serve, and serving it is what this replaces.
        #
        # trace_shared.jsonl carries output_tokens with mean 259.6 and sd 167.7,
        # p10..p90 spanning 102..466. That entire distribution used to collapse
        # to int(mean) = 259 for every request, one line after `replay` was built
        # with the per-row lengths still in scope. Constant durations are what
        # let the closed loop stay in lockstep -- see _closed_loop -- so the
        # stagger there only holds the workers apart, while this is what stops
        # them re-converging. It also means the tail is finally being measured:
        # a 466-token request and a 102-token one behave differently, and at a
        # single constant length neither was.
        self.max_tokens = int(fp.workload.mean_output_tokens)
        self.out_tokens = [max(1, int(r.get("output_tokens") or self.max_tokens))
                           for r in replay]
        self.in_tokens = [max(0, int(r.get("input_tokens") or 0)) for r in replay]

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
        # Kept separately: settle_s is clamped at both ends, so it cannot be
        # divided back out to recover the estimate the stagger needs.
        self.one_request_s = one_request_s
        self.qps = fp.workload.request_rate_qps
        self.conc = fp.workload.max_concurrency
        cal = STORE.get(fp)
        self.equiv_k = cal.equivalence_prefix_tokens() if cal else 8
        self.equiv_ref: list[str] | None = None
        self.base_url = f"http://{HOST}:{port}"
        # Set by a caller that wants to resume. Every method funnels through
        # measure(), so this one hook covers the DAG walk, the screen and yolo
        # rather than each growing its own resume logic.
        self.replay: dict | None = None

    def _score(self, model, config: dict, benchmarks, tag: str, el) -> dict:
        """Score every requested benchmark, dispatching by who owns it.

        TWO KINDS, MEASURED DIFFERENTLY ON PURPOSE.

        Ours (math_500, mbpp_plus, humaneval_plus) run through _greedy with this
        project's prompts, budgets and graders. They are internally consistent,
        which is all a walk needs to RANK configurations, and externally
        unquotable, because nobody can reproduce a number whose harness they do
        not have.

        leaderboard_* are handed to lm-evaluation-harness against the endpoint
        already serving, so the number means the same thing as anyone else's. It
        costs a subprocess and a second environment, and it buys a score that
        survives being asked where it came from.

        A LEADERBOARD TASK THAT FAILS IS RECORDED AS None, NOT AS ZERO. A gated
        dataset and a model that scores nothing are different facts, and a
        quality gate reading 0.0 would reject a configuration for a missing
        HuggingFace licence.
        """
        from inferopt.quality import resolution, run_benchmark
        qual: dict = {}
        d = self.run_dir / "launches" / tag
        for b in benchmarks:
            if b in leaderboard.TASKS:
                # max_length bounds prompt+generation and MUST fit the served
                # context, or every long-shot prompt comes back as an HTTP 400
                # that reads like a bad model rather than a bad ceiling.
                ceiling = int(config.get("max_model_len") or 0) or 8192
                try:
                    res = leaderboard.run(
                        [b], self.base_url, self.fp.model.id,
                        max_length=max(512, ceiling - CONTEXT_MARGIN_TOKENS),
                        out_dir=d, log=self.log)
                    row = res[b]
                    qual[b] = row.get("score")
                    d.mkdir(parents=True, exist_ok=True)
                    (d / f"leaderboard-{b}.json").write_text(
                        json.dumps(row, indent=2, default=str))
                except leaderboard.LeaderboardError as e:
                    qual[b] = None
                    self.log(f"        {el()} {b:20s} SKIPPED: {str(e).splitlines()[0]}")
                continue
            qual[b] = run_benchmark(
                b, lambda ps, mt: asyncio.run(self._greedy(model, ps, mt)),
                max_input_tokens=config.get("max_model_len"),
                model=self.fp.model.id,
                record=d / f"generations-{b}.jsonl")
            self.log(f"        {el()} {b:20s} {qual[b]:.4f}  "
                     f"(+/- {resolution(b):.1%} resolution at this sample size)")
        return qual

    def export_run(self, reqs: list[Req], t0: float, t1: float, *,
                   node_id: str, concurrency: int, label: str = "",
                   server_metrics: dict | None = None) -> None:
        """Write this measurement in AIPerf's schema, for anyone checking us.

        requests.jsonl.gz is ours and is compact on purpose; this is the same
        measurement in the format the rest of the field reads, so a reviewer can
        point their own tooling at it instead of taking our word. One directory
        per (node, concurrency), because those are different measurements.

        NEVER FATAL. A measurement that succeeded and then died writing its
        paperwork is a measurement lost for no reason, and these artifacts are
        evidence about a run, not part of it.
        """
        from goodput import export
        try:
            export.write_run(
                self.run_dir / "exports" / (node_id or "unknown") / f"L{concurrency}",
                reqs, t0, t1, self.slo,
                config=getattr(self, "_serving_config", None),
                run_info={"node_id": node_id, "concurrency": concurrency,
                          "label": label, "model": self.fp.model.id,
                          "trace": Path(self.trace_path).name,
                          **(getattr(self, "stamp", None) or {})},
                server_metrics=server_metrics,
                prompts=self.prompts, output_lengths=self.replay_lengths())
        except Exception as e:
            self.log(f"        export skipped ({type(e).__name__}: {e})")

    def dump_requests(self, reqs: list[Req], t0: float, t1: float, *,
                      node_id: str, concurrency: int, phase: str) -> None:
        """Persist the PER-REQUEST record every aggregate was collapsed from.

        summarize() reduces a few hundred requests to a dozen numbers and the
        requests are then dropped. That makes the SLO unchangeable after the
        fact: goodput counts only requests that individually met the bound, so
        re-asking "what if TTFT were 300ms instead of 500" needs the requests
        back, and no p99 can supply them. Every run so far has to be repeated
        in full to answer a question it already had the data for.

        Six numbers per request are enough to reconstruct summarize() exactly at
        ANY threshold -- start, ttft, latency, output tokens, tokens landing
        inside the window, and ok. Not token_times: those are ~260 floats a
        request and buy only sub-request timing, which no SLO here asks about.
        meets() uses the MEAN inter-token latency, (latency-ttft)/(n_out-1), and
        that is recoverable from what is stored.

        Rows outside the window are kept, with a negative `s`. summarize()
        counts their tokens toward throughput while excluding them from the
        started set, and a reader that dropped them would not match.

        Appended as gzip members, ~25KB per measurement point before
        compression, so a 20-launch screen costs a couple of MB.
        """
        path = self.run_dir / "requests.jsonl.gz"
        rows = [{"_": "meta", "node": node_id, "L": concurrency, "phase": phase,
                 "win_s": round(t1 - t0, 4), "n": len(reqs),
                 "slo": {"ttft_p99_ms": self.slo.ttft_p99_ms,
                         "itl_p99_ms": self.slo.itl_p99_ms}}]
        for r in reqs:
            rows.append({
                "s": round(r.start - t0, 4),
                "t": round(r.ttft * 1e3, 3) if r.ttft is not None else None,
                "l": round(r.latency * 1e3, 3),
                "n": r.n_out,
                "w": sum(1 for tt in r.token_times if t0 <= tt < t1),
                "k": bool(r.ok),
                **({"e": r.error} if r.error else {}),
            })
        try:
            with gzip.open(path, "at") as fh:
                fh.write("".join(json.dumps(r) + "\n" for r in rows))
        except Exception as e:                 # never lose a run over telemetry
            self.log(f"        (could not write {path.name}: {type(e).__name__}: {e})")

    def replay_lengths(self) -> list[int]:
        """Per-request output lengths, clamped to the context actually served.

        A constant 259 could never overflow; the real lengths can. The longest
        output in this trace is 1464 tokens and the longest prompt 4431, and a
        config serving max_model_len 2048 would take both and return HTTP 400 --
        which `_one` swallows into ok=False, so the requests would silently
        vanish from the window instead of failing loudly. Clamping keeps the
        length distribution as close to the trace as the config permits and
        never manufactures a request the server will refuse.
        """
        want_all = getattr(self, "out_tokens", None)
        if not want_all:                      # a subclass that set prompts by hand
            return [self.max_tokens] * len(self.prompts)
        in_all = getattr(self, "in_tokens", None) or [0] * len(want_all)
        cap = getattr(self, "_served_max_len", 0)
        if not cap:
            return list(want_all)
        out = []
        for want, inp in zip(want_all, in_all):
            room = cap - inp - CONTEXT_MARGIN_TOKENS
            out.append(max(1, min(want, room)) if room > 0 else 1)
        return out

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
            from inferopt.quantize import ensure_variant
            # A conversion that fails is a launch that fails: recorded as
            # goodput 0 on this node and the walk moves on. Left as its own
            # exception it escaped the walk, and a run that had measured seven
            # nodes died at the eighth because modelopt could not load the
            # checkpoint.
            try:
                path = ensure_variant(self.fp, kind, self.trace_path, log=self.log)
            except Exception as failed:
                raise LaunchError(f"quantization to {kind} failed: "
                                  f"{str(failed).splitlines()[0]}", str(failed)) from failed
            if path:
                config["model"] = path
            else:
                config["quantization"] = kind      # fp8: a load-time flag

        engine = self.engine
        if engine.name == "vllm":
            # The MoE backend lists this reads are vLLM's own.
            reconcile_moe_backend(config, log=self.log)

        model = config.get("model") or self.fp.model.id
        # THE FLAG CHECK MUST NOT DISABLE ITSELF. It used to fall back to "[]"
        # whenever the flag list came back empty, which is precisely the case
        # where it is most needed: an old build whose --help is not recognised
        # returns nothing, validation silently switches off, and every config is
        # passed blind to a binary that may not accept it. The launch then dies
        # with "exited 1 during startup" and no indication that a flag was the
        # reason.
        # Only demand the flag list when there is something to check against it.
        # A config with no keys beyond `model` cannot contain an unknown flag, so
        # refusing to launch it because --help was unreadable would block a bare
        # `serve <model>` for no reason. The check is on the flags the engine
        # would actually pass, after translation, not on the DAG's names.
        to_check = engine.flag_names(config)
        flags = engine.installed_flags(env=child_env()) if to_check else None
        if to_check and not flags and os.environ.get("INFEROPT_SKIP_FLAG_CHECK"):
            flags = None                        # explicit opt-out, launch blind
        elif to_check and not flags:
            raise LaunchError(
                f"could not read this {engine.name}'s flag list, so no config can "
                f"be validated before launching.\n"
                f"`{' '.join(engine.help_argv())}` produced nothing parseable.\n"
                f"Set INFEROPT_SKIP_FLAG_CHECK=1 to launch anyway and let the "
                f"engine reject bad flags itself, at the cost of a failed launch "
                f"per bad key instead of an instant error.")
        unknown = sorted(k for k in to_check if k not in flags) if flags else []
        if unknown:
            raise LaunchError(
                f"flags this {engine.name} ({engine.version()}) does not accept: "
                f"{unknown}. Flags move between releases -- vLLM 0.26 removed "
                f"--disable-log-requests, --swap-space and --cuda-graph-sizes, "
                f"and older builds take speculative decoding as flat flags "
                f"rather than --speculative-config JSON.")

        # What replay_lengths clamps against. Read from the finalized config,
        # after reconciliation, so it is what the server was actually given.
        self._served_max_len = int(config.get("max_model_len") or 0)
        # The FINALIZED config -- after quantization resolved to an artifact path
        # and after the MoE backend reconciliation -- because that is what was
        # actually served. Recording the caller's dict instead would describe a
        # launch that did not happen.
        self._serving_config = dict(config)

        d = self.run_dir / "launches" / tag
        d.mkdir(parents=True, exist_ok=True)
        err = d / "server.log"
        cmd = engine.serve_argv(model, HOST, self.port, config, workdir=d)
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
                    if httpx.get(f"{self.base_url}{engine.health_path}", timeout=2).status_code == 200:
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
        return self._parse_metrics(text)

    # GAUGES MUST BE SAMPLED WHILE LOAD IS ON THE SERVER. This was called only
    # after serving_metrics returned -- that is, after the load had drained --
    # so counters survived and gauges did not. kv_cache_usage_perc is a gauge,
    # and it read exactly 0.000 in 32 of 32 trials across three independent
    # search methods, which is not "the cache was empty" but "we measured it at
    # the one moment it is guaranteed to be empty".
    #
    # The cost was not the number itself. It is that the whole class of
    # KV-pressure reasoning had no data: retune_batching_after_kv exists to
    # re-tune batching once quantization frees KV, and the signal telling it
    # whether KV is the constraint was dead.
    def sample_gauges(self) -> dict:
        """Instantaneous metrics, for calling DURING a measurement window."""
        import httpx
        try:
            return self._parse_metrics(
                httpx.get(f"{self.base_url}/metrics", timeout=5).text)
        except Exception:
            return {}

    # Which reduction each metric takes across label sets. A Prometheus name is
    # not one number: vLLM exposes one series per (model, engine), and with
    # tensor or pipeline parallelism, or more than one served model, there are
    # several. The old parser added them all up, which is right for a counter
    # and nonsense for a fraction -- two engines at 0.5 KV utilisation summed to
    # 1.0, reporting a full cache on a half-empty server. Nothing caught it
    # because every run so far has been single-engine, single-model.
    # vLLM's table, kept on the class for the callers and tests that read it
    # unbound. An instance reads its engine's.
    _REDUCE = VllmEngine.REDUCE

    @staticmethod
    def parse_prometheus(text: str) -> dict[str, list[tuple[dict, float]]]:
        """Every series, labels intact, nothing collapsed.

        Reducing at parse time destroys the only evidence that a reduction was
        even needed. This keeps the granularity and leaves the choice to the
        caller, which is the whole point: the caller knows whether it is holding
        a counter or a gauge and the parser does not.
        """
        out: dict[str, list[tuple[dict, float]]] = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            head, _, val = line.rpartition(" ")
            if not head:
                continue
            name, _, rest = head.partition("{")
            labels: dict[str, str] = {}
            if rest:
                body = rest.rstrip("}")
                # Split on commas that are not inside a quoted value. Label
                # values may legitimately contain commas -- model paths do.
                part, quoted = "", False
                for ch in body:
                    if ch == '"':
                        quoted = not quoted
                    if ch == "," and not quoted:
                        k, _, v = part.partition("=")
                        if k:
                            labels[k.strip()] = v.strip().strip('"')
                        part = ""
                    else:
                        part += ch
                if part:
                    k, _, v = part.partition("=")
                    if k:
                        labels[k.strip()] = v.strip().strip('"')
            try:
                out.setdefault(name.strip(), []).append((labels, float(val)))
            except ValueError:
                continue
        return out

    def _parse_metrics(self, text: str) -> dict:
        """Derived scalars, each reduced the way ITS metric type requires.

        `series` travels with them, so a reader can always see what was
        collapsed and how many series there were. `multi_series` names any
        metric that had more than one -- on a single-engine run it is empty,
        and when it is not, a reduction was actually exercised and is worth
        looking at rather than trusting.
        """
        series = self.parse_prometheus(text)
        # getattr, not self.engine: a test calls this unbound with the class as
        # self, and subclasses that bypass __init__ are a normal thing here.
        engine = getattr(self, "engine", None) or vllm_engine
        table = engine.REDUCE

        def red(name: str):
            vals = [v for _, v in series.get(name, [])]
            if not vals:
                return None
            return max(vals) if table.get(name) == "max" else sum(vals)

        out = engine.derive(series, red)
        # Only the metrics the scalars above are built from -- the full scrape
        # includes every histogram bucket the engine exposes and would bloat
        # each trial record by orders of magnitude for data nothing reads.
        kept = {n: [{"labels": lb, "value": v} for lb, v in series[n]]
                for n in table if n in series}
        if kept:
            out["series"] = kept
        multi = sorted(n for n, vs in series.items()
                       if n in table and len(vs) > 1)
        if multi:
            out["multi_series"] = multi
        return out

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
    # KV utilisation is a GAUGE and preemptions is a COUNTER, and they must not
    # be treated the same way. `vllm:num_preemptions_total` only ever increases,
    # so keeping its maximum across a window returns the cumulative count since
    # the server started -- including warmup and every earlier sweep level. The
    # curve then shows a monotone staircase that reads as "preemptions rise with
    # concurrency" when every preemption may have happened at L=4.
    #
    # So: peak for the gauge, last-minus-first for the counter.
    _COUNTERS = ("preemptions",)
    _GAUGES = ("kv_cache_util",)

    def _gauge_watch(self, stop, out: dict, period: float = 2.0) -> None:
        """Poll instantaneous metrics until `stop` is set, keeping the PEAK.

        Peak, not mean: the question a gauge answers here is "did this
        configuration ever run out of KV", and an average over a window that
        includes ramp-up hides exactly the moment that matters.
        """
        import time as _t
        first: dict[str, float] = {}
        while not stop.is_set():
            g = self.sample_gauges()
            for k in self._GAUGES:
                v = g.get(k)
                if v is not None:
                    out[k] = max(out.get(k, 0.0), float(v))
            for k in self._COUNTERS:
                v = g.get(k)
                if v is not None:
                    first.setdefault(k, float(v))
                    # Counters can reset if the server restarts mid-window; a
                    # negative delta would be nonsense, so floor at the latest
                    # reading rather than reporting it.
                    out[k] = max(0.0, float(v) - first[k])
            stop.wait(period)

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
            asyncio.run(_load(self.base_url, model, self.prompts, self.replay_lengths(),
                              self.qps, self.conc, WARMUP_S, cursor=cursor))
        passes = []
        for i in range(REPEATS):
            reqs, t0, t1, offered, started = asyncio.run(
                _load(self.base_url, model, self.prompts, self.replay_lengths(),
                      self.qps, concurrency, WINDOW_S, cursor=cursor))
            m = summarize(reqs, t0, t1, self.slo)
            m["concurrency"] = concurrency
            m["offered"], m["started"] = offered, started
            passes.append(m)
            self.dump_requests(reqs, t0, t1, node_id=f"pass{i+1}",
                               concurrency=concurrency, phase="open_loop")
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
               cursor: list[int] | None = None, node_id: str = "") -> dict:
        """One closed-loop measurement at concurrency L on a live server.

        Settle and window are a single continuous run so the window observes a
        pipeline that is already full -- see _closed_loop.
        """
        # Gauges are sampled ACROSS the window and the peak kept. Read after
        # the load drains -- which is what happened until now -- an
        # instantaneous gauge reports the idle value, and kv_cache_usage_perc
        # read exactly 0.000 in 32 of 32 trials across three search methods.
        # Peak rather than mean, because the question is whether this config
        # ever ran out of KV, and an average over the ramp hides that.
        peak: dict = {}
        stop = threading.Event()
        w = threading.Thread(target=self._gauge_watch, args=(stop, peak), daemon=True)
        w.start()
        try:
            reqs, t0, t1 = asyncio.run(_closed_loop(
                self.base_url, model, self.prompts, self.replay_lengths(), L,
                getattr(self, "settle_s", SETTLE_S), SWEEP_WINDOW_S, cursor=cursor,
                stagger_s=getattr(self, "one_request_s", 0.0)))
        finally:
            stop.set(); w.join(timeout=5)
        m = summarize(reqs, t0, t1, self.slo)
        m["concurrency"] = L
        m.update(peak)                 # kv_cache_util / preemptions, measured live
        # node_id, not `label`. A run directory holds one requests.jsonl.gz for
        # every node it measured -- 20 rows of a screen, 10 nodes of a walk --
        # and writing "sweep" for all of them made the file unattributable: the
        # points were there and nothing said which configuration produced them.
        self.dump_requests(reqs, t0, t1, node_id=node_id or "unknown",
                           concurrency=L, phase=f"closed_loop{'/' + label if label else ''}")
        self.export_run(reqs, t0, t1, node_id=node_id or "unknown", concurrency=L,
                        label=label, server_metrics=peak)
        self.log(f"        {el()} L={L:<4d} goodput {m['goodput']:7.1f} tok/s "
                 f"({m['goodput_req_s']:.2f} req/s)  thru {m['throughput']:7.1f}  "
                 f"ttft_p99 {m['ttft_p99_ms']:6.0f}ms  slo {m['slo_attainment']:.0%}  "
                 f"({m['completed']} done"
                 + (f", {m['failed']} FAILED: "
                    + "; ".join(f"{k} x{v}" for k, v in m["failure_reasons"].items())
                    if m["failed"] else "")
                 + f"){'  ' + label if label else ''}")
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
        # peak(), not a second copy of it. This function's own docstring warns
        # that a duplicated operating-point selection is how two answers come to
        # disagree, and it contained one.
        pk = self.peak(curve) if curve else {
            "concurrency": t.concurrency, "goodput": t.goodput}
        return curve, pk

    # --- the Evaluator protocol ---
    def measure(self, config: dict[str, Any], *, probes: list[str],
                benchmarks: list[str], node_id: str,
                concurrency: int | None = None,
                levels: tuple[int, ...] | list[int] | None = None,
                fixed_concurrency: int | None = None) -> Trial:
        """Measure one config, scoring quality where the benchmark has room.

        Goodput is always measured under `config` exactly. Quality is scored in
        the same launch when the served context holds the benchmark, and under
        a second launch with the context raised when it does not.
        """
        wants_quality = "quality" in probes and bool(benchmarks)
        roomy = self._config_with_room(config, benchmarks) if wants_quality else None
        trial = self._measure_served(
            config, probes=probes, benchmarks=[] if roomy else benchmarks,
            node_id=node_id, concurrency=concurrency, levels=levels,
            fixed_concurrency=fixed_concurrency)
        # A second launch is not spent on a trial that came from the journal, or
        # on a config already out on goodput: neither can use the score.
        if roomy is None or (trial.diagnostics or {}).get("replayed") or not trial.slo_ok:
            return trial
        return self._score_relaunched(trial, roomy, benchmarks, node_id)

    def _config_with_room(self, config: dict, benchmarks: list[str]) -> dict | None:
        """`config` with the context raised until `benchmarks` fit, or None when
        they already do.

        max_model_len_rightsize sizes the context to the TRAFFIC, and a
        benchmark is not traffic: MATH-500 generates 1024 tokens, so under a
        right-sized 1024 nothing can be scored and a walk died at
        lossless_complete with two hours of launches behind it. Flooring the
        right-sized value at the benchmark's need would let the instrument set
        the deployed config. Quality is a property of the weights and the KV
        dtype, not of max_model_len, so it gets its own launch and goodput
        keeps the tight one.
        """
        served = int(config.get("max_model_len") or 0)
        if not served:
            return None
        need = context_needed(benchmarks, self.fp.model.id) + CONTEXT_MARGIN_TOKENS
        if served >= need:
            return None
        raised = min(-(-need // 1024) * 1024, self.fp.model.max_model_len)
        if raised <= served:
            # The model itself has no more context to give. Score in place and
            # let run_benchmark say so, which it does precisely.
            return None
        return repair({**config, "max_model_len": raised}, log=self.log)[0]

    def _score_relaunched(self, trial: Trial, roomy: dict,
                          benchmarks: list[str], node_id: str) -> Trial:
        """Fill in `trial.quality` from a launch that has room to score it."""
        tag = self._launch_tag(f"{node_id}-quality", roomy)
        t_start = time.time()
        el = lambda: f"+{(time.time()-t_start)/60:4.1f}m"
        self.log(f"        {el()} relaunching at max_model_len={roomy['max_model_len']} "
                 f"to score {', '.join(benchmarks)}: "
                 f"{trial.config.get('max_model_len')} fits the traffic, not the benchmark")
        try:
            with self._serve(roomy, tag) as model:
                trial.quality = self._score(model, roomy, benchmarks, tag, el)
        except LaunchError as e:
            # NOT KEEPABLE. The walk skips the quality gate when a trial carries
            # no scores, so a lossy config whose quality could not be measured
            # would otherwise be kept ungated.
            self.log(f"        quality launch failed: {e}")
            trial.slo_ok = False
            trial.diagnostics = {**(trial.diagnostics or {}),
                                 "quality_launch_error": str(e)}
            return trial
        trial.diagnostics = {**(trial.diagnostics or {}),
                             "quality_max_model_len": roomy["max_model_len"]}
        return trial

    def _launch_tag(self, node_id: str, config: dict) -> str:
        # sha256, not hash(). Python randomises string hashing per process
        # (PYTHONHASHSEED), so the same config produced a different launch
        # directory on every invocation and the directories could not be
        # correlated across runs. Same digest family as provenance.trial_stamp.
        return (f"{node_id}-"
                + hashlib.sha256(
                    json.dumps(config, sort_keys=True, default=str).encode()
                ).hexdigest()[:8])

    def _measure_served(self, config: dict[str, Any], *, probes: list[str],
                        benchmarks: list[str], node_id: str,
                        concurrency: int | None = None,
                        levels: tuple[int, ...] | list[int] | None = None,
                        fixed_concurrency: int | None = None) -> Trial:
        """Measure one config under one launch.

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
        # getattr, not self.replay: subclasses that bypass __init__ to fake a
        # server are a normal thing here, and one of them has done so since
        # before this hook existed. Same defensive read as settle_s below.
        replay = getattr(self, "replay", None)
        if replay is not None:
            from inferopt.resume import key
            hit = replay.get(key(node_id, config))
            if hit is not None:
                from inferopt.resume import to_trial
                t = to_trial(hit)
                t.diagnostics = {**(t.diagnostics or {}), "replayed": True}
                self.log(f"        replayed from journal, no launch spent "
                         f"({t.goodput:.1f} tok/s)")
                return t

        tag = self._launch_tag(node_id, config)
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
                                  self.replay_lengths(), self.qps, self.conc, WARMUP_S,
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
                    peak: dict = {}
                    _stop = threading.Event()
                    _w = threading.Thread(target=self._gauge_watch,
                                          args=(_stop, peak), daemon=True)
                    _w.start()
                    try:
                        med, passes = self.serving_metrics(
                            model, conc, el=el, cursor=cursor, warmup=False)
                    finally:
                        _stop.set(); _w.join(timeout=5)
                    # The live peak wins over the post-drain read for gauges;
                    # counters are identical either way.
                    diag = {**self._metrics(), **peak}
                    div = self._equivalence(model) if "equivalence" in probes else None
                    if div is not None:
                        self.log(f"        {el()} equivalence  {div:.1%} of "
                                 f"first-{self.equiv_k}-token prefixes differ")
                    qual = {}
                    if "quality" in probes and benchmarks:
                        qual = self._score(model, config, benchmarks, tag, el)
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
                                     "ttft_p95_ms": round(med.get("ttft_p95_ms", float("nan")), 1),
                                     "itl_p95_ms": round(med.get("itl_p95_ms", float("nan")), 2),
                                     "ttft_n": med.get("ttft_n", 0),
                                     "failure_reasons": passes[0].get("failure_reasons") or {},
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
                    pts.append(self._point(model, L, el, cursor=cursor, node_id=node_id))

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
                        pts.append(self._point(model, nxt, el, cursor=cursor, node_id=node_id))
                    elif best_i == 0 and pts[0]["concurrency"] > 2:
                        nxt = max(2, pts[0]["concurrency"] // 2)
                        self.log(f"        {el()} peak at the bottom of the bracket, extending to L={nxt}")
                        pts.insert(0, self._point(model, nxt, el, cursor=cursor, node_id=node_id))
                    else:
                        break

                peak = max(pts, key=lambda m: m["goodput"])
                conc = peak["concurrency"]
                self.log(f"        {el()} peak goodput {peak['goodput']:.1f} tok/s at L={conc}")

                # A second pass at the peak only. The bracket points establish
                # WHERE the peak is; the repeat establishes how noisy it is, and
                # only the peak's noise matters for the keep/revert gate.
                passes = [peak, self._point(model, conc, el, label="repeat", cursor=cursor,
                                    node_id=node_id)]

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

                # Counters from the post-run scrape, gauges from the live peak
                # across every level in the sweep -- the highest KV pressure any
                # operating point reached is the number that matters.
                diag = dict(self._metrics())
                for k in self._GAUGES:            # the highest pressure any level reached
                    vals = [p[k] for p in pts if p.get(k) is not None]
                    if vals:
                        diag[k] = max(vals)
                for k in self._COUNTERS:          # per-level deltas, so they ADD
                    vals = [p[k] for p in pts if p.get(k) is not None]
                    if vals:
                        diag[k] = sum(vals)
                div = None
                if "equivalence" in probes:
                    div = self._equivalence(model)
                    self.log(f"        {el()} equivalence  {div:.1%} of first-{self.equiv_k}-token "
                             f"prefixes differ from the reference")
                qual = {}
                if "quality" in probes and benchmarks:
                    qual = self._score(model, config, benchmarks, tag, el)
                mem = self._gpu_memory_gb()
                self.log(f"        {el()} done, tearing down")
        except LaunchError as e:
            self.log(f"        launch failed: {e}")
            # mkdir first. The launch directory is created when a server is
            # actually started, so a failure BEFORE that -- a rejected flag,
            # an unreadable flag list -- had nowhere to write its explanation
            # and raised FileNotFoundError from the error handler, replacing a
            # precise LaunchError with a confusing one about a missing file.
            wd = self.run_dir / "launches" / tag
            wd.mkdir(parents=True, exist_ok=True)
            (wd / "why.txt").write_text(f"{e}\n\n{getattr(e, 'stderr', '')}")
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
                                  "completed": med["completed"], "failed": med["failed"],
                                  # summarize() emits these; the whitelist below is
                                  # what reaches the Trial, and until they were added
                                  # here every record carried p95 nan and n=0.
                                  "ttft_p95_ms": round(med.get("ttft_p95_ms", float("nan")), 1),
                                  "itl_p95_ms": round(med.get("itl_p95_ms", float("nan")), 2),
                                  "ttft_n": med.get("ttft_n", 0),
                                  "failure_reasons": passes[0].get("failure_reasons") or {}},
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
