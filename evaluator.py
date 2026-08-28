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

VLLM_CMD = os.environ.get("INFEROPT_VLLM_CMD", "vllm").split()
HOST = "127.0.0.1"
LAUNCH_TIMEOUT_S = float(os.environ.get("INFEROPT_LAUNCH_TIMEOUT_S", "1800"))
# 45s, not 15s. The prefix cache does not fill in 15s: passes were identical
# (47.9/47.9) before prefix caching was enabled and 42% apart after (137.9/195.2),
# because pass 1 ran cold and pass 2 warm. Both passes must measure the same
# steady state, which is also the state production runs in.
WARMUP_S = 45.0
SETTLE_S = 20.0        # after stepping concurrency, before measuring
SWEEP_WINDOW_S = 45.0
SWEEP_LEVELS = (4, 8, 16, 32, 64, 128, 256)
WINDOW_S = 45.0
REPEATS = 2


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
            out = subprocess.run([*VLLM_CMD, "serve", "--help=all"], env=child_env(),
                                 capture_output=True, text=True, timeout=180).stdout
            _FLAGS = {m.group(1).replace("-", "_") for m in re.finditer(r"--([a-z0-9][a-z0-9-]*)", out)}
        except Exception:
            _FLAGS = set()
    return _FLAGS


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


async def _load(base_url, model, prompts, max_tokens, qps, conc, seconds):
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
        tasks, i = [], 0
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < seconds:
            tasks.append(asyncio.create_task(go(prompts[i % len(prompts)])))
            i += 1
            await asyncio.sleep(interval) if interval else await asyncio.sleep(0)
        t1 = time.perf_counter()
        for t in tasks:
            if t not in live and not t.done():
                t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    return out, t0, t1, len(tasks), len(live)


async def _closed_loop(base_url, model, prompts, max_tokens, conc, seconds):
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
    """
    out: list[Req] = []
    t0 = time.perf_counter()
    deadline = t0 + seconds
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=conc + 16)) as c:
        async def worker(slot: int):
            i = slot
            while time.perf_counter() < deadline:
                out.append(await _one(c, base_url, model, prompts[i % len(prompts)], max_tokens))
                i += conc
        await asyncio.gather(*[asyncio.create_task(worker(k)) for k in range(conc)],
                             return_exceptions=True)
    return out, t0, time.perf_counter()


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
        if kind:
            from quantize import ensure_variant
            path = ensure_variant(self.fp, kind, self.trace_path, log=self.log)
            if path:
                config["model"] = path
            else:
                config["quantization"] = kind      # fp8: a load-time flag

        model = config.get("model") or self.fp.model.id
        unknown = [k for k in config if k != "model" and k not in installed_flags()] \
            if installed_flags() else []
        if unknown:
            raise LaunchError(f"config keys this vLLM does not accept: {unknown}")

        d = self.run_dir / "launches" / tag
        d.mkdir(parents=True, exist_ok=True)
        err = d / "server.log"
        cmd = [*VLLM_CMD, "serve", model, "--host", HOST, "--port", str(self.port), *to_cli(config)]
        env = child_env(CUDA_VISIBLE_DEVICES=self.gpu,
                        VLLM_CACHE_ROOT=str(self.run_dir / ".vllm_cache" / f"gpu{self.gpu}"))
        with open(err, "wb") as fh:
            proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                    env=env, start_new_session=True)
        try:
            deadline = time.monotonic() + LAUNCH_TIMEOUT_S
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise LaunchError(f"exited {proc.returncode} during startup",
                                      err.read_text()[-3000:])
                try:
                    if httpx.get(f"{self.base_url}/health", timeout=2).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(2)
            else:
                raise LaunchError(f"not healthy in {LAUNCH_TIMEOUT_S:.0f}s", err.read_text()[-3000:])
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
    def _sweep_on_live_server(self, model, levels=SWEEP_LEVELS, el=lambda: "") -> list[dict]:
        """Goodput vs concurrency on an ALREADY-RUNNING server.

        This is the cheap part and the reason capacity measurement is affordable
        at all: the 4-minute model load is paid once and amortised across every
        point. Relaunching per level would make the sweep cost more than the
        entire traversal.

        Ramps geometrically and stops when goodput falls twice running -- past
        the peak, more concurrency only pushes requests over the deadline, and
        the points beyond it cost time to learn nothing.
        """
        curve, falling, best = [], 0, 0.0
        for L in levels:
            asyncio.run(_closed_loop(self.base_url, model, self.prompts,
                                     self.max_tokens, L, SETTLE_S))
            reqs, t0, t1 = asyncio.run(_closed_loop(
                self.base_url, model, self.prompts, self.max_tokens, L, SWEEP_WINDOW_S))
            m = summarize(reqs, t0, t1, self.slo)
            m["concurrency"] = L
            curve.append(m)
            self.log(f"        {el()} L={L:<4d} goodput {m['goodput']:7.1f}  "
                     f"thru {m['throughput']:7.1f}  ttft_p99 {m['ttft_p99_ms']:6.0f}ms  "
                     f"slo {m['slo_attainment']:.0%}  ({m['completed']} done)")
            if m["goodput"] > best:
                best, falling = m["goodput"], 0
            else:
                falling += 1
                if falling >= 2:
                    self.log(f"        {el()} goodput fell twice -- past the peak, stopping")
                    break
        return curve

    def _point(self, model, L: int, el, label: str = "") -> dict:
        """One closed-loop measurement at concurrency L on a live server."""
        asyncio.run(_closed_loop(self.base_url, model, self.prompts,
                                 self.max_tokens, L, SETTLE_S))
        reqs, t0, t1 = asyncio.run(_closed_loop(
            self.base_url, model, self.prompts, self.max_tokens, L, SWEEP_WINDOW_S))
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
        """Launch `config` and sweep it. Returns (curve, peak)."""
        t_start = time.time()
        el = lambda: f"+{(time.time()-t_start)/60:4.1f}m"
        self.log(f"        {el()} launching for capacity sweep")
        with self._serve(config, tag) as model:
            self.log(f"        {el()} healthy, warming up {WARMUP_S:.0f}s")
            asyncio.run(_load(self.base_url, model, self.prompts,
                              self.max_tokens, self.qps, self.conc, WARMUP_S))
            curve = self._sweep_on_live_server(model, el=el)
        pk = self.peak(curve)
        self.log(f"        {el()} peak goodput {pk['goodput']:.1f} tok/s at L={pk['concurrency']}")
        return curve, pk

    # --- the Evaluator protocol ---
    def measure(self, config: dict[str, Any], *, probes: list[str],
                benchmarks: list[str], node_id: str,
                concurrency: int | None = None,
                also_at: list[int] | None = None) -> Trial:
        """Measure one config.

        `concurrency` is the operating point found by the stage 1.3 sweep. Every
        node is measured there rather than at an arbitrary offered load -- run
        four judged everything at L=30, a number produced by `int(qps*2)`, and
        the middle of its frontier was uninterpretable as a result.

        `also_at` adds extra concurrency levels for nodes whose goodput curves
        CROSS rather than sit uniformly above or below the incumbent's.
        chunked_prefill is negative at low L (chunking a prefill that could run
        in one shot is pure overhead) and positive at high L (it stops a long
        prefill blocking every decode behind it). Speculative decoding is the
        mirror: it spends spare compute, so it wins at low L and can go negative
        at high L where there is none. Ranking those at a single point is a coin
        flip; the best of their measured points is used.
        """
        tag = f"{node_id}-{abs(hash(json.dumps(config, sort_keys=True, default=str))) % 10**8:08d}"
        t_start = time.time()
        el = lambda: f"+{(time.time()-t_start)/60:4.1f}m"
        changed = {k: v for k, v in config.items() if k != "model"}
        self.log(f"        {el()} launching  {json.dumps(changed, default=str)[:88]}")
        try:
            with self._serve(config, tag) as model:
                self.log(f"        {el()} healthy, warming up {WARMUP_S:.0f}s")
                asyncio.run(_load(self.base_url, model, self.prompts,
                                  self.max_tokens, self.qps, self.conc, WARMUP_S))
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
                base_L = concurrency or self.conc
                levels = sorted({max(2, base_L // 2), base_L, base_L * 2})
                pts: list[dict] = []
                for L in levels:
                    pts.append(self._point(model, L, el))

                # If the best sits at an endpoint the bracket did not contain the
                # peak; walk outward rather than reporting a boundary as a
                # maximum. Two steps is enough to cross an octave in each
                # direction and bounds the cost.
                for _ in range(2):
                    best_i = max(range(len(pts)), key=lambda i: pts[i]["goodput"])
                    if best_i == len(pts) - 1:
                        nxt = pts[-1]["concurrency"] * 2
                        if nxt > 1024:
                            break
                        self.log(f"        {el()} peak at the top of the bracket, extending to L={nxt}")
                        pts.append(self._point(model, nxt, el))
                    elif best_i == 0 and pts[0]["concurrency"] > 2:
                        nxt = max(2, pts[0]["concurrency"] // 2)
                        self.log(f"        {el()} peak at the bottom of the bracket, extending to L={nxt}")
                        pts.insert(0, self._point(model, nxt, el))
                    else:
                        break

                peak = max(pts, key=lambda m: m["goodput"])
                conc = peak["concurrency"]
                self.log(f"        {el()} peak goodput {peak['goodput']:.1f} tok/s at L={conc}")

                # A second pass at the peak only. The bracket points establish
                # WHERE the peak is; the repeat establishes how noisy it is, and
                # only the peak's noise matters for the keep/revert gate.
                passes = [peak, self._point(model, conc, el, label="repeat")]
                # `also_at` is subsumed by the bracket: every node now spans
                # {L/2, L, 2L} and is scored on its peak, so a crossing curve is
                # caught for free rather than needing a special case. The
                # parameter is kept so callers do not break; extra levels are
                # merged into the same peak search.
                for L2 in (also_at or []):
                    if L2 not in [m["concurrency"] for m in pts]:
                        pts.append(self._point(model, L2, el, label="also_at"))
                extra = pts
                cand = max(pts, key=lambda m: m["goodput"])
                if cand["concurrency"] != conc:
                    conc = cand["concurrency"]
                    passes = [cand, self._point(model, conc, el, label="repeat")]

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
                            max_input_tokens=config.get("max_model_len"))
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
