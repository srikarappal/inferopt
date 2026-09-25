"""Load generation against an OpenAI-compatible completions endpoint.

Moved out of inferopt.evaluator unchanged. Two drivers, differing in what they
hold fixed, which is what decides the right instrument for a question:

    _load         fixes the ARRIVAL RATE and lets concurrency emerge. Matches
                  production, where users arrive when they arrive. Above
                  capacity it does not converge, so a 45s window and a 90s
                  window give different answers.
    _closed_loop  fixes CONCURRENCY and lets the rate emerge. Bounded by
                  construction, so it converges and repeats. Right for finding
                  where goodput peaks, wrong for validating an operating point,
                  because holding L constant removes the burstiness that moves
                  the TTFT tail.

Sweep closed, validate open.

The settle-into-window design in _closed_loop is load bearing and is the thing
most other harnesses get wrong: measurement must flow continuously out of
warmup, never restart from an idle server, or the window opens with every
worker firing at once and the first L requests all queue behind each other.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field

import httpx

from goodput.slo import LatencyTarget as SLO


@dataclass
class Req:
    start: float = 0.0
    ttft: float | None = None
    latency: float = 0.0
    n_out: int = 0
    # Input length as the SERVER counted it, from the usage block, not from a
    # local tokenizer and not from whatever the trace claimed. Those three
    # disagree often enough to matter: a trace's recorded length was produced by
    # some other tokenizer at some other time, and a chat template adds tokens
    # nobody counted. 0 means the server did not report it.
    n_in: int = 0
    ok: bool = False
    error: str = ""
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


def _mt(max_tokens, idx: int) -> int:
    """Resolve the output length for request `idx`.

    Accepts a scalar for the callers that genuinely want one -- the equivalence
    probe compares token-identical outputs and the accuracy benchmarks have
    their own budgets -- and a per-request sequence for load replay.
    """
    if isinstance(max_tokens, (list, tuple)):
        return int(max_tokens[idx % len(max_tokens)]) if max_tokens else 1
    return int(max_tokens)


# Greedy and seeded, so two servers given the same prompt can be compared token
# for token. Not every server takes it: vLLM refuses temperature and seed on a
# diffusion LM (DiffusionGemma) outright, and the evaluator clears this before
# launching one. Module state rather than an argument, because the four load
# paths and their callers would otherwise all have to carry it.
SAMPLING: dict = {"temperature": 0.0, "seed": 0}


def use_sampling(engine: str, decoding: str) -> dict:
    """The sampling parameters a server of this engine and decoding accepts."""
    global SAMPLING
    SAMPLING = {} if (engine == "vllm" and decoding == "diffusion") else {"temperature": 0.0, "seed": 0}
    return SAMPLING


async def _one(client, base_url, model, prompt, max_tokens, stream=True) -> Req:
    r = Req(start=time.perf_counter())
    payload = {"model": model, "prompt": prompt, "max_tokens": max_tokens,
               **SAMPLING, "stream": stream}
    if stream:
        # continuous_usage_stats: usage in EVERY chunk, not only the last. A
        # chunk is not a token. An autoregressive server sends one token per
        # chunk and the two coincide; a diffusion LM commits a whole block per
        # forward pass and SGLang streams the block as one chunk, so counting
        # chunks read a 34 tokens/s server as 1.3. With usage per chunk each
        # one is credited with the tokens it carried. Both vLLM and SGLang
        # accept the field; a server that ignores it falls back to one per
        # chunk below, which is exact for the autoregressive case.
        payload["stream_options"] = {"include_usage": True,
                                     "continuous_usage_stats": True}
    try:
        if not stream:
            resp = await client.post(f"{base_url}/v1/completions", json=payload, timeout=900.0)
            resp.raise_for_status()
            j = resp.json()
            r.text = j["choices"][0]["text"]
            r.n_out = (j.get("usage") or {}).get("completion_tokens", 0)
            r.n_in = (j.get("usage") or {}).get("prompt_tokens", 0)
            r.latency = time.perf_counter() - r.start
            r.ttft, r.ok = r.latency, True
            return r
        async with client.stream("POST", f"{base_url}/v1/completions",
                                 json=payload, timeout=900.0) as resp:
            if resp.status_code != 200:
                r.error = f"HTTP {resp.status_code}"
                return r
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                body = line[6:].strip()
                if body == "[DONE]":
                    break
                ch = json.loads(body)
                usage = ch.get("usage") or {}
                if ch.get("choices") and ch["choices"][0].get("text"):
                    now = time.perf_counter()
                    if r.ttft is None:
                        r.ttft = now - r.start
                    r.text += ch["choices"][0]["text"]
                    # Tokens this chunk carried: the usage delta when the
                    # server reports usage per chunk, else one.
                    reported = usage.get("completion_tokens")
                    carried = (reported - r.n_out) if reported else 1
                    r.n_out += max(1, carried)
                    r.token_times.extend([now] * max(1, carried))
                if usage:
                    r.n_out = usage.get("completion_tokens") or r.n_out
                    r.n_in = usage.get("prompt_tokens") or r.n_in
        r.latency = time.perf_counter() - r.start
        r.ok = r.ttft is not None
    except Exception as e:
        # The reason, not just the fact. Every failure mode -- a 400 from an
        # over-length generation, a dropped connection, a server that died --
        # collapsed into ok=False with nothing to tell them apart, so a
        # systematic refusal looked exactly like flaky networking.
        r.error = f"{type(e).__name__}: {e}"[:200]
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
        async def go(p, mt):
            async with sem:
                live.add(asyncio.current_task())
                out.append(await _one(c, base_url, model, p, mt))
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
            j = (base + i) % len(prompts)
            tasks.append(asyncio.create_task(go(prompts[j], _mt(max_tokens, j))))
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
                       cursor: list[int] | None = None, stagger_s: float = 0.0):
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

    WORKERS ARE STAGGERED AND EVERY REQUEST CARRIES ITS OWN LENGTH. Without
    both, the driver does not measure a pipeline at all. Starting all L workers
    at one instant is what created the lockstep; giving every request the same
    max_tokens is what made it permanent, because identical durations mean the
    workers that started together also finish together, forever. The load became
    a CONVOY of L requests arriving at once every request-duration. Measured directly at L=128, arrivals peaked at 10.5-14.9x
    their mean rate, and goodput across two identical drives came out 205.1 and
    2181.6 tok/s -- a 10.6x spread from the alignment of the window against the
    convoy. Staggering the first request over one request-duration gave 1891.0
    and 1893.8, a 1.00x spread, with TTFT p99 248 vs 258ms and 100% attainment
    both times. At L=256 the spread went 1.42x -> 1.01x.

    That convoy is also why L=32 measured WORSE than L=64 (673-987ms vs
    149-165ms) -- no capacity argument permits that, but phase alignment does.

    The stagger is capped at the settle interval so every worker is in flight
    before the window opens; the offsets then persist on their own, because a
    worker re-fires only when its own request returns.

    Note the convoy is not purely an artifact: batching L prefills at once uses
    the GPU harder, so locked measured ~8.5 starts/s against dephased ~7.2. It
    is a real throughput/latency trade -- but it is one no production arrival
    process produces, so optimising against it optimised for the wrong workload.

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
            if stagger_s > 0:
                await asyncio.sleep(min(stagger_s, settle_s) * slot / max(1, conc))
            i = slot
            while time.perf_counter() < t1:
                # Same reason as _load: phases must not replay the same prompts,
                # or a warm prefix cache scores full-prompt hits that no
                # production workload would produce.
                issued[0] += 1
                j = (base + i) % len(prompts)
                # Prompt and length come from the SAME row. Pairing them is the
                # point: a 4431-token prompt in this trace does not ask for a
                # 102-token answer, and splitting them would replay a workload
                # that never existed.
                out.append(await _one(c, base_url, model, prompts[j],
                                      _mt(max_tokens, j)))
                i += conc
        await asyncio.gather(*[asyncio.create_task(worker(k)) for k in range(conc)],
                             return_exceptions=True)
    if cursor is not None:
        cursor[0] = base + issued[0]
    return out, t0, t1
