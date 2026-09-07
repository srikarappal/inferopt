"""Stage 3, turn 3 -- is the TTFT tail the SERVER, or the load driver's own convoy?

Turn 2 ran one configuration three times and got TTFT p99 of 1040, 300 and 279ms
at L=128, and SLO attainment of 0.67, 0.93 and 1.00 at L=256. Those are not
tails of one distribution. They are BIMODAL: every launch lands in a ~300ms /
100% mode or a ~1100ms / 68% mode, with nothing in between. Gaussian measurement
noise does not do that. Something discrete is flipping.

THE SUSPECT IS THE INSTRUMENT, not the server. `_closed_loop` starts all L
workers at the same instant and gives every request the same max_tokens, so the
workers finish together, re-fire together, and stay in lockstep. The load is not
L requests in steady flight; it is a CONVOY of L requests arriving at once,
every one_request_s. In a convoy of 256 simultaneous prefills the last one
served waits behind 255 others, so a third of the convoy blows the 500ms TTFT
bar -- which is exactly the 0.68 attainment. Whether the 45s window happens to
open on a convoy head or between convoys is a coin flip, and only ~2-3 convoys
fit in the window, so the coin is flipped 2-3 times per launch. That produces
bimodality, and it also explains the non-monotonicity turn 2 left unexplained:
L=32 was WORSE than L=64 (673-987ms vs 149-165ms), which no capacity argument
allows but phase alignment does.

If that is right, the fix is free: stagger the workers' first request so the
convoy spreads into a pipeline. No longer window, no extra launches.

ONE server launch. Four drives per arm, two arms, L in {128, 256}. Nothing in
inferopt/ is modified -- the dephased driver is a local copy, so this turn can
only produce evidence, not a regression. Per-request arrival and TTFT are
recorded so the convoy can be seen directly rather than inferred from p99.
"""
import asyncio
import json
import statistics
import time
from pathlib import Path

import httpx

from inferopt.evaluator import SWEEP_WINDOW_S, VllmEvaluator, _one, summarize
from inferopt.methods import setup

BEST = {
    "gpu_memory_utilization": 0.75, "max_num_seqs": 256, "max_model_len": 6144,
    "enable_prefix_caching": True, "enable_chunked_prefill": False,
    "enforce_eager": False,
}
LEVELS = [128, 256]
REPS = 2
OUT = Path("runs/stage3-turn3")


async def drive(base_url, model, prompts, max_tokens, conc, settle_s, window_s,
                stagger_s: float, cursor: list[int]):
    """`_closed_loop`, plus an optional per-worker start offset.

    stagger_s=0 reproduces the shipped driver exactly. stagger_s>0 spreads the
    workers' FIRST request over that interval; after that the loop is
    self-clocking and the offsets persist, because each worker re-fires only
    when its own request returns.
    """
    out, issued, base = [], [0], cursor[0]
    start = time.perf_counter()
    t0 = start + settle_s
    t1 = t0 + window_s
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=conc + 16)) as c:
        async def worker(slot: int):
            if stagger_s:
                await asyncio.sleep(stagger_s * slot / conc)
            i = slot
            while time.perf_counter() < t1:
                issued[0] += 1
                out.append(await _one(c, base_url, model,
                                      prompts[(base + i) % len(prompts)], max_tokens))
                i += conc
        await asyncio.gather(*[asyncio.create_task(worker(k)) for k in range(conc)],
                             return_exceptions=True)
    cursor[0] = base + issued[0]
    return out, t0, t1


def burstiness(reqs, t0, t1) -> dict:
    """How lumpy were the ARRIVALS? This is the convoy, measured directly.

    A pipeline holding L in flight with mean service time S should start about
    L/S requests per second, evenly. A convoy starts L of them at once and then
    nothing for S seconds. peak_1s/mean_1s separates the two without needing to
    know S.
    """
    starts = sorted(r.start for r in reqs if t0 <= r.start < t1)
    if len(starts) < 8:
        return {}
    span = max(1e-9, t1 - t0)
    bins = [0] * max(1, int(span))
    for s in starts:
        bins[min(len(bins) - 1, int(s - t0))] += 1
    mean = sum(bins) / len(bins)
    gaps = [b - a for a, b in zip(starts, starts[1:])]
    mg = sum(gaps) / len(gaps)
    return {
        "peak_1s": max(bins),
        "mean_1s": mean,
        "burst_ratio": max(bins) / mean if mean else 0.0,
        # CV of inter-arrival gaps: 0 = perfectly even, >1 = clustered.
        "gap_cv": (statistics.pstdev(gaps) / mg) if mg else 0.0,
    }


def head_penalty(reqs, t0, t1) -> float:
    """TTFT of requests arriving in the busiest second, over the quietest.

    If the tail is the server saturating, this is ~1: every request is equally
    slow. If it is a convoy, the requests that arrive with 200 siblings wait
    behind them and this is large.
    """
    inw = [r for r in reqs if t0 <= r.start < t1 and r.ok and r.ttft is not None]
    if len(inw) < 16:
        return float("nan")
    bins: dict[int, list] = {}
    for r in inw:
        bins.setdefault(int(r.start - t0), []).append(r.ttft)
    if len(bins) < 2:
        return float("nan")
    busiest = max(bins.values(), key=len)
    quietest = min(bins.values(), key=len)
    q = statistics.median(quietest)
    return statistics.median(busiest) / q if q else float("nan")


fp, slo, ctx = setup("Qwen/Qwen3-1.7B", "data/trace_shared.jsonl", 500, 250, qps=16)
ev = VllmEvaluator(fp, slo, "data/trace_shared.jsonl", str(OUT))
OUT.mkdir(parents=True, exist_ok=True)

# The convoy period is one request duration -- spread the workers over exactly
# that and consecutive workers are one inter-arrival apart, which is the
# pipeline the closed loop was supposed to be.
one_request_s = ev.settle_s / 1.2
print(f"\n  settle {ev.settle_s:.0f}s   window {SWEEP_WINDOW_S:.0f}s   "
      f"est. request {one_request_s:.0f}s   stagger {one_request_s:.0f}s")

rows = []
with ev._serve(BEST, "turn3") as model:
    for L in LEVELS:
        for arm, stagger in (("locked", 0.0), ("dephased", one_request_s)):
            for rep in range(REPS):
                cursor = [0]
                reqs, t0, t1 = asyncio.run(drive(
                    ev.base_url, model, ev.prompts, ev.max_tokens, L,
                    ev.settle_s, SWEEP_WINDOW_S, stagger, cursor))
                m = summarize(reqs, t0, t1, slo)
                m.update(burstiness(reqs, t0, t1))
                m["head_penalty"] = head_penalty(reqs, t0, t1)
                m.update(concurrency=L, arm=arm, rep=rep + 1)
                rows.append(m)
                print(f"    L={L:<4} {arm:9s} rep{rep+1}  "
                      f"gp {m['goodput']:7.1f}  ttft99 {m['ttft_p99_ms']:7.0f}ms  "
                      f"p95 {m['ttft_p95_ms']:6.0f}ms  n={m['ttft_n']:<4} "
                      f"slo {m.get('slo_attainment', 0):5.1%}  "
                      f"burst {m.get('burst_ratio', 0):5.2f}x  "
                      f"headpen {m['head_penalty']:5.2f}x", flush=True)

(OUT / "trials.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def band(vals):
    vals = [v for v in vals if v is not None and v == v]
    if not vals:
        return "  --"
    lo, hi = min(vals), max(vals)
    return f"{lo:8.1f}..{hi:8.1f}  ({hi/lo:4.2f}x)" if lo else f"{lo:8.1f}..{hi:8.1f}"


print("\n  === locked vs dephased ===")
for L in LEVELS:
    print(f"\n  L={L}")
    for arm in ("locked", "dephased"):
        g = [r for r in rows if r["concurrency"] == L and r["arm"] == arm]
        print(f"    {arm:9s} goodput  {band([r['goodput'] for r in g])}")
        print(f"    {'':9s} ttft_p99 {band([r['ttft_p99_ms'] for r in g])}")
        print(f"    {'':9s} slo      {band([r.get('slo_attainment') for r in g])}")
        print(f"    {'':9s} burst    {band([r.get('burst_ratio') for r in g])}")
        print(f"    {'':9s} headpen  {band([r['head_penalty'] for r in g])}")

print("\n  READ: if 'locked' shows burst >> 1 and headpen >> 1 while 'dephased'")
print("        shows both near 1 AND a tighter ttft_p99 band, the bimodality is")
print("        the driver's convoy and the fix costs nothing. If both arms are")
print("        equally bursty, the convoy theory is dead and the tail is real.")
