"""Turning a list of requests into the numbers a config is judged on.

The definition everything rests on: goodput counts tokens only from requests
that met the SLO, so `throughput = goodput + tokens that arrived late`. The rest
of this module exists to stop a reader mistaking one statistic for another.
"""
from __future__ import annotations

from goodput.driver import Req
from goodput.slo import LatencyTarget as SLO


def _reasons(reqs: list[Req]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in reqs:
        if not r.ok and r.error:
            out[r.error] = out.get(r.error, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1])[:5])


def summarize(reqs: list[Req], t0: float, t1: float, slo: SLO) -> dict:
    win = max(1e-9, t1 - t0)
    in_win = lambda tt: t0 <= tt < t1
    all_tok = sum(1 for r in reqs for tt in r.token_times if in_win(tt))
    good_tok = sum(1 for r in reqs if r.meets(slo) for tt in r.token_times if in_win(tt))
    started = [r for r in reqs if t0 <= r.start < t1]
    done = [r for r in started if r.ok]
    ttfts = sorted(r.ttft for r in done if r.ttft is not None)
    itls = sorted(r.itl_s()
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
        # DENOMINATOR IS `started`, NOT `done`. Req.meets() returns False for a
        # request that failed, so excluding failures from the denominator
        # contradicts the very predicate this is a fraction of: a config where
        # a fifth of requests error out but the rest are quick would report
        # 100% attainment. Measured on the existing corpus this moved one trial
        # by more than two points (14b-seqdag spec_decode_ngram, 0.774 -> 0.747),
        # so it is small in practice and wrong in principle.
        "slo_attainment": (sum(r.meets(slo) for r in started) / len(started)) if started else 0.0,
        "ttft_p99_ms": pct(ttfts, 0.99) * 1e3,
        "itl_p99_ms": pct(itls, 0.99) * 1e3,
        # p95 ALONGSIDE p99, and the sample count that both are computed over.
        #
        # A 45s window at L=128 completes ~384 requests, so p99 is the slowest
        # 3.8 of them -- a max over a handful, not a percentile. Measured across
        # three identical launches of one configuration, TTFT p99 varied 6.63x
        # at L=64 (149, 160, 988 ms) and 3.72x at L=128, while goodput over the
        # same launches varied 1.06x and 1.20x. The instability is in the
        # statistic, not the server.
        #
        # p95 over the same window is the slowest ~19 requests, which is a
        # percentile rather than a maximum. It is reported BESIDE p99 rather
        # than replacing it, because p99 is what the SLO is written against and
        # silently redefining a target is worse than exposing a noisy one.
        #
        # ttft_n is the count both are drawn from. A p99 over 4 requests and a
        # p99 over 1500 are different measurements wearing the same name, and
        # carrying the count is what lets a reader tell them apart. Lengthening
        # the window would fix it properly and multiplies every run's cost by
        # four, which this project does not have.
        #
        # None of this touches the search: goodput counts SLO-conforming
        # requests aggregated over all ~384 completions, which is why it is
        # stable, and goodput is what keep/revert runs on.
        "ttft_p95_ms": pct(ttfts, 0.95) * 1e3,
        "itl_p95_ms": pct(itls, 0.95) * 1e3,
        "ttft_n": len(ttfts),
        "completed": len(done), "failed": len(started) - len(done), "window_s": win,
        # The distinct reasons, most common first. A screen losing a quarter of
        # its rows to one repeated HTTP 400 is a different problem from losing
        # them to assorted timeouts, and "failed: 3" cannot tell them apart.
        "failure_reasons": _reasons(started),
    }

