"""Measure what an LLM serving endpoint actually delivers under load.

    from goodput import Latency, closed_loop, summarize

    reqs, t0, t1 = asyncio.run(closed_loop(url, model, prompts, lengths, conc=128,
                                           settle_s=20, window_s=45))
    print(summarize(reqs, t0, t1, Latency(ttft_p99_ms=500, itl_p99_ms=250)))

Independent of inferopt: no DAG, no fingerprint, no quantizer, no vLLM import.
It speaks the OpenAI completions API and counts tokens as they arrive.
"""
from goodput import export
from goodput.driver import Req, _closed_loop, _load, _mt, _one
from goodput.metrics import summarize
from goodput.slo import Latency, LatencyTarget

# Public spellings for the internal names, which keep their underscores so the
# moved code and its callers in inferopt stay byte-identical.
closed_loop = _closed_loop
open_loop = _load
one_request = _one

__all__ = ["Req", "Latency", "LatencyTarget", "summarize", "export",
           "closed_loop", "open_loop", "one_request",
           "_closed_loop", "_load", "_one", "_mt"]
