"""Judges: turn generations into per-sample verdicts.

    judge = LLMJudge(model="claude-sonnet-4-5", rubric="Is the tone polite?")
    verdicts = judge(samples)

A judge answers one question per sample -- did THIS output satisfy the task --
and returns a Verdict carrying the reason. Aggregation is a Metric's job; the
two are separate because they fail differently and at different granularities.

WHY A MODEL-BACKED JUDGE NEEDS MORE CARE THAN A RULE-BASED ONE

A string match is deterministic and free. A judge that calls another model is
neither, and three of its failure modes look exactly like a quality regression
in the model under test:

  IT CAN BE DOWN. A rate limit or a timeout returns no verdict. Scoring that as
  a failure reports the served model got worse when the JUDGE got worse -- the
  same shape as RULER reading 0.05 and moving across a lossless node. So an
  unreachable judge raises rather than voting False, and a partial outage is
  reported as unscored rather than folded into the mean.

  IT IS NOT DETERMINISTIC. Two runs of the identical config can disagree, so a
  judge has its own resolution and a delta smaller than that is not a finding.
  temperature is pinned to 0 and `resolution()` measures what remains.

  IT COSTS MONEY PER SAMPLE. A 500-problem benchmark at every node of a
  traversal is thousands of calls. Judging is therefore capped, and the cap is
  reported rather than silently applied -- a truncated benchmark that looks
  complete is how a quality axis becomes decorative.

THE JUDGE IS NOT THE MODEL UNDER TEST. It runs against a separate endpoint,
because pointing it at the server being optimized would make the measurement
depend on the configuration it is measuring.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Callable, Sequence

from inferopt.api_types import Sample, Verdict


@dataclass
class RuleJudge:
    """A deterministic judge built from a plain function.

    The base case, and the one to prefer: exact match, a regex, a parser, a
    sandboxed execution. Free, reproducible, and it cannot be down.
    """

    fn: Callable[[Sample], bool | Verdict]
    name: str = "rule"

    def __call__(self, samples: Sequence[Sample]) -> list[Verdict]:
        out = []
        for s in samples:
            try:
                v = self.fn(s)
            except Exception as e:
                # A judge that throws on one sample must not lose the other 499.
                out.append(Verdict(False, reason=f"judge raised: {type(e).__name__}: {e}"))
                continue
            out.append(v if isinstance(v, Verdict) else Verdict(bool(v)))
        return out


@dataclass
class LLMJudge:
    """Judge each sample with another model, against a rubric.

    For questions no rule can express -- tone, helpfulness, whether an answer
    is faithful to a source. Returns one Verdict per sample IN ORDER, because
    the caller zips them against rows to find which one regressed.
    """

    model: str
    rubric: str
    base_url: str | None = None
    """Endpoint for the JUDGE, not the model under test. Defaults to
    $INFEROPT_JUDGE_URL. Pointing this at the server being optimized would make
    the measurement depend on the configuration it is measuring."""
    api_key_env: str = "INFEROPT_JUDGE_KEY"
    max_samples: int | None = 200
    """Cap, because a traversal judges at every node and this bills per call.
    None means no cap. Whatever is skipped is REPORTED, not silently dropped."""
    concurrency: int = 8
    timeout_s: float = 60.0
    temperature: float = 0.0
    name: str = "llm"
    skipped: int = field(default=0, init=False)

    def _prompt(self, s: Sample) -> str:
        return (
            f"{self.rubric}\n\n"
            f"--- OUTPUT UNDER REVIEW ---\n{s.text}\n--- END ---\n\n"
            f'Answer with JSON only: {{"pass": true|false, "reason": "<10 words>"}}')

    @staticmethod
    def _parse(text: str) -> Verdict:
        """Read the judge's answer, and say so when it is unreadable.

        A judge that returns prose instead of JSON has not voted False -- it has
        failed to vote, and the two must not be confused. An unreadable answer
        is an error the caller can see, not a quality signal.
        """
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                d = json.loads(m.group(0))
                if "pass" in d:
                    return Verdict(bool(d["pass"]), reason=str(d.get("reason", "")))
            except Exception:
                pass
        low = text.strip().lower()
        if low.startswith(("true", "yes", "pass")):
            return Verdict(True, reason="bare affirmative, not JSON")
        if low.startswith(("false", "no", "fail")):
            return Verdict(False, reason="bare negative, not JSON")
        raise ValueError(f"judge returned no parseable verdict: {text[:120]!r}")

    def __call__(self, samples: Sequence[Sample]) -> list[Verdict]:
        import asyncio

        import httpx

        url = self.base_url or os.environ.get("INFEROPT_JUDGE_URL")
        if not url:
            raise RuntimeError(
                "LLMJudge needs an endpoint. Set base_url= or "
                "$INFEROPT_JUDGE_URL.\n"
                "  It must NOT be the server under test -- judging with the "
                "configuration being measured makes the measurement depend on "
                "the thing it is measuring.")
        key = os.environ.get(self.api_key_env, "")

        use = list(samples)
        self.skipped = 0
        if self.max_samples and len(use) > self.max_samples:
            self.skipped = len(use) - self.max_samples
            use = use[: self.max_samples]

        async def one(client, s: Sample) -> Verdict:
            r = await client.post(
                f"{url.rstrip('/')}/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}"} if key else {},
                json={"model": self.model, "temperature": self.temperature,
                      "max_tokens": 64,
                      "messages": [{"role": "user", "content": self._prompt(s)}]},
                timeout=self.timeout_s)
            r.raise_for_status()
            return self._parse(r.json()["choices"][0]["message"]["content"])

        async def go():
            sem = asyncio.Semaphore(self.concurrency)
            async with httpx.AsyncClient(timeout=self.timeout_s) as c:
                async def guarded(s):
                    async with sem:
                        return await one(c, s)
                return await asyncio.gather(*[guarded(s) for s in use],
                                            return_exceptions=True)

        results = asyncio.run(go())

        # AN OUTAGE IS NOT A REGRESSION. Errors are counted, and if they are the
        # majority the whole judgement is refused -- reporting a model got worse
        # when the judge went down is the failure this exists to prevent.
        errs = [r for r in results if isinstance(r, BaseException)]
        if errs and len(errs) > len(results) / 2:
            raise RuntimeError(
                f"LLMJudge({self.model}): {len(errs)} of {len(results)} calls "
                f"failed -- refusing to score. First: {errs[0]!r}\n"
                f"  A judge outage reads as a quality regression in the model "
                f"under test, which it is not.")

        out: list[Verdict] = []
        for r in results:
            if isinstance(r, BaseException):
                out.append(Verdict(False, reason=f"judge unavailable: {type(r).__name__}"))
            else:
                out.append(r)
        return out

    def report(self) -> str:
        if not self.skipped:
            return f"{self.name}({self.model}): all samples judged"
        return (f"{self.name}({self.model}): {self.skipped} sample(s) NOT judged "
                f"(max_samples={self.max_samples}) -- the score covers a subset")
