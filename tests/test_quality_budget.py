"""The quality gate's generation budget follows the traffic when its answers
run longer than a benchmark's own (4 Oct 2026: a fifth of Qwen3's MATH-500
answers ran past the 1,024 tokens the benchmark allowed, and were scored cut
off before their final line)."""

from inferopt import quality


def test_the_traffic_floor_is_its_p99_with_a_quarter_of_headroom_in_32_token_steps():
    assert quality.traffic_floor(1640) == 2080
    assert quality.traffic_floor(0) == 0


def test_the_budget_rises_to_the_traffic_and_never_falls_below_the_benchmark():
    own = quality.BENCHMARKS["math_500"].max_tokens
    assert quality.generation_budget("math_500", 2080) == 2080
    assert quality.generation_budget("math_500", 64) == own, "a short-answer eval does not cut MATH-500"
    assert quality.generation_budget("math_500") == own


def test_the_scorer_asks_for_the_raised_budget(monkeypatch):
    asked = []
    monkeypatch.setattr(quality, "_load", lambda name, n: [{"problem": "1+1", "answer": "2"}])

    def gen(prompts, max_tokens):
        asked.append(max_tokens)
        return [type("Out", (), {"text": "\\boxed{2}"})() for _ in prompts]

    quality.run_benchmark("math_500", gen, max_tokens=2080)
    assert asked == [2080]


def test_a_callers_own_eval_is_scored_on_its_rows_by_its_metric(monkeypatch):
    """4 Oct 2026: every search scored MATH-500, whatever eval the customer
    brought; their rows and their metric are the frontier's quality axis."""
    from inferopt.api_types import Metric, Verdict

    def judge(rows, texts):
        return [Verdict(ok=text == row["expected"], value=1.0 if text == row["expected"] else 0.0)
                for row, text in zip(rows, texts)]

    rows = tuple({"prompt": f"q{i}", "expected": "yes"} for i in range(4))
    mean = Metric("f1", fn=lambda samples, verdicts: sum(v.value for v in verdicts) / len(verdicts))
    quality.register_benchmark("customer_eval", quality.Benchmark(
        judge=judge, prompt=lambda row: row["prompt"], metric=mean, n_full=4, max_tokens=512, chat=False,
        rows=rows))
    try:
        answers = iter(["yes", "no", "yes", "yes"])

        def gen(prompts, max_tokens):
            return [type("Out", (), {"text": next(answers)})() for _ in prompts]

        assert quality.run_benchmark("customer_eval", gen, full=True) == 0.75
        assert quality._load("customer_eval", 2) == list(rows[:2]), "the first n: a prefix the caller ordered"
        assert quality.BENCHMARKS["customer_eval"].metric_spec is mean
    finally:
        quality.BENCHMARKS.pop("customer_eval", None)


def test_a_registered_benchmark_must_carry_its_rows():
    import pytest

    with pytest.raises(ValueError, match="carries its own rows"):
        quality.register_benchmark("x", quality.Benchmark(
            judge=lambda rows, texts: [], prompt=str, metric="exact_match", n_full=1, max_tokens=64))
