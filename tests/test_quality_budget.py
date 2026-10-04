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
