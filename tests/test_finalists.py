"""Stage 2.1: the finalists swept densely, with the profile a DAG node gets.
See src/inferopt/finalists.py."""

import json
from types import SimpleNamespace

from inferopt import finalists
from inferopt.traverse import Trial


def _fp(decoding="autoregressive", pipeline=False):
    fp = SimpleNamespace(model=SimpleNamespace(decoding=decoding))
    if pipeline:
        fp.diffusion = SimpleNamespace(kind="video")
    return fp


def _trial(node_id, goodput, config=None, **over):
    fields = dict(node_id=node_id, config=config or {"x": node_id}, goodput=goodput,
                  ttft_p99_ms=100.0, itl_p99_ms=20.0, memory_gb=10.0,
                  concurrency=8, curve=[{"concurrency": 8, "goodput": goodput}],
                  provenance={"model": "m"})
    fields.update(over)
    return Trial(**fields)


class _Evaluator:
    """Measures whatever it is asked: one curve point per level, the profile
    a walk trial gets, and a failure for a config that says so."""

    def __init__(self):
        self.calls = []

    def measure(self, config, *, probes, benchmarks, node_id, levels, **_):
        self.calls.append((node_id, tuple(levels), tuple(probes), tuple(benchmarks)))
        if config.get("x") == "broken":
            raise RuntimeError("no such server")
        curve = [{"concurrency": L, "goodput": float(100 - abs(L - 12) * 5) if L <= 20 else 0.0}
                 for L in levels]
        return Trial(node_id=node_id, config=dict(config), goodput=max(c["goodput"] for c in curve),
                     ttft_p99_ms=90.0, itl_p99_ms=18.0, memory_gb=10.0, curve=curve,
                     concurrency=12, diagnostics={"profile": {"gpu_busy": 0.8, "families": {"attn": {"pct": 40}}}})


def test_the_ladder_is_sixteen_for_an_llm_eight_for_a_dllm_six_for_a_pipeline():
    assert len(finalists.ladder_for(_fp(), {})) == 16
    assert len(finalists.ladder_for(_fp("diffusion"), {})) == 8
    assert len(finalists.ladder_for(_fp(pipeline=True), {})) == 6


def test_the_ladder_stops_at_the_configurations_own_cap():
    """The sweep measures the configuration that ships; raising max_num_seqs
    to see more curve would measure a different one."""
    assert finalists.ladder_for(_fp("diffusion"), {"max_num_seqs": 4}) == (1, 2, 3, 4)
    assert finalists.ladder_for(_fp(), {"max_num_seqs": 5}) == (1, 2, 3, 4)
    assert finalists.ladder_for(_fp(), {"max_num_seqs": 1}) == (1,)


def test_the_incumbent_is_swept_even_when_off_the_frontier():
    a, b, c, inc = _trial("a", 90), _trial("b", 80), _trial("c", 70), _trial("kept_one", 60)
    picked = finalists.pick([a, b, c, inc], [a, b, c], "kept_one", n=2)
    assert [t.node_id for t in picked] == ["a", "b", "kept_one"]
    assert [t.node_id for t in finalists.pick([a, b], [a, b], "a", n=2)] == ["a", "b"]


def test_a_finalist_gets_its_dense_curve_its_peak_and_its_profile(tmp_path):
    ev = _Evaluator()
    a, b = _trial("a", 90), _trial("b", 80)
    journal = tmp_path / "trials.jsonl"
    journal.write_text("")
    notes = []
    out = finalists.sweep_finalists(ev, _fp(), [a, b], [a, b], "a", n=1,
                                    journal=journal, log=notes.append)

    assert [c[0] for c in ev.calls] == ["finalist:a"]
    assert ev.calls[0][1] == finalists.LADDER_LLM and ev.calls[0][3] == ()
    assert len(a.curve) == 16 and a.concurrency == 12, "the dense curve replaced the bracket"
    assert a.goodput == 90, "the walk's own number is left as measured"
    assert a.diagnostics["finalist"]["profile"]["gpu_busy"] == 0.8
    assert out["a"]["peak"]["concurrency"] == 12
    assert b.curve == [{"concurrency": 8, "goodput": 80}], "b was not a finalist"
    rows = [json.loads(l) for l in journal.read_text().splitlines()]
    assert [r["node_id"] for r in rows] == ["finalist:a"] and rows[0]["provenance"] == {"model": "m"}
    assert any("past the first miss" in n for n in notes)
    assert any("peak 100.0 at L=12" in n and "levels served" in n for n in notes)


def test_a_finalist_that_will_not_start_keeps_the_walks_measurement(tmp_path):
    ev = _Evaluator()
    broken = _trial("z", 50, config={"x": "broken"})
    notes = []
    out = finalists.sweep_finalists(ev, _fp(), [broken], [broken], "z", n=1,
                                    journal=tmp_path / "j.jsonl", log=notes.append)
    assert out == {} and broken.curve == [{"concurrency": 8, "goodput": 50}]
    assert any("sweep failed" in n for n in notes)


def test_a_replayed_sweep_is_not_journaled_twice(tmp_path):
    class Replaying(_Evaluator):
        def measure(self, config, **kw):
            t = super().measure(config, **kw)
            t.diagnostics["replayed"] = True
            return t
    journal = tmp_path / "trials.jsonl"
    journal.write_text("")
    a = _trial("a", 90)
    finalists.sweep_finalists(Replaying(), _fp(), [a], [a], "a", n=1, journal=journal,
                              log=lambda *_: None)
    assert journal.read_text() == "" and len(a.curve) == 16


def test_nothing_within_the_target_means_no_sweep():
    ev = _Evaluator()
    notes = []
    assert finalists.sweep_finalists(ev, _fp(), [_trial("a", 0.0)], [], "a", n=3,
                                     log=notes.append) == {}
    assert ev.calls == [] and any("nothing met the target" in n for n in notes)
    assert finalists.sweep_finalists(ev, _fp(), [_trial("a", 5.0)], [_trial("a", 5.0)], "a",
                                     n=0, log=lambda *_: None) == {}
