"""Integration tests: real traverse() runs against a scripted evaluator.

    python test_dag_integration.py

No GPU and no server. The evaluator is a stub whose numbers are chosen per test,
so a traversal's DECISIONS can be checked exactly: which nodes were kept, what
the incumbent accumulated, where the operating point moved, what reached the
frontier, what the journal recorded.

This is the layer selftest.py does not reach. selftest runs one traversal over
the real DAG and asserts it completes; these run many traversals over small
purpose-built DAGs and assert what the traversal DECIDED. Every case is either
an invariant the search depends on, or a shape that has produced a wrong result
in this project before.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

FAIL: list[str] = []
N = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global N
    N += 1
    if not cond:
        FAIL.append(f"{name}: {detail}")
        print(f"  FAIL  {name}" + (f"  -- {detail}" if detail else ""))


def section(t: str) -> None:
    print(f"\n=== {t} ===")


# --------------------------------------------------------------------- rig

class ScriptedEvaluator:
    """Returns programmed measurements. Records every config it was asked to run.

    `script` maps node_id -> a value or list of values (one per variant):
        float                       goodput, everything else defaulted
        dict                        any Trial field, e.g. {"goodput":…, "quality":…}
        Exception instance          raised, to simulate a launch failure
    """

    def __init__(self, script: dict, default=10.0):
        self.script = script
        self.default = default
        self.calls: list[tuple[str, dict]] = []
        self.concurrencies: list[int | None] = []

    def measure(self, config, *, probes, benchmarks, node_id,
                concurrency=None, levels=None, fixed_concurrency=None):
        from traverse import Trial
        self.calls.append((node_id, dict(config)))
        self.concurrencies.append(concurrency)
        spec = self.script.get(node_id, self.default)
        if isinstance(spec, list):
            i = sum(1 for c, _ in self.calls if c == node_id) - 1
            spec = spec[min(i, len(spec) - 1)]
        if isinstance(spec, BaseException):
            raise spec
        if isinstance(spec, (int, float)):
            spec = {"goodput": float(spec)}
        d = {"goodput": 10.0, "ttft_p99_ms": 100.0, "itl_p99_ms": 10.0,
             "memory_gb": 10.0, "quality": {}, "slo_ok": True,
             "concurrency": concurrency or 8, **spec}
        return Trial(node_id=node_id, config=dict(config), **d)


def mkdag(nodes, budget_launches=100, budget_minutes=600):
    return {
        "traversal": {"start": "root",
                      "budget_guard": {"max_launches": budget_launches,
                                       "max_minutes": budget_minutes}},
        "nodes": [{"id": "root", "class": "root", "status": "active",
                   "on_keep": [nodes[0]["id"]]}] + nodes +
                 [{"id": "end", "class": "terminal", "status": "active"}],
    }


def node(i, nxt="end", cls="lossless", **kw):
    n = {"id": i, "class": cls, "status": "active",
         "action": {"set": {i: True}}, "probes": ["goodput"],
         "on_keep": [nxt], "on_revert": [nxt]}
    n.update(kw)
    return n


def ctx(**over):
    from fingerprint import (Context, Fingerprint, HardwareFingerprint, SLO,
                             LoraFingerprint, ModelFingerprint, WorkloadFingerprint)
    fp = Fingerprint(
        model=ModelFingerprint(id="t/m", architecture="A", n_params_b=14.0,
                               n_layers=40, hidden_size=5120, n_heads=40,
                               n_kv_heads=8, attention_type="gqa",
                               max_model_len=32768, bytes_per_param=2.0),
        hw=HardwareFingerprint(gpu_name="G", compute_capability="9.0",
                               memory_gb=80.0, memory_bandwidth_gb_s=3350.0,
                               system_ram_gb=200.0, cpu_cores=32),
        workload=WorkloadFingerprint(n_requests=800, mean_input_tokens=620.0,
                                     p99_input_tokens=2660, p999_input_tokens=4380,
                                     mean_output_tokens=260.0, p99_output_tokens=804,
                                     request_rate_qps=16.0, max_concurrency=32,
                                     prefix_overlap=0.31,
                                     prefix_overlap_per_adapter=0.31),
        lora=LoraFingerprint())
    c = Context(fingerprint=fp,
                slo=SLO(ttft_p99_ms=500, itl_p99_ms=250, quality_budget=0.1,
                        lossless_quality_budget=0.03))
    c.accept_band = 0.05
    for k, v in over.items():
        setattr(c, k, v)
    return c


def run(dag, script, *, default=10.0, c=None, **kw):
    from traverse import traverse
    ev = ScriptedEvaluator(script, default)
    res = traverse(dag, c or ctx(), ev, log=lambda *a: None, **kw)
    return res, ev


def kept(res):
    return [t.node_id for t in res.trials if t.kept]


# ==================================================================== tests

def test_keep_revert_band():
    section("keep/revert honours the accept band")
    d = mkdag([node("a", "b"), node("b")])

    # a beats the 10.0 incumbent by 50% -> keep. b beats a by 2% -> revert.
    res, ev = run(d, {"incumbent": 10.0, "a": 15.0, "b": 15.3})
    check("a clear win is kept", "a" in kept(res), f"kept={kept(res)}")
    check("a win inside the band is reverted", "b" not in kept(res),
          f"kept={kept(res)} -- 15.3 vs 15.0 is +2%, under a 5% band")

    # Exactly at the band must NOT be kept: the rule is strictly greater.
    res, _ = run(d, {"incumbent": 10.0, "a": 10.5, "b": 1.0})
    check("exactly at the band is not a keep", "a" not in kept(res),
          f"kept={kept(res)} -- 10.5 is exactly +5%, the band is a floor to EXCEED")

    # A regression is never kept.
    res, _ = run(d, {"incumbent": 100.0, "a": 50.0, "b": 1.0})
    check("a regression is reverted", not kept(res), f"kept={kept(res)}")


def test_incumbent_accumulates():
    section("the incumbent accumulates on keep and not on revert")
    d = mkdag([node("a", "b"), node("b", "c"), node("c")])
    res, ev = run(d, {"incumbent": 10.0, "a": 20.0, "b": 5.0, "c": 40.0})
    cfgs = dict((n, c) for n, c in ev.calls)
    check("b sees a's setting", cfgs["b"].get("a") is True, f"b saw {cfgs['b']}")
    check("c sees a's setting", cfgs["c"].get("a") is True, f"c saw {cfgs['c']}")
    check("c does NOT see reverted b", "b" not in cfgs["c"],
          f"c saw {cfgs['c']} -- a reverted node must not leak into the incumbent")
    check("final incumbent has the kept nodes only",
          res.incumbent.get("a") is True and res.incumbent.get("c") is True
          and "b" not in res.incumbent, f"{res.incumbent}")


def test_operating_point_follows():
    section("the operating point follows the incumbent")
    d = mkdag([node("a", "b"), node("b")])
    res, ev = run(d, {"incumbent": {"goodput": 10.0, "concurrency": 4},
                      "a": {"goodput": 50.0, "concurrency": 32},
                      "b": {"goodput": 60.0, "concurrency": 64}},
                  concurrency=4)
    # b must be measured at the concurrency a's keep established, not the seed's.
    at_b = [c for n, c in zip([n for n, _ in ev.calls], ev.concurrencies) if n == "b"]
    check("a kept node moves the operating point for the next node",
          at_b and at_b[0] == 32,
          f"b measured at {at_b} -- run five pinned every node to the seed's peak "
          f"and produced three identical numbers")
    check("the Result records the final operating point", res.concurrency == 64,
          f"{res.concurrency}")


def test_quality_gate():
    section("quality gate on lossy nodes")
    LOSSY = dict(cls="lossy", quality_benchmarks=["math_500"])
    d = mkdag([node("q", **LOSSY)])
    c = ctx()
    c.quality_baseline = {"math_500": 0.80}

    # Big goodput win, quality loss WITHIN budget (0.1) -> kept.
    res, _ = run(d, {"incumbent": 10.0,
                     "q": {"goodput": 100.0, "quality": {"math_500": 0.75}}}, c=ctx())
    c2 = ctx(); c2.quality_baseline = {"math_500": 0.80}
    res, _ = run(d, {"incumbent": 10.0,
                     "q": {"goodput": 100.0, "quality": {"math_500": 0.75}}}, c=c2)
    check("a loss inside allow_loss is kept", "q" in kept(res),
          f"kept={kept(res)} -- 0.80->0.75 is 0.05, budget is 0.10")

    # Quality loss OVER budget -> rejected however good the goodput.
    c3 = ctx(); c3.quality_baseline = {"math_500": 0.80}
    res, _ = run(d, {"incumbent": 10.0,
                     "q": {"goodput": 1000.0, "quality": {"math_500": 0.50}}}, c=c3)
    check("a loss over allow_loss is rejected", "q" not in kept(res),
          f"kept={kept(res)} -- 0.30 loss against a 0.10 budget, 100x goodput "
          f"must not buy it")

    # A LOSSLESS node with the same quality drop is NOT gated -- by design the
    # gate only runs for class == "lossy". Worth pinning: it means a mislabelled
    # node skips the gate entirely.
    d2 = mkdag([node("l", quality_benchmarks=["math_500"])])
    c4 = ctx(); c4.quality_baseline = {"math_500": 0.80}
    res, _ = run(d2, {"incumbent": 10.0,
                      "l": {"goodput": 1000.0, "quality": {"math_500": 0.10}}}, c=c4)
    check("the quality gate is keyed on class == 'lossy'", "l" in kept(res),
          f"kept={kept(res)} -- documents that a mislabelled lossy node is ungated")


def test_budget_guard():
    section("budget guard stops before spending")
    ns = [node(f"n{i}", f"n{i+1}") for i in range(6)]
    ns[-1]["on_keep"] = ns[-1]["on_revert"] = ["end"]
    for n in ns:
        n["cost_launches"] = 1
    d = mkdag(ns, budget_launches=3)
    res, ev = run(d, {}, default=10.0)
    check("the guard stops the run", res.stopped_early is not None,
          "ran past the launch budget")
    check("it stops at or under the budget", len(ev.calls) <= 3,
          f"{len(ev.calls)} launches against a budget of 3")
    check("a partial result is still returned", isinstance(res.trials, list),
          "the guard must leave a usable Result, not raise")


def test_lossless_only():
    section("lossless_only parks the lossy branch")
    d = mkdag([node("a", "q"), node("q", cls="lossy",
                                    quality_benchmarks=["math_500"])])
    res, ev = run(d, {"incumbent": 10.0, "a": 20.0, "q": 999.0}, lossless_only=True)
    check("the lossy node is never launched",
          "q" not in [n for n, _ in ev.calls], f"calls={[n for n, _ in ev.calls]}")
    check("it is recorded as skipped, with a reason",
          any(s[0] == "q" for s in res.skipped), f"skipped={res.skipped}")
    check("the lossless node still ran", "a" in [n for n, _ in ev.calls],
          f"calls={[n for n, _ in ev.calls]}")


def test_skips_are_free():
    section("skipped nodes cost nothing and are recorded")
    d = mkdag([node("off", "on", status="todo"),
               node("on", applicable_when="workload.prefix_overlap > 0.9")])
    res, ev = run(d, {"incumbent": 10.0})
    names = [n for n, _ in ev.calls]
    check("an inactive node is not launched", "off" not in names, f"{names}")
    check("an inapplicable node is not launched", "on" not in names, f"{names}")
    check("both are recorded as skipped", len(res.skipped) == 2, f"{res.skipped}")
    check("the skip reason is kept", all(s[1] for s in res.skipped), f"{res.skipped}")


def test_predicate_error_skips():
    section("a broken predicate skips rather than crashes")
    d = mkdag([node("bad", applicable_when="nonexistent.field > 1")])
    try:
        res, ev = run(d, {"incumbent": 10.0})
    except Exception as e:
        check("a broken predicate does not kill the traversal", False,
              f"{type(e).__name__}: {e}")
        return
    check("a broken predicate does not kill the traversal", True)
    check("the node is skipped", any(s[0] == "bad" for s in res.skipped),
          f"{res.skipped}")
    check("the skip reason names the predicate failure",
          any("predicate error" in s[1] for s in res.skipped),
          f"{res.skipped} -- a silently disabled node is the bug validate_dag exists "
          f"to prevent, so the reason must say so")


def test_launch_failure():
    section("a launch failure does not become the incumbent")
    d = mkdag([node("a", "b"), node("b")])
    res, ev = run(d, {"incumbent": 10.0,
                      "a": {"goodput": 0.0, "ttft_p99_ms": float("inf"),
                            "itl_p99_ms": float("inf"), "slo_ok": False},
                      "b": 20.0})
    check("a failed launch is not kept", "a" not in kept(res), f"kept={kept(res)}")
    cfgs = dict((n, c) for n, c in ev.calls)
    check("the next node does not inherit the failed config", "a" not in cfgs["b"],
          f"b saw {cfgs['b']}")
    check("the failure is not on the frontier",
          "a" not in [t.node_id for t in res.frontier()],
          f"frontier={[t.node_id for t in res.frontier()]}")
    check("the traversal continues past it", "b" in [n for n, _ in ev.calls],
          f"calls={[n for n, _ in ev.calls]}")


def test_evaluator_exception():
    section("an evaluator exception does not lose completed trials")
    d = mkdag([node("a", "boom"), node("boom", "c"), node("c")])
    from traverse import traverse
    ev = ScriptedEvaluator({"incumbent": 10.0, "a": 50.0,
                            "boom": RuntimeError("engine died"), "c": 99.0})
    with tempfile.TemporaryDirectory() as td:
        j = Path(td) / "trials.jsonl"
        try:
            traverse(mkdag([node("a", "boom"), node("boom", "c"), node("c")]),
                     ctx(), ev, log=lambda *a: None, journal=j)
            raised = False
        except Exception:
            raised = True
        lines = [json.loads(l) for l in j.read_text().splitlines()] if j.exists() else []
        got = [l.get("node_id") for l in lines if "node_id" in l]
        check("trials completed before the exception are journaled",
              "a" in got,
              f"journal has {got} -- a KeyError in the quality probe once discarded "
              f"nine launches because result.json is only written on return")


def test_journal():
    section("the journal is complete and readable mid-run")
    d = mkdag([node("a", "b"), node("b")])
    from traverse import traverse
    ev = ScriptedEvaluator({"incumbent": 10.0, "a": 50.0, "b": 12.0})
    with tempfile.TemporaryDirectory() as td:
        j = Path(td) / "t.jsonl"
        res = traverse(d, ctx(), ev, log=lambda *a: None, journal=j)
        lines = [json.loads(l) for l in j.read_text().splitlines()]
        trials = [l for l in lines if "node_id" in l]
        decisions = [l for l in lines if "decision" in l]
        check("every measured trial is journaled",
              {t["node_id"] for t in trials} >= {"a", "b"},
              f"{[t['node_id'] for t in trials]}")
        check("keep/revert decisions are journaled too", len(decisions) >= 2,
              f"{len(decisions)} decisions -- record() runs BEFORE the decision, so "
              f"trials.jsonl alone says kept=False for everything")
        byid = {d_["decision"]: d_ for d_ in decisions}
        check("the journal's decision matches the Result",
              byid.get("a", {}).get("kept") is True
              and byid.get("b", {}).get("kept") is False,
              f"journal={ {k: v.get('kept') for k, v in byid.items()} } "
              f"result kept={kept(res)}")
        check("each decision records what it was measured against",
              all("prev_goodput" in d_ and "accept_band" in d_ for d_ in decisions),
              "a verdict with no baseline or band cannot be re-checked later")


def test_frontier_includes_reverted():
    section("reverted configs still reach the frontier")
    d = mkdag([node("a", "b"), node("b")])
    # b loses on goodput but wins on latency -> reverted, yet non-dominated.
    res, _ = run(d, {"incumbent": 10.0,
                     "a": {"goodput": 100.0, "ttft_p99_ms": 400.0},
                     "b": {"goodput": 90.0, "ttft_p99_ms": 50.0}})
    fr = [t.node_id for t in res.frontier()]
    check("a reverted but non-dominated config is on the frontier", "b" in fr,
          f"frontier={fr} -- 'worse goodput, better latency' is a trade, not a failure")
    check("the kept config is on it too", "a" in fr, f"frontier={fr}")


def test_quality_inheritance():
    section("lossless nodes inherit the quality baseline")
    d = mkdag([node("a")])
    c = ctx(); c.quality_baseline = {"math_500": 0.73}
    res, _ = run(d, {"incumbent": 10.0, "a": 50.0}, c=c)
    a = [t for t in res.trials if t.node_id == "a"][0]
    check("a lossless trial carries a quality coordinate",
          a.quality.get("math_500") == 0.73, f"{a.quality}")
    check("and it is flagged as inherited, not measured", a.quality_inherited,
          "an inherited score reported as measured would look like evidence")


def test_determinism():
    section("the same inputs produce the same decisions")
    d = mkdag([node("a", "b"), node("b", "c"), node("c")])
    script = {"incumbent": 10.0, "a": 20.0, "b": 20.5, "c": 60.0}
    r1, e1 = run(d, script)
    r2, e2 = run(d, script)
    check("kept sets match", kept(r1) == kept(r2), f"{kept(r1)} vs {kept(r2)}")
    check("incumbents match", r1.incumbent == r2.incumbent,
          f"{r1.incumbent} vs {r2.incumbent}")
    check("the same configs were measured in the same order",
          [n for n, _ in e1.calls] == [n for n, _ in e2.calls],
          f"{[n for n, _ in e1.calls]} vs {[n for n, _ in e2.calls]}")


def test_sweep_variants_all_measured():
    section("every sweep variant is measured and the best one wins")
    d = mkdag([node("s", sweep=[{"k": 1}, {"k": 2}, {"k": 3}])])
    res, ev = run(d, {"incumbent": 10.0, "s": [12.0, 90.0, 30.0]})
    ks = [c.get("k") for n, c in ev.calls if n == "s"]
    check("all three variants ran", ks == [1, 2, 3], f"{ks}")
    check("the best variant is the one kept",
          res.incumbent.get("k") == 2, f"incumbent={res.incumbent}")
    check("every variant is in the trial database for the frontier",
          len([t for t in res.trials if t.node_id == "s"]) == 3,
          "losing variants must still be able to reach the frontier")


def test_real_dag_walks():
    section("the real DAG walks end to end")
    dag = json.loads(Path("dag/llm.json").read_text())
    res, ev = run(dag, {}, default=10.0, c=ctx(), lossless_only=True)
    check("it terminates", isinstance(res.visited, list) and res.visited,
          "no nodes visited")
    check("it does not exceed its own launch guard",
          res.launches <= dag["traversal"]["budget_guard"]["max_launches"]["default"],
          f"{res.launches} launches")
    check("every visited node exists in the DAG",
          set(res.visited) <= {n["id"] for n in dag["nodes"]},
          f"{set(res.visited) - {n['id'] for n in dag['nodes']}}")
    check("no lossy node was launched under lossless_only",
          not [n for n, _ in ev.calls
               if n in {x["id"] for x in dag["nodes"] if x.get("class") == "lossy"}],
          f"{[n for n, _ in ev.calls]}")


# ==========================================================================
def main() -> int:
    tests = [test_keep_revert_band, test_incumbent_accumulates,
             test_operating_point_follows, test_quality_gate, test_budget_guard,
             test_lossless_only, test_skips_are_free, test_predicate_error_skips,
             test_launch_failure, test_evaluator_exception, test_journal,
             test_frontier_includes_reverted, test_quality_inheritance,
             test_determinism, test_sweep_variants_all_measured,
             test_real_dag_walks,
             test_tolerance_is_reporting_not_gating,
             test_checkpoint_adopts_tolerance, test_zero_width_gate,
             test_measurements_visible_downstream, test_no_infinite_loop,
             test_all_variants_fail, test_zero_incumbent, test_baseline_carried,
             test_report_renders]
    for fn in tests:
        try:
            fn()
        except Exception as e:
            import traceback
            print(f"  ERROR in {fn.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
            FAIL.append(f"{fn.__name__} raised: {e}")
    print(f"\n  {N - len(FAIL)}/{N} checks passed")
    if FAIL:
        print(f"\n  {len(FAIL)} FAILURE(S):")
        for f in FAIL:
            print(f"    - {f}")
        return 1
    print("  all integration checks passed")
    return 0




# ================================================= second round: deeper probes

def test_tolerance_is_reporting_not_gating():
    """PINS a subtlety that reads like a bug and is not.

    The lossy gate consults two numbers, but only ONE of them can reject:

        if delta <= tolerance:  continue      # dismissed as noise
        if delta >  budget:     keep = False  # rejected as unaffordable
        log("real loss, within budget")       # kept, and SAID OUT LOUD

    Nothing between tolerance and budget is rejected -- it is kept and
    reported. So swapping the measured tolerance for the default changes what
    the run SAYS about a variant, never what it DOES with it. A mutation test
    flagged that as uncovered; it is an equivalent mutant for the decision.

    What makes that safe is the checkpoint's zero-width check, which stops the
    run when tolerance >= budget. Without it, a tolerance above the budget
    would swallow losses the user asked to reject -- so these two pieces are
    load-bearing together, and this test fails if either is removed.
    """
    section("quality tolerance reports; budget decides")
    d = mkdag([node("q", cls="lossy", quality_benchmarks=["math_500"])])

    for label, tol in (("measured 0.10", 0.10), ("default", None)):
        c = ctx()
        c.quality_baseline = {"math_500": 0.80}
        c.slo.quality_budget = 0.20
        if tol is not None:
            c.quality_tolerance = {"math_500": tol}
        res, _ = run(d, {"incumbent": 10.0,
                         "q": {"goodput": 100.0, "quality": {"math_500": 0.74}}}, c=c)
        check(f"a 0.06 loss under a 0.20 budget is kept ({label})",
              "q" in kept(res), f"kept={kept(res)}")

    # Budget is what actually rejects, at any tolerance.
    for label, tol in (("measured 0.10", 0.10), ("default", None)):
        c = ctx()
        c.quality_baseline = {"math_500": 0.80}
        c.slo.quality_budget = 0.02
        if tol is not None:
            c.quality_tolerance = {"math_500": tol}
        res, _ = run(d, {"incumbent": 10.0,
                         "q": {"goodput": 100.0, "quality": {"math_500": 0.74}}}, c=c)
        # tolerance 0.10 >= budget 0.02 is the zero-width condition; with no
        # checkpoint in this DAG it is never detected, and the loss is silently
        # swallowed. That is exactly why the checkpoint check has to exist.
        if tol == 0.10:
            check("WITHOUT a checkpoint, a tolerance above the budget swallows "
                  "a loss the user asked to reject",
                  "q" in kept(res),
                  f"kept={kept(res)} -- documents why the zero-width check is "
                  f"load-bearing, not decorative")
        else:
            check(f"a 0.06 loss over a 0.02 budget is rejected ({label})",
                  "q" not in kept(res), f"kept={kept(res)}")


def test_checkpoint_adopts_tolerance():
    section("checkpoint adopts measured noise as the lossy tolerance")
    d = mkdag([node("a", "cp"),
               node("cp", "q", cls="checkpoint", quality_benchmarks=["math_500"]),
               node("q", cls="lossy", quality_benchmarks=["math_500"])])
    c = ctx()
    c.quality_baseline = {"math_500": 0.80}
    c.slo.quality_budget = 0.20
    # The lossless branch moves quality by 0.05. It cannot really have -- that
    # movement IS the noise floor, and every later lossy node must be judged
    # against it rather than against a guessed default.
    res, _ = run(d, {"incumbent": 10.0, "a": 20.0,
                     "cp": {"goodput": 30.0, "quality": {"math_500": 0.75}},
                     "q": {"goodput": 100.0, "quality": {"math_500": 0.72}}}, c=c)
    check("the checkpoint adopts the observed drift as tolerance",
          c.quality_tolerance.get("math_500", 0) >= 0.05,
          f"tolerance={c.quality_tolerance} -- 0.80->0.75 across a branch that "
          f"cannot move quality is a 0.05 noise floor")
    # q drops 0.75 -> 0.72 = 0.03, UNDER the adopted 0.05 tolerance, so it is
    # not a real loss and must not be gated out.
    check("a lossy delta under the adopted tolerance is not treated as a loss",
          "q" in kept(res), f"kept={kept(res)}")


def test_zero_width_gate():
    section("a zero-width quality gate stops the run instead of wasting launches")
    d = mkdag([node("cp", "q", cls="checkpoint", quality_benchmarks=["math_500"]),
               node("q", cls="lossy", quality_benchmarks=["math_500"])])
    c = ctx()
    c.quality_baseline = {"math_500": 0.80}
    c.slo.quality_budget = 0.03          # equal to the drift below
    res, ev = run(d, {"incumbent": 10.0,
                      "cp": {"goodput": 30.0, "quality": {"math_500": 0.77}},
                      "q": {"goodput": 999.0, "quality": {"math_500": 0.79}}}, c=c)
    check("the run stops rather than entering an un-passable branch",
          res.stopped_early is not None,
          "tolerance 0.03 >= budget 0.03 leaves no width: every lossy node is "
          "either dismissed as noise or rejected as unaffordable")
    check("no lossy node is launched after the stop",
          "q" not in [n for n, _ in ev.calls], f"calls={[n for n, _ in ev.calls]}")
    check("the stop reason names the two numbers",
          res.stopped_early and "toleran" in res.stopped_early.lower(),
          f"{res.stopped_early}")


def test_measurements_visible_downstream():
    section("a node can gate on an earlier node's outcome")
    d = mkdag([node("a", "b"),
               node("b", applicable_when="measurements.a.kept")])
    res, ev = run(d, {"incumbent": 10.0, "a": 50.0, "b": 60.0})
    check("b runs when a was kept", "b" in [n for n, _ in ev.calls],
          f"calls={[n for n, _ in ev.calls]}")

    res, ev = run(d, {"incumbent": 10.0, "a": 10.1, "b": 60.0})
    check("b is skipped when a was reverted", "b" not in [n for n, _ in ev.calls],
          f"calls={[n for n, _ in ev.calls]} -- a's tiny gain is inside the band")
    check("the skip is recorded", any(s[0] == "b" for s in res.skipped),
          f"{res.skipped}")

    # A SKIPPED node must also leave a measurements entry, or a downstream
    # predicate reading it raises instead of evaluating to False.
    d2 = mkdag([node("a", "b", status="todo"),
                node("b", applicable_when="measurements.a.kept")])
    res, ev = run(d2, {"incumbent": 10.0, "b": 60.0})
    check("gating on a SKIPPED node does not raise",
          not any("predicate error" in s[1] for s in res.skipped),
          f"{res.skipped} -- a skipped node must still record kept=False")


def test_no_infinite_loop():
    section("a cyclic DAG terminates instead of hanging")
    d = mkdag([node("a", "b"), node("b", "a")])
    d["traversal"]["budget_guard"]["max_launches"] = 6
    for n in d["nodes"]:
        if n["id"] in ("a", "b"):
            n["cost_launches"] = 1
    res, ev = run(d, {"incumbent": 10.0, "a": 10.0, "b": 10.0})
    check("a cycle is bounded by the launch guard", len(ev.calls) <= 7,
          f"{len(ev.calls)} launches -- a cycle with no guard runs forever")
    check("it reports why it stopped", res.stopped_early is not None,
          "a run that hit a cycle must say so")


def test_all_variants_fail():
    section("a node whose every variant fails the SLO")
    d = mkdag([node("s", "b", sweep=[{"k": 1}, {"k": 2}]), node("b")])
    dead = {"goodput": 0.0, "ttft_p99_ms": float("inf"),
            "itl_p99_ms": float("inf"), "slo_ok": False}
    res, ev = run(d, {"incumbent": 10.0, "s": [dead, dead], "b": 40.0})
    check("nothing is kept from a node with no viable variant",
          "s" not in kept(res), f"kept={kept(res)}")
    check("the incumbent is unchanged", "k" not in res.incumbent,
          f"{res.incumbent}")
    check("the traversal continues", "b" in [n for n, _ in ev.calls],
          f"calls={[n for n, _ in ev.calls]}")


def test_zero_incumbent():
    section("a seed measuring zero goodput")
    d = mkdag([node("a")])
    # incumbent_goodput == 0 short-circuits the band, so the first improvement
    # is kept unconditionally. Pinned because it is the only path where the
    # band does not apply.
    res, _ = run(d, {"incumbent": 0.0, "a": 0.001})
    check("any improvement over a zero baseline is kept", "a" in kept(res),
          f"kept={kept(res)} -- with a zero incumbent every ratio is undefined, "
          f"so the band is bypassed by design")


def test_baseline_carried():
    section("the stage 1.3 baseline reaches the Result")
    from traverse import traverse, Trial
    base = Trial(node_id="stage_1_3", config={"seed": True}, goodput=5.7,
                 ttft_p99_ms=5419.0, itl_p99_ms=175.0, memory_gb=95.0,
                 quality={"math_500": 0.71})
    ev = ScriptedEvaluator({"a": 50.0})
    c = ctx()
    from fingerprint import NodeMeasurement
    c.incumbent_metrics = NodeMeasurement(kept=True, goodput=5.7)
    res = traverse(mkdag([node("a")]), c, ev, log=lambda *a: None, baseline=base)
    check("Result.baseline is the seed measurement",
          res.baseline is not None and res.baseline.goodput == 5.7,
          f"{res.baseline}")
    check("a supplied baseline is not re-measured",
          "incumbent" not in [n for n, _ in ev.calls],
          f"calls={[n for n, _ in ev.calls]} -- stage 1.3 already paid for it")


def test_report_renders():
    section("report() renders every shape without raising")
    from traverse import report, Result, Trial
    def T(**kw):
        d = {"node_id": "n", "config": {}, "goodput": 10.0, "ttft_p99_ms": 100.0,
             "itl_p99_ms": 10.0, "memory_gb": 10.0}
        d.update(kw)
        return Trial(**d)
    cases = {
        "empty result": Result(trials=[], incumbent={}, visited=[], skipped=[],
                               launches=0, minutes=0.0),
        "no baseline": Result(trials=[T()], incumbent={}, visited=["n"],
                              skipped=[], launches=1, minutes=1.0),
        "with baseline": Result(trials=[T()], incumbent={}, visited=["n"],
                                skipped=[], launches=1, minutes=1.0, baseline=T(goodput=5.0)),
        "zero-goodput baseline": Result(trials=[T()], incumbent={}, visited=["n"],
                                        skipped=[], launches=1, minutes=1.0,
                                        baseline=T(goodput=0.0)),
        "inf latency trial": Result(trials=[T(ttft_p99_ms=float("inf"), slo_ok=False)],
                                    incumbent={}, visited=["n"], skipped=[],
                                    launches=1, minutes=1.0),
    }
    for name, r in cases.items():
        try:
            report(r, log=lambda *a: None, demand_tok_s=4159.0)
            check(f"report survives: {name}", True)
        except Exception as e:
            check(f"report survives: {name}", False, f"{type(e).__name__}: {e}")
    # and with an incumbent curve, including a peak at the edge
    try:
        report(cases["with baseline"], log=lambda *a: None, demand_tok_s=4159.0,
               incumbent_curve=[{"concurrency": 2, "goodput": 40.0,
                                 "ttft_p99_ms": 300.0, "slo_attainment": 1.0}])
        check("report survives a single-point curve", True)
    except Exception as e:
        check("report survives a single-point curve", False, f"{type(e).__name__}: {e}")

if __name__ == "__main__":
    sys.exit(main())
