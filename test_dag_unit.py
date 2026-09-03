"""Probing unit tests for the DAG machinery. No GPU, no server, seconds to run.

    python test_dag_unit.py

Covers the pure logic that decides what gets measured and what survives:

    predicates.py   expression parsing, schema checking, evaluation
    traverse.py     _value, _variants, Trial.axes, Result.frontier
    dag/llm.json    structural invariants the validator does not assert

These are adversarial on purpose. selftest.py checks that the happy path works;
this file tries to break things. Every case here is either an invariant the code
must hold or a shape that has already caused a wrong number somewhere in this
project.
"""

from __future__ import annotations

import json
import math
import sys
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


def raises(fn, *exc) -> bool:
    try:
        fn()
    except (exc or (Exception,)):
        return True
    except Exception:
        return False
    return False


# ==========================================================================
def test_predicates():
    section("predicates: parsing and schema checking")
    from predicates import Predicate, PredicateError

    ok = [
        "workload.prefix_overlap > 0.05",
        "fingerprint.model.n_params_b >= 7 and fingerprint.hw.memory_gb > 40",
        "not fingerprint.model.is_dense",
        "workload.p99_input_tokens > 1024 or workload.mean_output_tokens > 64",
        "fingerprint.model.weight_gb * 2 < fingerprint.hw.memory_gb",
        "1 if fingerprint.model.is_dense else 2",
        "fingerprint.model.attention_type in ('mha', 'gqa')",
    ]
    for e in ok:
        try:
            Predicate(e)
        except Exception as ex:
            check(f"parses: {e}", False, f"{type(ex).__name__}: {ex}")

    # Expressions that must be REFUSED. A predicate language that can import or
    # call arbitrary code is a DAG file that can run anything.
    evil = [
        "__import__('os').system('true')",
        "open('/etc/passwd').read()",
        "(lambda: 1)()",
        "[x for x in range(10)]",
        "fingerprint.model.__class__.__mro__",
        "eval('1+1')",
        "globals()",
    ]
    for e in evil:
        p = None
        try:
            p = Predicate(e)
        except Exception:
            continue                       # refused at parse: good
        # If it parsed, evaluating it must still fail rather than execute.
        from fingerprint import Context
        check(f"refuses to execute: {e}",
              raises(lambda: p.evaluate(_ctx())),
              "parsed AND evaluated -- the predicate language executes arbitrary code")

    # Typos must be caught by check(), which is the entire reason it exists:
    # a mistyped path silently disables a node otherwise.
    from predicates import resolve_path_type
    bad_paths = ["workload.prefix_overlapp", "fingerprint.model.n_param_b",
                 "fingerprint.hw.memry_gb", "nonexistent.field", "model.n_params_b"]
    for bp in bad_paths:
        errs = Predicate(f"{bp} > 1").check(set())
        check(f"check() flags bad path {bp!r}", bool(errs), "no error reported")

    good_paths = ["workload.prefix_overlap", "fingerprint.model.n_params_b",
                  "fingerprint.hw.memory_gb", "slo.ttft_p99_ms"]
    for gp in good_paths:
        errs = Predicate(f"{gp} > 1").check(set())
        check(f"check() accepts {gp!r}", not errs, f"reported {errs}")


def _ctx(**over):
    """A Context with a real fingerprint, cheap and offline."""
    from fingerprint import (Context, Fingerprint, HardwareFingerprint,
                             ModelFingerprint, WorkloadFingerprint, LoraFingerprint, SLO)
    model = ModelFingerprint(
        id="test/model", architecture="TestForCausalLM", n_params_b=14.0,
        n_layers=40, hidden_size=5120, n_heads=40, n_kv_heads=8,
        attention_type="gqa", max_model_len=32768, bytes_per_param=2.0)
    hw = HardwareFingerprint(
        gpu_name="TestGPU", gpu_count=1, compute_capability="9.0",
        memory_gb=80.0, memory_bandwidth_gb_s=3350.0, unified_memory=False,
        system_ram_gb=200.0, cpu_cores=32)
    wl = WorkloadFingerprint(
        n_requests=800, mean_input_tokens=620.0, p99_input_tokens=2660,
        p999_input_tokens=4380, mean_output_tokens=260.0, p99_output_tokens=804,
        request_rate_qps=16.0, max_concurrency=32, burstiness=1.5,
        prefix_overlap=0.31, prefix_overlap_per_adapter=0.31, multi_turn=False,
        greedy=True, temperature=0.0, top_p=1.0, structured_generation=0.0,
        trace_ref="x")
    fp = Fingerprint(model=model, hw=hw, workload=wl, lora=LoraFingerprint())
    slo = SLO(ttft_p99_ms=500, itl_p99_ms=250, quality_budget=0.1,
              lossless_quality_budget=0.03)
    c = Context(fingerprint=fp, slo=slo)
    for k, v in over.items():
        setattr(c, k, v)
    return c


def test_predicate_eval():
    section("predicates: evaluation against a Context")
    from predicates import Predicate
    ctx = _ctx()
    cases = [
        ("workload.prefix_overlap > 0.05", True),
        ("workload.prefix_overlap > 0.5", False),
        ("fingerprint.model.n_params_b >= 7 and fingerprint.hw.memory_gb > 40", True),
        ("not fingerprint.model.is_dense", False),
        ("fingerprint.model.weight_gb * 2 < fingerprint.hw.memory_gb", True),
        ("fingerprint.model.weight_gb * 4 < fingerprint.hw.memory_gb", False),
        ("workload.p99_input_tokens > 1024", True),
        ("fingerprint.model.attention_type in ('mha', 'gqa')", True),
        ("fingerprint.model.attention_type in ('mla',)", False),
    ]
    for expr, want in cases:
        try:
            got = Predicate(expr).evaluate(ctx)
        except Exception as e:
            check(f"eval {expr}", False, f"{type(e).__name__}: {e}")
            continue
        check(f"eval {expr} == {want}", bool(got) == want, f"got {got!r}")

    # Division by zero must not take down a traversal 9 launches in.
    z = _ctx()
    z.fingerprint.workload.mean_output_tokens = 0.0
    check("division by zero raises rather than returning garbage",
          raises(lambda: Predicate("100 / workload.mean_output_tokens").evaluate(z)),
          "should raise ZeroDivisionError, not silently produce inf")


# ==========================================================================
def test_value():
    section("traverse._value: expression detection")
    from traverse import _value
    ctx = _ctx()

    # Plain strings that CONTAIN punctuation must survive unchanged. The
    # detector is a heuristic -- any of "+-*/()" -- so every value that happens
    # to contain a hyphen goes down the expression path and must come back out.
    literals = ["Qwen/Qwen3-14B", "fp8-dynamic", "auto", "w4a16",
                "/abs/path/model", "float(-inf)", "a-b-c", "e5-mistral",
                "meta-llama/Llama-3.1-8B", ""]
    for lit in literals:
        got = _value(lit, ctx)
        check(f"literal survives: {lit!r}", got == lit, f"became {got!r}")

    # Real expressions must evaluate.
    check("expression evaluates", _value("2 * 3", ctx) == 6, f"{_value('2 * 3', ctx)}")
    check("expression reads the fingerprint",
          _value("fingerprint.model.n_layers * 2", ctx) == 80, f"{_value('fingerprint.model.n_layers * 2', ctx)}")

    # Non-strings pass through untouched, including the falsy ones.
    for v in (0, False, None, 1.5, True, [], {}):
        check(f"non-string passes through: {v!r}", _value(v, ctx) == v or
              (_value(v, ctx) is v), f"became {_value(v, ctx)!r}")


def test_variants():
    section("traverse._variants: config construction")
    from traverse import _variants
    ctx = _ctx()
    base = {"a": 1, "keep": "me"}

    n = {"id": "x", "action": {"set": {"b": 2}}}
    v = _variants(n, base, ctx)
    check("no sweep -> one variant", len(v) == 1, f"{len(v)}")
    check("base keys preserved", v[0]["keep"] == "me" and v[0]["a"] == 1, f"{v[0]}")
    check("action applied", v[0]["b"] == 2, f"{v[0]}")
    check("base not mutated", "b" not in base, f"base is {base}")

    n = {"id": "x", "action": {"set": {"b": 2}}, "sweep": [{"c": 1}, {"c": 2}, {"c": 3}]}
    v = _variants(n, base, ctx)
    check("sweep -> one variant per entry", len(v) == 3, f"{len(v)}")
    check("each carries the action", all(x["b"] == 2 for x in v), f"{v}")
    check("sweep values differ", [x["c"] for x in v] == [1, 2, 3], f"{v}")
    check("variants are independent objects",
          len({id(x) for x in v}) == 3, "variants share a dict")

    # Nested keys: "speculative_config.num_speculative_tokens"
    n = {"id": "x", "action": {"set": {"speculative_config": {"method": "ngram"}}},
         "sweep": [{"speculative_config.num_speculative_tokens": 3},
                   {"speculative_config.num_speculative_tokens": 5}]}
    v = _variants(n, base, ctx)
    check("nested sweep merges rather than replacing",
          all(x["speculative_config"].get("method") == "ngram" for x in v),
          f"{[x.get('speculative_config') for x in v]}")
    check("nested sweep sets the leaf",
          [x["speculative_config"]["num_speculative_tokens"] for x in v] == [3, 5],
          f"{v}")
    # THE ALIASING TRAP: two variants must not share one nested dict.
    v[0]["speculative_config"]["num_speculative_tokens"] = 999
    check("nested dicts are not shared between variants",
          v[1]["speculative_config"]["num_speculative_tokens"] == 5,
          "mutating one variant changed another -- they alias the same dict")

    # An empty sweep list is not the same as no sweep.
    n = {"id": "x", "action": {"set": {"b": 2}}, "sweep": []}
    v = _variants(n, base, ctx)
    check("empty sweep still yields a measurable variant", len(v) == 1,
          f"{len(v)} variants -- an empty sweep would skip the node entirely")


# ==========================================================================
def _t(node_id="n", goodput=10.0, ttft=100.0, itl=10.0, mem=10.0,
       quality=None, slo_ok=True, **kw):
    from traverse import Trial
    return Trial(node_id=node_id, config=kw.pop("config", {}), goodput=goodput,
                 ttft_p99_ms=ttft, itl_p99_ms=itl, memory_gb=mem,
                 quality=quality if quality is not None else {"math_500": 0.5},
                 slo_ok=slo_ok, **kw)


def test_trial_axes():
    section("Trial: axes and min_quality")
    t = _t(quality={"a": 0.9, "b": 0.4})
    check("min_quality is the WORST benchmark, not the mean",
          t.min_quality == 0.4, f"{t.min_quality}")

    empty = _t(quality={})
    check("a trial with NO quality does not claim perfect quality",
          empty.min_quality < 1.0,
          f"min_quality={empty.min_quality} -- an unmeasured trial dominates every "
          f"measured one on the quality axis and lands on the frontier for free")

    ax = _t().axes()
    from traverse import OBJECTIVES
    check("axes covers exactly the objectives", set(ax) == set(OBJECTIVES),
          f"axes={sorted(ax)} objectives={sorted(OBJECTIVES)}")


def test_frontier():
    section("Result.frontier: non-domination")
    from traverse import Result

    def R(trials):
        return Result(trials=trials, incumbent={}, visited=[], skipped=[],
                      launches=0, minutes=0.0)

    check("empty stays empty", R([]).frontier() == [])

    # b strictly better on every axis -> a is dominated.
    a = _t("a", goodput=10, ttft=200, itl=20, mem=20, quality={"q": 0.5})
    b = _t("b", goodput=20, ttft=100, itl=10, mem=10, quality={"q": 0.9})
    fr = R([a, b]).frontier()
    check("strictly dominated point is excluded",
          [t.node_id for t in fr] == ["b"], f"{[t.node_id for t in fr]}")

    # A trade is NOT domination: worse goodput, better ttft.
    a = _t("fast", goodput=10, ttft=50, itl=10, mem=10, quality={"q": 0.5})
    b = _t("big", goodput=100, ttft=500, itl=10, mem=10, quality={"q": 0.5})
    fr = R([a, b]).frontier()
    check("a genuine trade keeps both", len(fr) == 2, f"{[t.node_id for t in fr]}")

    # Identical trials: neither dominates (no strict improvement), so both stay.
    a, b = _t("x"), _t("y")
    fr = R([a, b]).frontier()
    check("identical points do not eliminate each other", len(fr) == 2,
          f"{[t.node_id for t in fr]} -- ties must not be dropped")

    # slo_ok=False is excluded entirely, even if it dominates.
    good = _t("ok", goodput=10)
    bad = _t("failed", goodput=9999, ttft=1, itl=1, mem=1,
             quality={"q": 1.0}, slo_ok=False)
    fr = R([good, bad]).frontier()
    check("SLO failures never reach the frontier",
          [t.node_id for t in fr] == ["ok"], f"{[t.node_id for t in fr]}")

    # A launch failure (goodput 0, inf latency) must not survive.
    dead = _t("dead", goodput=0.0, ttft=float("inf"), itl=float("inf"),
              mem=0.0, quality={}, slo_ok=False)
    live = _t("live", goodput=50)
    fr = R([dead, live]).frontier()
    check("a dead launch is not a frontier point",
          [t.node_id for t in fr] == ["live"], f"{[t.node_id for t in fr]}")

    # NON-FINITE AXES. Every comparison against NaN is False, so worse_none is
    # False and a NaN point is never dominated -- it survives against anything.
    # The evaluator emits NaN percentiles whenever a window completes zero
    # requests, so this is a shape the code actually produces.
    nan = float("nan")
    strong = _t("strong", goodput=100.0)
    nanny = _t("nan_ttft", goodput=1.0, ttft=nan)
    fr = R([strong, nanny]).frontier()
    check("a NaN axis does not buy a frontier slot",
          [t.node_id for t in fr] == ["strong"],
          f"{[t.node_id for t in fr]} -- goodput 1.0 with a NaN TTFT must not "
          f"survive against goodput 100.0")
    infy = _t("inf_ttft", goodput=1.0, ttft=float("inf"))
    fr = R([strong, infy]).frontier()
    check("an infinite axis does not buy a frontier slot either",
          [t.node_id for t in fr] == ["strong"], f"{[t.node_id for t in fr]}")
    # itl_p99_ms is NOT a frontier axis -- OBJECTIVES is goodput, quality,
    # ttft_p99_ms and memory_gb -- so a NaN there is irrelevant to domination
    # and must not silently exclude an otherwise good point.
    fr = R([_t("nan_itl_only", goodput=50.0, itl=nan)]).frontier()
    check("a NaN on a NON-axis field does not exclude the trial",
          [t.node_id for t in fr] == ["nan_itl_only"], f"{[t.node_id for t in fr]}")
    # A run where EVERY trial has a non-finite AXIS yields an empty frontier
    # rather than an arbitrary one.
    fr = R([nanny, _t("also_nan", goodput=nan)]).frontier()
    check("all-non-finite yields an empty frontier, not a guess", fr == [],
          f"{[t.node_id for t in fr]}")

    # Sorted by goodput descending.
    ts = [_t("lo", goodput=1), _t("hi", goodput=100), _t("mid", goodput=50)]
    fr = R(ts).frontier()
    check("frontier is sorted by goodput descending",
          [t.goodput for t in fr] == sorted([t.goodput for t in fr], reverse=True),
          f"{[t.goodput for t in fr]}")

    # Every frontier member must actually be non-dominated -- brute force.
    import random
    rng = random.Random(0)
    ts = [_t(f"n{i}", goodput=rng.uniform(1, 100), ttft=rng.uniform(10, 1000),
             itl=rng.uniform(1, 100), mem=rng.uniform(1, 100),
             quality={"q": rng.uniform(0, 1)}) for i in range(60)]
    fr = R(ts).frontier()
    from traverse import OBJECTIVES
    def dominates(x, y):
        xa, ya = x.axes(), y.axes()
        return (all((xa[k] >= ya[k]) if d == "max" else (xa[k] <= ya[k])
                    for k, d in OBJECTIVES.items())
                and any((xa[k] > ya[k]) if d == "max" else (xa[k] < ya[k])
                        for k, d in OBJECTIVES.items()))
    bad = [t.node_id for t in fr if any(dominates(o, t) for o in ts if o is not t)]
    check("no frontier member is dominated by anything", not bad, f"dominated: {bad}")
    missing = [t.node_id for t in ts
               if not any(dominates(o, t) for o in ts if o is not t) and t not in fr]
    check("no non-dominated point is missing from the frontier", not missing,
          f"missing: {missing}")


# ==========================================================================
def test_dag_file():
    section("dag/llm.json: structural invariants")
    d = json.loads(Path("dag/llm.json").read_text())
    nodes = {n["id"]: n for n in d["nodes"]}
    check("node ids are unique", len(nodes) == len(d["nodes"]),
          f"{len(d['nodes'])} nodes, {len(nodes)} unique ids")

    # Every edge target exists.
    for n in d["nodes"]:
        for edge in ("on_keep", "on_revert"):
            for tgt in (n.get(edge) or []):
                check(f"{n['id']}.{edge} -> {tgt} exists", tgt in nodes or tgt == "frontier",
                      "dangling edge")
        for req in (n.get("requires") or []):
            check(f"{n['id']}.requires {req} exists", req in nodes or req == "incumbent",
                  "dangling requirement")

    # Every benchmark a node asks for must be registered in quality.py, or the
    # traversal raises at the first quality node -- hours in.
    from quality import BENCHMARKS
    for n in d["nodes"]:
        for b in (n.get("quality_benchmarks") or []):
            check(f"{n['id']} asks for a registered benchmark: {b}", b in BENCHMARKS,
                  f"not in quality.py ({sorted(BENCHMARKS)})")
    for b in d.get("benchmarks", []):
        check(f"declared benchmark {b['id']} is registered", b["id"] in BENCHMARKS,
              f"declared in the DAG but absent from quality.py")

    # Every predicate parses AND type-checks against the real schema.
    from predicates import Predicate
    ids = set(nodes) | {"incumbent"}
    for n in d["nodes"]:
        e = n.get("applicable_when")
        if not e:
            continue
        try:
            errs = Predicate(e).check(ids)
        except Exception as ex:
            check(f"{n['id']}.applicable_when parses", False, f"{type(ex).__name__}: {ex}")
            continue
        check(f"{n['id']}.applicable_when type-checks", not errs, f"{errs}")

    # Probes must be ones the evaluator implements.
    KNOWN = {"goodput", "equivalence", "quality"}
    for n in d["nodes"]:
        for p in (n.get("probes") or []):
            check(f"{n['id']} uses a known probe: {p}", p in KNOWN, f"unknown probe")

    # class must be one traverse() understands; it branches on "lossy" and
    # "checkpoint" by string, so a typo silently downgrades a lossy node to one
    # with no quality gate at all.
    KNOWN_CLASS = {"lossless", "lossy", "checkpoint", "terminal", "root"}
    for n in d["nodes"]:
        c = n.get("class")
        check(f"{n['id']}.class is known: {c!r}", c in KNOWN_CLASS,
              f"traverse() gates quality on class == 'lossy'; an unknown class "
              f"means a weight-rewriting node runs with NO quality gate")

    # A lossy node without a quality benchmark cannot be gated at all.
    for n in d["nodes"]:
        if n.get("class") == "lossy" and n.get("status") == "active":
            check(f"lossy node {n['id']} declares a quality benchmark",
                  bool(n.get("quality_benchmarks")),
                  "a lossy node with no benchmark is kept on goodput alone")

    # Sweeps must be lists of dicts; a bare list of scalars silently produces
    # variants that are all identical to the base config.
    for n in d["nodes"]:
        sw = n.get("sweep")
        if sw is None:
            continue
        check(f"{n['id']}.sweep is a list", isinstance(sw, list), f"{type(sw)}")
        check(f"{n['id']}.sweep entries are dicts",
              all(isinstance(e, dict) for e in sw),
              f"got {[type(e).__name__ for e in sw]} -- scalars produce identical variants")
        if n.get("status") == "active":
            check(f"active node {n['id']} has a non-empty sweep", len(sw) > 0,
                  "an empty sweep on an ACTIVE node measures nothing new")


def test_requires_matches_edges():
    section("dag: `requires` agrees with the actual edges")
    import networkx as nx
    d = json.loads(Path("dag/llm.json").read_text())
    nodes = {n["id"]: n for n in d["nodes"]}
    g = nx.DiGraph()
    for i in nodes:
        g.add_node(i)
    for i, n in nodes.items():
        for e in ("on_keep", "on_revert"):
            for t in (n.get(e) or []):
                if t in nodes:
                    g.add_edge(i, t)
    # requires is documentation people read to understand ordering. It was
    # checked only for EXISTENCE, so it was free to state the opposite of the
    # real order: prefix_caching declared requires=[chunked_prefill] while
    # running two nodes ahead of it, contradicting its own rationale.
    for i, n in nodes.items():
        for req in (n.get("requires") or []):
            if req not in nodes or req == i:
                continue
            check(f"{i} requires {req}, which is an ancestor",
                  nx.has_path(g, req, i),
                  f"no on_keep/on_revert path from {req} to {i} -- the stated "
                  f"prerequisite runs AFTER the node that claims it")


def test_reachability():
    section("dag/llm.json: reachability and termination")
    d = json.loads(Path("dag/llm.json").read_text())
    nodes = {n["id"]: n for n in d["nodes"]}
    start = d.get("traversal", {}).get("start") or "incumbent"

    # Walk both edges from the start; every active node should be reachable.
    seen, stack = set(), [start]
    while stack:
        cur = stack.pop()
        if cur in seen or cur not in nodes:
            continue
        seen.add(cur)
        for edge in ("on_keep", "on_revert"):
            stack.extend(nodes[cur].get(edge) or [])
    unreachable = [i for i, n in nodes.items()
                   if i not in seen and n.get("status") == "active"]
    check("every active node is reachable from the start", not unreachable,
          f"unreachable: {unreachable}")

    # Both edges must terminate. A node with neither ends the traversal; that is
    # only correct for a terminal node.
    for i, n in nodes.items():
        if n.get("status") != "active":
            continue
        has = bool(n.get("on_keep")) or bool(n.get("on_revert"))
        check(f"{i} either continues or is terminal",
              has or n.get("class") in ("terminal", "checkpoint") or i == "frontier",
              "no outgoing edge and not marked terminal -- the traversal stops here")


# ==========================================================================
def main() -> int:
    for fn in (test_predicates, test_predicate_eval, test_value, test_variants,
               test_trial_axes, test_frontier, test_dag_file,
               test_requires_matches_edges, test_reachability):
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
    print("  all unit checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
