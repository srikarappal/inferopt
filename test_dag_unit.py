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

    # THE SUPPORTED SURFACE, pinned. A DAG author needs to know what is legal,
    # and a silent change here turns a working predicate into a skipped node.
    from fingerprint import Context
    surface = [
        ("2 ** 3", 8), ("7 // 2", 3), ("7 % 3", 1), ("-5", -5),
        ("1 < 2 < 3", True), ("'a' == 'a'", True),
        ("max(1, 2)", 2), ("min(1, 2)", 1), ("abs(-3)", 3),
        ("len('abc')", 3), ("round(1.6)", 2), ("int(1.9)", 1), ("float(1)", 1.0),
        # `is` is deliberately NOT permitted; None is compared with ==, which
        # works because the schema uses Optional rather than sentinels.
        ("slo.throughput_target_tok_s == None", True),
    ]
    for expr, want in surface:
        try:
            got = Predicate(expr).evaluate(_ctx())
            check(f"supported: {expr}", got == want, f"got {got!r}, want {want!r}")
        except Exception as ex:
            check(f"supported: {expr}", False, f"{type(ex).__name__}: {ex}")
    check("`is` stays refused",
          raises(lambda: Predicate("None is None")),
          "identity comparison has no place in a config predicate")

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

    # Pure arithmetic evaluates even with no fingerprint reference.
    check("pure arithmetic evaluates", _value("1-2", ctx) == -1, f"{_value('1-2', ctx)}")

    # A BROKEN EXPRESSION MUST RAISE, not become a config literal. It used to
    # be handed to vLLM as the string "workload.nonexistent * 2", where it is
    # either rejected as an unknown value or, worse, accepted as one.
    import traceback
    try:
        got = _value("workload.nonexistent * 2", ctx)
        check("a broken expression raises rather than leaking a string", False,
              f"returned {got!r}")
    except ValueError:
        check("a broken expression raises rather than leaking a string", True)
    except Exception as e:
        check("a broken expression raises ValueError specifically", False,
              f"{type(e).__name__}: {e}")

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
def test_pb_design():
    """Plackett-Burman screening: the design, and that it recovers known effects.

    The method rests entirely on ORTHOGONALITY -- each factor ON in half the
    runs, and every PAIR of factors balanced across those halves. If that
    breaks, the difference of means stops isolating one factor and silently
    starts measuring a blend of several, with no symptom at all: the table
    still prints, the numbers still look like effects, and they are wrong.
    """
    section("plackett-burman: design properties")
    from pb_screen import pb_design, effects

    for nf, want_n in ((5, 12), (8, 12), (11, 12), (12, 20), (15, 20)):
        design, n = pb_design(nf)
        check(f"{nf} factors -> N={want_n}", n == want_n, f"got {n}")
        check(f"N={n}: one row per run", len(design) == n, f"got {len(design)}")
        check(f"N={n}: every row covers every factor",
              all(len(r) == nf for r in design), "ragged design matrix")
        cols = [[design[r][c] for r in range(n)] for c in range(nf)]
        check(f"N={n}: every factor ON in exactly half the runs",
              all(sum(c) == n // 2 for c in cols),
              f"ON counts {sorted({sum(c) for c in cols})}, want {n//2}")
        # THE property. Without pairwise balance the main effects are
        # confounded with EACH OTHER, not merely with interactions, and the
        # whole screen is void.
        bad = [(i, j) for i in range(nf) for j in range(i + 1, nf)
               if sum(1 for r in range(n)
                      if design[r][i] and design[r][j]) != n // 4]
        check(f"N={n}: every PAIR of factors is balanced", not bad,
              f"{len(bad)} unbalanced pairs, e.g. {bad[:3]}")
        check(f"N={n}: exactly one all-off row",
              sum(1 for r in design if not any(r)) == 1,
              "the all-off row anchors the design to the stock config")
        check(f"N={n}: no row turns everything on",
              not any(all(r) for r in design),
              "an all-on row is the YOLO experiment, not a screen")
        check(f"N={n}: no two rows identical",
              len({tuple(r) for r in design}) == n, "a repeated row wastes a launch")

    check("more factors than the largest design raises",
          raises(lambda: pb_design(64)),
          "must refuse rather than silently screen a truncated factor set")

    section("plackett-burman: recovers known effects")
    import random
    TRUTH = {"a": 35.0, "b": 20.0, "c": -12.0, "d": -8.0, "e": 2.0, "f": 0.0}
    names = list(TRUTH)
    fs = [{"id": k} for k in names]
    design, _ = pb_design(len(names))
    clean = [40.0 + sum(TRUTH[names[c]] for c in range(len(names)) if row[c])
             for row in design]

    # Noise-free, an orthogonal design must recover every effect EXACTLY.
    got = {x["id"]: x["effect"] for x in effects(design, fs, clean)}
    worst = max(abs(got[k] - TRUTH[k]) for k in TRUTH)
    check("noise-free recovery is exact", worst < 1e-9,
          f"largest error {worst:.6g} -- design is not orthogonal")
    check("a truly dead factor reads as zero", abs(got["f"]) < 1e-9,
          f"got {got['f']}")
    check("effects are returned largest-magnitude first",
          [x["id"] for x in effects(design, fs, clean)]
          == sorted(TRUTH, key=lambda k: -abs(TRUTH[k])),
          "the ranking IS the output of a screen")

    # At the across-launch spread we actually measured (~5%), the RANKING must
    # survive, which is all a screen is asked for.
    rng = random.Random(0)
    noisy = [g * (1 + rng.gauss(0, 0.05)) for g in clean]
    order = [x["id"] for x in effects(design, fs, noisy)]
    check("under 5% noise the two largest effects still rank first",
          set(order[:2]) == {"a", "b"}, f"got {order}")
    small = [x for x in effects(design, fs, noisy) if x["id"] in ("e", "f")]
    check("under noise, small effects land inside their own error bars",
          all(abs(x["effect"]) < 2 * x["se"] for x in small),
          "a screen that calls a 2-unit effect resolved at 5% noise is lying")

    # A launch that died must not be scored as zero goodput -- that would
    # fabricate an enormous negative effect for every factor ON in that row.
    holed = list(clean); holed[3] = None
    e = effects(design, fs, holed)
    check("a failed row is dropped, not counted as zero",
          all(x["n_on"] + x["n_off"] == len(design) - 1 for x in e),
          f"row counts {[(x['n_on'], x['n_off']) for x in e][:3]}")
    check("effects stay roughly right with one row missing",
          max(abs(x["effect"] - TRUTH[x["id"]]) for x in e) < 12.0,
          "one lost row should perturb, not destroy, the screen")

    allgone = effects(design, fs, [None] * len(design))
    check("every row failing does not crash the screen", len(allgone) == len(fs))
    check("every row failing yields no effect at all",
          all(x["effect"] is None for x in allgone),
          "no data must read as unknown, never as zero")

    # The error bar is not decoration: it is what separates "RESOLVED" from
    # "inside the noise", which is the entire output of a screen. Two mutants
    # survived the checks above -- dropping the sqrt (reporting a variance as
    # if it were a standard error) and computing the spread from only one of
    # the two groups. Both make the screen over-confident, and neither shows
    # up as anything but a slightly different number.
    section("plackett-burman: the error bar itself")
    base = [40.0 + (17.0 if row[0] else 0.0) + (3.0 if row[1] else 0.0)
            for row in design]
    jitter = [1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 1.5, -1.5, 0.25, -0.25, 3.0, -3.0]
    obs = [b + j for b, j in zip(base, jitter)]
    e0 = {x["id"]: x for x in effects(design, fs, obs)}["a"]

    # A standard error carries the same units as the effect. Scale every
    # goodput by k and both must scale by k -- a variance would scale by k*k.
    e2 = {x["id"]: x for x in effects(design, fs, [v * 3 for v in obs])}["a"]
    check("se scales linearly with the data, like the effect does",
          abs(e2["se"] - 3 * e0["se"]) < 1e-9,
          f"se {e0['se']:.4f} -> {e2['se']:.4f}; x9 means a variance is being "
          f"reported as a standard error")
    check("effect scales linearly too", abs(e2["effect"] - 3 * e0["effect"]) < 1e-9)

    # Both groups contribute. Widen ONLY the off rows: the error on the
    # difference must grow, or the off group is being ignored.
    wide = [v + (8.0 if (i % 2 and not design[i][0]) else 0.0)
            for i, v in enumerate(obs)]
    ew = {x["id"]: x for x in effects(design, fs, wide)}["a"]
    check("noise in the OFF group alone still widens the error bar",
          ew["se"] > e0["se"] + 1e-9,
          f"se {e0['se']:.4f} -> {ew['se']:.4f}; the off group is not counted")

    won = [v + (8.0 if (i % 2 and design[i][0]) else 0.0)
           for i, v in enumerate(obs)]
    en = {x["id"]: x for x in effects(design, fs, won)}["a"]
    check("noise in the ON group alone still widens the error bar",
          en["se"] > e0["se"] + 1e-9,
          f"se {e0['se']:.4f} -> {en['se']:.4f}; the ON group is not counted")

    # And more runs at the same spread must shrink it.
    d20, _ = pb_design(15)
    f20 = [{"id": f"x{i}"} for i in range(15)]
    mk = lambda dd: [40.0 + (17.0 if r[0] else 0.0) + (1.0 if i % 2 else -1.0)
                     for i, r in enumerate(dd)]
    s12 = {x["id"]: x for x in effects(design, fs, mk(design))}["a"]["se"]
    s20 = {x["id"]: x for x in effects(d20, f20, mk(d20))}["x0"]["se"]
    check("more runs at the same spread give a tighter error bar", s20 < s12,
          f"N=12 se {s12:.4f} vs N=20 se {s20:.4f} -- averaging is not helping")


# ==========================================================================
def test_replay():
    """The optimizer-scoring harness. Its job is to be trusted about which
    strategy wins, so its own arithmetic has to be beyond doubt."""
    section("replay: the table")
    import random as _r
    from replay import (Table, regret, sequential_dag, pb_then_factorial,
                        pb_anchored, random_search, yolo, screen_fidelity,
                        STRATEGIES, _spearman)

    EFF = {"a": 10.0, "b": 5.0, "c": -3.0, "d": 0.0}
    t = Table.synthetic(EFF, base=20.0, noise=0.0, repeats=2, seed=1)
    check("synthetic table has 2^n cells", len(t.cells) == 16, f"{len(t.cells)}")
    check("all-off cell is the base", abs(t.truth("0000") - 20.0) < 1e-9)
    check("truth adds the stated effects",
          abs(t.truth("1100") - 35.0) < 1e-9, f"{t.truth('1100')}")
    check("a negative factor lowers truth",
          abs(t.truth("0010") - 17.0) < 1e-9, f"{t.truth('0010')}")
    m, g = t.virtual_best()
    # d has effect 0.0, so 1100 and 1101 are genuinely tied. Either is correct;
    # asserting one of them would be testing dict order, not the function.
    check("virtual best is a max-truth cell",
          m in ("1100", "1101") and abs(g - 35.0) < 1e-9, f"got {m} at {g}")
    check("true_effects recovers an additive model exactly",
          all(abs(t.true_effects()[k] - v) < 1e-9 for k, v in EFF.items()),
          f"{t.true_effects()}")
    check("an empty table is refused", raises(lambda: Table(["a"], {})),
          "scoring against nothing must not silently return 0 regret")

    ti = Table.synthetic({"a": 1.0, "b": 1.0}, base=10.0, noise=0.0,
                         interactions={("a", "b"): 8.0}, seed=1)
    check("interactions are applied only when both factors are on",
          abs(ti.truth("11") - 20.0) < 1e-9 and abs(ti.truth("10") - 11.0) < 1e-9,
          f"11={ti.truth('11')} 10={ti.truth('10')}")

    # A MEASURED table has no analytic truth -- its truth is the mean of the
    # repeats, which is why repeats are not optional. Every synthetic table
    # short-circuits that path, so it needs a table built from cells directly.
    meas = Table(["x", "y"], {"00": [10.0, 12.0, 14.0], "01": [20.0, 20.0, 20.0],
                              "10": [9.0, 9.0, 9.0], "11": [30.0, 10.0, 20.0]})
    check("a measured cell's truth is the MEAN of its repeats",
          abs(meas.truth("00") - 12.0) < 1e-9, f"got {meas.truth('00')}")
    check("a wide spread does not raise a cell's truth",
          abs(meas.truth("11") - 20.0) < 1e-9,
          f"got {meas.truth('11')}; taking the max would reward a lucky launch")
    check("virtual best on a measured table uses those means",
          meas.virtual_best()[0] in ("01", "11"), f"{meas.virtual_best()}")

    section("replay: regret is scored on truth, not on what was observed")
    check("a trace holding only the best cell has zero regret",
          abs(regret(t, [("1101", 99.0)])) < 1e-9)
    check("an empty trace is total regret, not zero",
          regret(t, []) == 1.0, "a strategy that measured nothing must not win")
    # THE property. A strategy that gets a lucky draw on a bad cell must be
    # charged for the bad cell it would ship, not credited with the lucky number.
    lucky = regret(t, [("0000", 999.0), ("1101", 1.0)])
    check("a lucky reading on a bad cell is still scored as the bad cell",
          abs(lucky - (35.0 - 20.0) / 35.0) < 1e-9,
          f"got {lucky}; scoring on observed values would reward noise")
    check("regret rises as the shipped cell gets worse",
          regret(t, [("0010", 1.0)]) > regret(t, [("1000", 1.0)]) > 0,
          "1100 is itself optimal, so it cannot be the worse of the pair")

    section("replay: strategies are budget-honest")
    for name, fn in STRATEGIES.items():
        for b in (1, 3, 7, 12, 20):
            tr = fn(t, b, _r.Random(0))
            check(f"{name}: respects a budget of {b}", len(tr) <= b,
                  f"spent {len(tr)}")
            check(f"{name}: never invents a cell (budget {b})",
                  all(m in t.cells for m, _ in tr), "measured a nonexistent config")
        check(f"{name}: returns nothing on a zero budget",
              fn(t, 0, _r.Random(0)) == [], "a free lunch is a bug")

    # A half-fraction table has holes. A strategy must skip them, not crash and
    # not fabricate -- this is what a real partially-measured table looks like.
    holed = Table(t.factors, {k: v for i, (k, v) in enumerate(t.cells.items())
                              if i % 2 == 0})
    for name, fn in STRATEGIES.items():
        tr = fn(holed, 12, _r.Random(0))
        check(f"{name}: survives an incompletely measured table",
              all(m in holed.cells for m, _ in tr), "read a hole as a number")

    section("replay: the sequential walk behaves as traverse.py does")
    walk = sequential_dag(t, 99, _r.Random(0))
    check("the walk costs one launch per factor, plus the baseline",
          len(walk) == len(t.factors) + 1, f"spent {len(walk)}")
    check("the walk starts from all-off", walk[0][0] == "0000", f"{walk[0][0]}")
    check("extra budget does not help the walk",
          len(sequential_dag(t, 500, _r.Random(0))) == len(walk),
          "the walk cannot spend more than one pass -- that is the point")
    # It must actually turn on a factor worth far more than the band.
    check("the walk keeps a large win", walk[-1][0].count("1") >= 1,
          f"ended at {walk[-1][0]} having seen a +10 factor")

    # The band is the walk's whole decision rule. On a table where every factor
    # pays less than it, the walk must end where it started -- otherwise it is
    # accepting noise, which is the failure the band exists to prevent.
    # The incumbent is not in the trace, but it is visible in it: while the
    # incumbent stays all-off, every candidate carries exactly one 1. The moment
    # a factor is kept, later candidates carry two. So the maximum popcount over
    # the trace says whether anything was ever accepted.
    tiny = Table.synthetic({"a": 0.2, "b": 0.3, "c": 0.1}, base=100.0,
                           noise=0.0, repeats=1, seed=5)
    w = sequential_dag(tiny, 99, _r.Random(0), band=0.05)
    check("the walk accepts no gain smaller than the band",
          max(m.count("1") for m, _ in w) <= 1,
          f"trace {[m for m, _ in w]} -- a 0.3% gain was kept against a 5% band")
    big = Table.synthetic({"a": 40.0, "b": 30.0}, base=100.0,
                          noise=0.0, repeats=1, seed=5)
    wb = sequential_dag(big, 99, _r.Random(0), band=0.05)
    check("a gain far above the band IS kept",
          max(m.count("1") for m, _ in wb) == 2,
          f"trace {[m for m, _ in wb]} -- the band must not reject a 40% win")

    section("replay: screening fidelity")
    clean = Table.synthetic({"a": 20.0, "b": 10.0, "c": 5.0, "d": -8.0,
                             "e": 0.5, "f": -0.2}, base=30.0, noise=0.0,
                            repeats=1, seed=3)
    fid = screen_fidelity(clean, seeds=10)
    check("a noiseless additive space is screened perfectly",
          fid["rank_correlation"] > 0.999 and fid["top3_recall"] > 0.999,
          f"{fid} -- resolution III is exact when there are no interactions")
    check("spearman is 1.0 for identical orderings",
          abs(_spearman([1, 2, 3], [10, 20, 30]) - 1.0) < 1e-9)
    check("spearman is -1.0 for reversed orderings",
          abs(_spearman([1, 2, 3], [30, 20, 10]) + 1.0) < 1e-9)


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
               test_trial_axes, test_frontier, test_pb_design, test_replay, test_dag_file,
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
