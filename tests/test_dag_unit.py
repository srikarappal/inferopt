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
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent.parent / 'src'))

from inferopt._paths import default_dag

_DAG = default_dag()
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
    from inferopt.predicates import Predicate, PredicateError

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
        from inferopt.fingerprint import Context
        check(f"refuses to execute: {e}",
              raises(lambda: p.evaluate(_ctx())),
              "parsed AND evaluated -- the predicate language executes arbitrary code")

    # THE SUPPORTED SURFACE, pinned. A DAG author needs to know what is legal,
    # and a silent change here turns a working predicate into a skipped node.
    from inferopt.fingerprint import Context
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
    from inferopt.predicates import resolve_path_type
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


def _mk(td, name):
    """An empty artifact directory under a temp dir."""
    d = Path(td) / name
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


def _ctx(**over):
    """A Context with a real fingerprint, cheap and offline."""
    from inferopt.fingerprint import (Context, Fingerprint, HardwareFingerprint,
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
    from inferopt.predicates import Predicate
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
    from inferopt.traverse import _value
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
    from inferopt.traverse import _variants
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
    from inferopt.traverse import Trial
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
    from inferopt.traverse import OBJECTIVES
    check("axes covers exactly the objectives", set(ax) == set(OBJECTIVES),
          f"axes={sorted(ax)} objectives={sorted(OBJECTIVES)}")


def test_frontier():
    section("Result.frontier: non-domination")
    from inferopt.traverse import Result

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
    from inferopt.traverse import OBJECTIVES
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
    from inferopt.pb_screen import pb_design, effects

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
    from inferopt.replay import (Table, regret, sequential_dag, pb_then_factorial,
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
def test_moe_backend_and_int_flags():
    """Two bugs that each cost real launches, and the guards against their class.

    Both were invisible in the same way: a launch that dies during startup is
    journalled as goodput 0.0, which is indistinguishable from a configuration
    that simply did not help.
    """
    section("moe backend: the expert layers, not the declared algorithm")
    import tempfile
    from inferopt.evaluator import (moe_expert_state, reconcile_moe_backend,
                           _moe_backends, to_cli)

    # vLLM's own lists. An empty set here is not a neutral result -- the caller
    # reads it as "no opinion" and skips the correction, which is precisely the
    # failure mode that let moe_backend=triton reach a quantized MoE.
    q, u = _moe_backends("nvfp4"), _moe_backends("unquantized")
    check("vLLM's NVFP4 backend list is readable", bool(q),
          "empty means the reconciliation silently does nothing")
    check("vLLM's unquantized backend list is readable", bool(u))
    check("marlin is accepted for a quantized NVFP4 MoE", "marlin" in q, f"{sorted(q)}")
    check("triton is accepted for an unquantized MoE", "triton" in u, f"{sorted(u)}")
    check("the two lists overlap, so a MIXED checkpoint has a legal backend",
          bool(q & u), "no backend serves both -- a mixed artifact is unservable")

    def fixture(d, quantization):
        (Path(d) / "hf_quant_config.json").write_text(
            json.dumps({"quantization": quantization}))
        return d

    with tempfile.TemporaryDirectory() as td:
        # Single-format: no quantized_layers map, the declared algo is the truth.
        a = _mk(td, "nvfp4"); fixture(a, {"quant_algo": "NVFP4"})
        check("a single-format NVFP4 artifact reads as fully quantized",
              moe_expert_state(a) == "full", moe_expert_state(a))
        b = _mk(td, "w4a16"); fixture(b, {"quant_algo": "W4A16_NVFP4"})
        check("W4A16_NVFP4 also reads as fully quantized",
              moe_expert_state(b) == "full", moe_expert_state(b))
        c = _mk(td, "plain"); fixture(c, {"quant_algo": "FP8"})
        check("a non-NVFP4 single-format artifact reads as unquantized experts",
              moe_expert_state(c) == "none", moe_expert_state(c))

        # MIXED_PRECISION: the map decides, and this is where the bug lived.
        full = _mk(td, "mixed_full")
        fixture(full, {"quant_algo": "MIXED_PRECISION", "quantized_layers": {
            f"model.layers.{i}.self_attn.o_proj": {"quant_algo": "NVFP4"}
            for i in range(4)} | {
            f"model.layers.{i}.mlp.experts": {"quant_algo": "NVFP4"}
            for i in range(4)}})
        check("MIXED_PRECISION with every expert layer quantized reads full",
              moe_expert_state(full) == "full", moe_expert_state(full))

        mixed = _mk(td, "mixed_partial")
        fixture(mixed, {"quant_algo": "MIXED_PRECISION", "quantized_layers": {
            f"model.layers.{i}.self_attn.o_proj": {"quant_algo": "NVFP4"}
            for i in range(4)} | {
            f"model.layers.{i}.mlp.experts": {"quant_algo": "NVFP4"}
            for i in range(3)}})           # layer 3's experts left in bf16
        check("MIXED_PRECISION with ONE expert layer left out reads mixed",
              moe_expert_state(mixed) == "mixed", moe_expert_state(mixed))

        none = _mk(td, "attn_only")
        fixture(none, {"quant_algo": "MIXED_PRECISION", "quantized_layers": {
            f"model.layers.{i}.self_attn.o_proj": {"quant_algo": "NVFP4"}
            for i in range(4)}})
        check("MIXED_PRECISION touching no experts reads as unquantized experts",
              moe_expert_state(none) == "none", moe_expert_state(none))

        gone = _mk(td, "nothing")
        check("an artifact with no quant config is unknown, not assumed",
              moe_expert_state(gone) == "unknown", moe_expert_state(gone))

        # The decision itself.
        pick = lambda path: reconcile_moe_backend(
            {"model": path, "moe_backend": "triton"}, log=lambda *_: None)["moe_backend"]
        check("a fully quantized MoE moves off triton",
              pick(full) != "triton" and pick(full) in q, pick(full))
        check("a fully quantized MoE prefers marlin", pick(full) == "marlin", pick(full))
        # THE REGRESSION. The first version switched this to marlin, and vLLM
        # rejected it with "not supported for unquantized MoE".
        check("a MIXED artifact gets a backend accepted by BOTH sides",
              pick(mixed) in (q & u), f"{pick(mixed)} is not in {sorted(q & u)}")
        check("a MIXED artifact is not sent to marlin", pick(mixed) != "marlin",
              "marlin refuses the unquantized expert layers")
        check("a MIXED artifact is not left on triton", pick(mixed) != "triton",
              "triton refuses the quantized expert layers")
        check("experts left unquantized keep the unquantized backend",
              pick(none) == "triton", pick(none))
        check("an unknown artifact is not touched", pick(gone) == "triton", pick(gone))
        # With the preferred names removed, the fallback must STILL choose from
        # the overlap. Picking from the NVFP4 list alone would return cutlass,
        # which vLLM refuses for the unquantized expert layers -- the same class
        # of failure as the original bug, one branch further down.
        import inferopt.evaluator as _ev
        real = _ev._moe_backends
        try:
            _ev._moe_backends = lambda kind: (
                {"cutlass", "marlin", "flashinfer_b12x", "humming"}
                if kind == "nvfp4" else {"triton", "aiter", "humming"})
            got = reconcile_moe_backend({"model": mixed, "moe_backend": "triton"},
                                        log=lambda *_: None)["moe_backend"]
            check("with no preferred backend available, mixed still picks from "
                  "the overlap", got == "humming",
                  f"got {got}; the overlap was {{'humming'}}")
        finally:
            _ev._moe_backends = real

        check("a backend already valid is left alone",
              reconcile_moe_backend({"model": full, "moe_backend": "marlin"},
                                    log=lambda *_: None)["moe_backend"] == "marlin")
        check("a config with no model is returned unchanged",
              reconcile_moe_backend({"moe_backend": "triton"},
                                    log=lambda *_: None)["moe_backend"] == "triton")

    section("integer flags: a float is a launch failure, not a rounding style")
    def flags(cfg):
        """Parse to_cli output. Boolean flags are LONE tokens, so naive
        zip(args[::2], args[1::2]) pairing silently misaligns everything after
        the first bool -- which is how this test first failed against correct
        code."""
        a, out, i = to_cli(cfg), {}, 0
        while i < len(a):
            if i + 1 < len(a) and not a[i + 1].startswith("--"):
                out[a[i]] = a[i + 1]; i += 2
            else:
                out[a[i]] = True; i += 1
        return out

    pair = flags({"max_num_seqs": 384.0, "max_num_batched_tokens": 512.0,
                  "max_model_len": 7168, "gpu_memory_utilization": 0.75,
                  "enforce_eager": False, "kv_cache_dtype": "fp8_e4m3"})
    args = to_cli({"enforce_eager": False})
    check("an integral float for an int flag is emitted without a decimal point",
          pair.get("--max-num-seqs") == "384", pair.get("--max-num-seqs"))
    check("...for every int flag, not just the one that broke",
          pair.get("--max-num-batched-tokens") == "512",
          pair.get("--max-num-batched-tokens"))
    check("a non-integral float for an int flag is ROUNDED, not truncated",
          flags({"max_num_seqs": 151.5})["--max-num-seqs"] == "152",
          "truncation systematically under-shoots a swept count; "
          "argparse rejects '151.5' and the server dies before loading weights")
    check("a genuinely fractional flag keeps its decimal point",
          pair.get("--gpu-memory-utilization") == "0.75",
          pair.get("--gpu-memory-utilization"))
    check("ints are untouched", pair.get("--max-model-len") == "7168")
    check("booleans still use the --no- form", "--no-enforce-eager" in args)
    check("strings are untouched", pair.get("--kv-cache-dtype") == "fp8_e4m3")

    section("the DAG's own arithmetic yields integers")
    d = json.loads(_DAG.read_text())
    from inferopt.predicates import Predicate
    ctx = _ctx(incumbent={"max_num_seqs": 256})
    bad = []
    for n in d["nodes"]:
        for sweep in (n.get("sweep") or []):
            for k, v in sweep.items():
                if k not in ("max_num_seqs", "max_num_batched_tokens",
                             "max_model_len", "block_size"):
                    continue
                if not isinstance(v, str):
                    continue
                try:
                    out = Predicate(v).evaluate(ctx)
                except Exception:
                    continue
                if isinstance(out, float):
                    bad.append((n["id"], k, v, out))
    check("no sweep computes a float for an integer-valued knob", not bad,
          f"{bad} -- vLLM's argparse rejects these and the launch dies")


# ==========================================================================
def test_qps_source():
    """qps is the denominator of every replica count, so where it comes from
    matters as much as its value."""
    section("qps: stated, derived, or refused")
    import tempfile
    from inferopt.fingerprint import WorkloadFingerprint

    rows = [{"prompt": "x", "input_tokens": 100, "output_tokens": 50,
             "arrival_ts": i * 0.5, "prefix_id": None, "temperature": 0.0}
            for i in range(20)]

    with tempfile.TemporaryDirectory() as td:
        withts = Path(td) / "with.jsonl"
        withts.write_text("".join(json.dumps(r) + "\n" for r in rows))
        nots = Path(td) / "without.jsonl"
        nots.write_text("".join(
            json.dumps({k: v for k, v in r.items() if k != "arrival_ts"}) + "\n"
            for r in rows))

        w = WorkloadFingerprint.from_trace(str(withts))
        # 20 requests, first at 0.0 and last at 9.5, so the span is 9.5s.
        check("qps is derived from the trace when timestamps are present",
              abs(w.request_rate_qps - 20 / 9.5) < 1e-6, f"{w.request_rate_qps}")

        w = WorkloadFingerprint.from_trace(str(withts), request_rate_qps=40.0)
        check("a stated rate overrides the trace's timestamps",
              w.request_rate_qps == 40.0, f"{w.request_rate_qps}")

        # THE REGRESSION GUARD. This used to return qps=0.0 in silence, which
        # made demand 0 tok/s and every replica count meaningless, on the one
        # field a caller is least likely to have.
        check("a trace with no arrival_ts and no stated rate RAISES",
              raises(lambda: WorkloadFingerprint.from_trace(str(nots))),
              "silently returning qps=0 is what this replaces")
        try:
            WorkloadFingerprint.from_trace(str(nots))
        except ValueError as e:
            check("...and the error names the way out", "--qps" in str(e),
                  "an error that does not say what to do is only half a fix")

        w = WorkloadFingerprint.from_trace(str(nots), request_rate_qps=16.0)
        check("a stated rate makes a timestamp-free trace usable",
              w.request_rate_qps == 16.0, f"{w.request_rate_qps}")
        check("the other statistics still come from the trace itself",
              w.n_requests == 20 and w.mean_output_tokens == 50.0,
              f"{w.n_requests} reqs, out {w.mean_output_tokens}")

    section("qps: the request surface")
    from inferopt.request import InferOptRequest
    tr = "data/trace_shared.jsonl"      # InferOptRequest validates the path
    if Path(tr).exists():
        check("qps must be positive",
              raises(lambda: InferOptRequest(model="m", trace=tr, qps=0)),
              "a zero rate is the bug this flag exists to prevent")
        check("a negative qps is refused",
              raises(lambda: InferOptRequest(model="m", trace=tr, qps=-1)))
        check("qps is optional",
              InferOptRequest(model="m", trace=tr).qps is None)


# ==========================================================================
def test_methods_comparable():
    """Three search methods must emit records that can be put in one table."""
    section("methods: yolo's cells")
    import tempfile
    from inferopt.pb_screen import pb_design

    # yolo measures exactly two DISTINCT configs however many launches it makes.
    # That is its defining property: cheapest possible, and unable to attribute
    # a result to any single factor.
    factors = [{"id": f"f{i}", "on": {f"flag{i}": True}} for i in range(6)]
    base = {"model": "m", "gpu_memory_utilization": 0.9}
    all_on = dict(base)
    for f in factors:
        all_on.update(f["on"])
    check("all-on turns on every factor",
          all(all_on.get(f"flag{i}") for i in range(6)), f"{all_on}")
    check("all-off is exactly the seed", base == {"model": "m",
                                                  "gpu_memory_utilization": 0.9})
    check("yolo considers 2 configs regardless of factor count",
          len({json.dumps(base, sort_keys=True),
               json.dumps(all_on, sort_keys=True)}) == 2)

    section("methods: PB stage 2 enumerates a full factorial")
    # k survivors must give 2^k DISTINCT configs, each differing only in the
    # varied factors -- the pinned background has to be identical across them
    # or the factorial measures a moving target.
    top = ["f0", "f2", "f4"]
    pinned = {"f1": True, "f3": False, "f5": True}
    by_id = {f["id"]: f for f in factors}
    cfgs = []
    for i in range(2 ** len(top)):
        bits = [(i >> j) & 1 for j in range(len(top))]
        c = dict(base)
        for fid, on in pinned.items():
            if on:
                c.update(by_id[fid]["on"])
        for fid, on in zip(top, bits):
            if on:
                c.update(by_id[fid]["on"])
        cfgs.append(c)
    check("2^k configs for k survivors", len(cfgs) == 8, f"{len(cfgs)}")
    check("all 2^k are distinct",
          len({json.dumps(c, sort_keys=True) for c in cfgs}) == 8)
    check("the pinned background is identical in every cell",
          all(c.get("flag1") and c.get("flag5") and "flag3" not in c for c in cfgs),
          "a moving background confounds the factorial it exists to resolve")
    check("one cell has none of the varied factors on",
          any(not any(c.get(f"flag{f[1]}") for f in top) for c in cfgs),
          "without it there is no anchor for the varied set")
    check("one cell has all of them on",
          any(all(c.get(f"flag{f[1]}") for f in top) for c in cfgs))

    section("methods: compare.py reads every shape")
    from inferopt.compare import load, num, acc
    check("num formats and falls back", num(1.234) == "1.2" and num(None) == "-")
    check("num survives a bad type", num("x") == "-", "must not raise mid-table")
    check("acc reads a benchmark out of a trial",
          acc({"quality": {"math_500": 0.5}}, "math_500") == 0.5)
    check("acc on a trial with no quality is None",
          acc({}, "math_500") is None, "absence must not read as zero")

    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "empty"; d.mkdir()
        check("a directory with no result.json loads as None", load(d) is None)

        # A traversal result.json -- the shape run.py writes -- must be mapped,
        # not skipped, or every historical run drops out of the comparison.
        d2 = Path(td) / "trav"; d2.mkdir()
        (d2 / "result.json").write_text(json.dumps({
            "baseline": {"node_id": "seed", "goodput": 10.0, "quality": {"math_500": 0.5}},
            "trials": [{"node_id": "a", "goodput": 20.0, "kept": True,
                        "quality": {"math_500": 0.6}},
                       {"node_id": "b", "goodput": 0.0, "kept": False}],
            "incumbent": {"config": {"x": 1}},
            "incumbent_peak": {"goodput": 25.0, "concurrency": 8},
            "launches": 3, "minutes": 30.0}))
        r = load(d2)
        check("a traversal is labelled seqDAG", r["method"] == "seqDAG", r["method"])
        check("its shipped goodput comes from the peak sweep",
              r["chosen"]["goodput"] == 25.0, f"{r['chosen']}")
        check("its shipped ACCURACY comes from the last kept trial",
              r["chosen"]["quality"]["math_500"] == 0.6,
              "incumbent_peak carries no quality, so it must be sourced")
        check("the seed is included as a trial",
              any("seed" in str(t.get("node_id")) for t in r["trials"]),
              "the baseline is the anchor every percentage is against")
        check("failed launches are counted, not dropped",
              r["failed_launches"] == 1, f"{r['failed_launches']}")
        check("best_seen is the best MEASURED trial",
              r["best_seen"]["goodput"] == 20.0, f"{r['best_seen']}")
        check("the shipped config is named after the node that produced it",
              r["chosen"]["node_id"] == "a",
              "'incumbent' matches no trial, so nothing is ever marked shipped")

    section("compare: the joint Pareto frontier")
    from inferopt.compare import pareto, collect, AXES

    # Domination, on points where the answer is not in doubt.
    pts = [{"goodput": 100.0, "quality": 0.9, "ttft_p99_ms": 100.0},   # best at all
           {"goodput": 50.0,  "quality": 0.5, "ttft_p99_ms": 200.0},   # dominated
           {"goodput": 10.0,  "quality": 0.95, "ttft_p99_ms": 400.0},  # best accuracy
           {"goodput": 90.0,  "quality": 0.5, "ttft_p99_ms": 50.0}]    # best ttft
    f = set(pareto(pts, AXES))
    check("a point beaten on every axis is dominated", 1 not in f, f"{f}")
    check("the all-round best survives", 0 in f)
    check("a point best on ONE axis survives", 2 in f and 3 in f,
          "a frontier that keeps only the goodput winner is not a frontier")

    # A point missing an axis must be dropped, not defaulted -- filling an
    # unmeasured accuracy with 0 dominates nothing and is dominated by all.
    holed = pts + [{"goodput": 999.0, "quality": None, "ttft_p99_ms": 1.0}]
    check("a point missing an axis is excluded, not defaulted",
          4 not in set(pareto(holed, AXES)),
          "an unmeasured accuracy must not enter the frontier as 0 or as free")
    check("excluding it does not disturb the rest",
          set(pareto(holed, AXES)) == f)
    check("an empty input yields an empty frontier", pareto([], AXES) == [])
    check("a single point is its own frontier",
          pareto([pts[1]], AXES) == [0])

    # A node_id covering several variants must star exactly one point.
    runs2 = [{"method": "m", "chosen": {"node_id": "q"},
              "trials": [{"node_id": "q", "goodput": 10.0,
                          "config": {"quantize": "nvfp4"}},
                         {"node_id": "q", "goodput": 30.0,
                          "config": {"quantize": "w4a16"}},
                         {"node_id": "q", "goodput": 20.0, "config": {}}]}]
    got = collect(runs2, "math_500")
    check("exactly one point is marked shipped",
          sum(1 for p in got if p["shipped"]) == 1,
          f"{sum(1 for p in got if p['shipped'])} -- one node_id covers four "
          f"quantization variants and starring all of them is wrong")
    check("the shipped point is the best variant of that node",
          next(p["goodput"] for p in got if p["shipped"]) == 30.0)
    check("variant labels distinguish rows sharing a node_id",
          len({p["label"] for p in got}) == 3, f"{[p['label'] for p in got]}")
    check("a dead launch is not a point in the space",
          all(p["goodput"] for p in got))


# ==========================================================================
def test_doe_analysis():
    """The analysis layered on the screen: aliases, Lenth, every response."""
    section("aliases: what a main effect is confounded with")
    from inferopt.pb_screen import pb_design, aliases, lenth, response_effects, effects

    names = [f"f{i}" for i in range(6)]
    fs = [{"id": n} for n in names]
    design, N = pb_design(len(names))
    al = aliases(design, fs)

    check("every factor gets an alias list", set(al) == set(names))
    check("a factor is never aliased with itself",
          all(f not in h["with"].split("*") for f in names for h in al[f]),
          "an effect confounded with its own interaction is a bug")
    rs = [abs(h["r"]) for f in names for h in al[f]]
    check("aliasing is PARTIAL, not total", rs and max(rs) < 1.0,
          f"max |r| {max(rs) if rs else 0}; r=1.0 would be a regular fraction, "
          f"not Plackett-Burman")
    check("aliasing is non-zero", rs and min(rs) > 0.0,
          "no confounding at all would mean this is not resolution III")
    check("correlations are bounded by 1", all(r <= 1.0 + 1e-9 for r in rs))

    section("Lenth's PSE")
    # One huge effect among inert ones: PSE must come from the inert ones, so
    # the big one lands far outside the margin.
    L = lenth([40.0, 0.5, -0.4, 0.6, -0.3, 0.2])
    check("PSE is estimated from the small effects, not the large one",
          L["pse"] < 2.0, f"pse {L['pse']} -- a 40.0 effect has contaminated it")
    check("the large effect exceeds the margin", 40.0 > L["me"], f"me {L['me']}")
    check("an inert effect does not", 0.6 < L["me"])
    # All effects equal: nothing should be called active.
    L2 = lenth([5.0, -5.0, 5.0, -5.0, 5.0, -5.0])
    check("when every effect is the same size, none stands out",
          5.0 <= L2["me"], f"me {L2['me']} -- would call all six active")
    check("too few effects is refused, not guessed",
          lenth([1.0, 2.0]).get("pse") is None)
    check("PSE scales with the data",
          abs(lenth([80.0, 1.0, -0.8, 1.2, -0.6, 0.4])["pse"] - 2 * L["pse"]) < 1e-9,
          "doubling every effect must double the noise estimate")

    section("effects on every response")
    # A factor that leaves goodput alone and wrecks TTFT must be visible.
    gp = [100.0 for _ in design]
    ttft = [300.0 + (250.0 if row[0] else 0.0) for row in design]
    allf = response_effects(design, fs, {"goodput": gp, "ttft_p99_ms": ttft})
    check("both responses are analysed", set(allf) == {"goodput", "ttft_p99_ms"})
    g0 = next(x["effect"] for x in allf["goodput"] if x["id"] == "f0")
    t0 = next(x["effect"] for x in allf["ttft_p99_ms"] if x["id"] == "f0")
    check("a factor flat on goodput reads ~0 there", abs(g0) < 1e-9, f"{g0}")
    check("...and is caught on TTFT", abs(t0 - 250.0) < 1e-9,
          f"{t0}; a goodput-only screen would call this factor harmless")
    check("an unrelated factor moves neither response",
          abs(next(x["effect"] for x in allf["ttft_p99_ms"]
                   if x["id"] == "f3")) < 1e-9)


# ==========================================================================
def test_seed_from_run():
    """Continuing from a previous run's answer, rather than from the seed."""
    section("seed-from-run: the config a stage starts at")
    import tempfile
    from inferopt.evaluator import hardware_defaults

    prev = {
        "incumbent": {"config": {"enable_prefix_caching": True,
                                 "max_model_len": 7168, "max_num_seqs": 256}},
        "incumbent_peak": {"goodput": 118.7},
        "trials": [{"node_id": "prefix_caching", "kept": True, "goodput": 100.0},
                   {"node_id": "chunked_prefill", "kept": False, "goodput": 90.0}],
    }
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "prev"; d.mkdir()
        (d / "result.json").write_text(json.dumps(prev))

        # The loader is the piece under test: read the incumbent, let
        # hardware_defaults fill only what the incumbent does not state.
        r = json.loads((d / "result.json").read_text())
        inc = (r.get("incumbent") or {})
        inc = inc.get("config") or inc
        fp = _ctx().fingerprint
        cfg = {**inc, **{k: v for k, v in hardware_defaults(fp).items() if k not in inc}}

        check("the previous incumbent's settings survive",
              cfg["enable_prefix_caching"] is True and cfg["max_model_len"] == 7168,
              f"{cfg}")
        check("the incumbent WINS over a hardware default it also sets",
              cfg["max_num_seqs"] == 256,
              "a measured config must not be overwritten by a default")
        for k, v in hardware_defaults(fp).items():
            if k not in inc:
                check(f"hardware default {k} is still applied", cfg.get(k) == v,
                      "rails the previous config predates must still land")
                break

        # A run that kept nothing still has an incumbent -- the seed -- and must
        # not be mistaken for an empty file.
        (d / "result.json").write_text(json.dumps(
            {"incumbent": {"config": {"max_model_len": 4096}}, "trials": []}))
        r2 = json.loads((d / "result.json").read_text())
        i2 = (r2.get("incumbent") or {}); i2 = i2.get("config") or i2
        check("a run that kept nothing still yields its config", i2 == {"max_model_len": 4096})

        # An older result.json stores the incumbent flat rather than under
        # "config"; both shapes are on disk in runs/ today.
        r3 = {"incumbent": {"max_model_len": 2048}}
        i3 = (r3.get("incumbent") or {}); i3 = i3.get("config") or i3
        check("a flat incumbent (older result.json) is read too",
              i3 == {"max_model_len": 2048},
              "runs/ holds both shapes; only one of them is nested")

        empty = Path(td) / "empty"; empty.mkdir()
        check("a directory with no result.json is detectable",
              not (empty / "result.json").exists())


# ==========================================================================
def test_api_types():
    """The user-facing quality layer: judge, metric, and a reported change."""
    section("Metric: direction is mandatory, never inferred")
    from inferopt.api_types import Sample, Verdict, Metric, QualityChange

    check("a built-in gets its direction", Metric("pass@1").direction == "max")
    check("a lower-is-better built-in is not assumed to be max",
          Metric("wer").direction == "min",
          "wer and pass@1 move opposite ways and no naming rule separates them")
    check("higher_is_better follows direction",
          Metric("pass@1").higher_is_better and not Metric("wer").higher_is_better)

    # THE TYPO CASE. A hyphen where the built-in has an underscore would
    # otherwise become a silent custom metric with no direction and no fn,
    # scoring 0.0 for every config -- which reads as "quality unchanged".
    check("an unknown name RAISES rather than defaulting",
          raises(lambda: Metric("exact-match")),
          "a silent custom metric scores 0.0 everywhere and looks like no change")
    check("the error names the built-ins",
          "exact_match" in str(_err(lambda: Metric("exact-match"))))
    check("a custom metric is fine WITH a direction and fn",
          Metric("mine", direction="max", fn=lambda s, v: 1.0).compute([], []) == 1.0)
    check("a custom metric with a direction but no fn raises",
          raises(lambda: Metric("mine", direction="max")),
          "nothing to compute")

    section("Verdict carries a reason and still acts like a bool")
    vs = [Verdict(True), Verdict(False, reason="JSON did not parse"), Verdict(True)]
    check("bool() works, so existing sum()/mean code is unaffected",
          bool(vs[0]) and not bool(vs[1]))
    check("pass@1 aggregates verdicts", abs(Metric("pass@1").compute([], vs) - 2/3) < 1e-9)
    check("error_rate is the complement",
          abs(Metric("error_rate").compute([], vs) - 1/3) < 1e-9)
    check("the reason survives to the caller",
          vs[1].reason == "JSON did not parse",
          "a False with no reason is what made RULER's 0.05 undiagnosable")
    check("aggregating no verdicts raises rather than returning 0.0",
          raises(lambda: Metric("pass@1").compute([], [])),
          "0.0 from an empty set is indistinguishable from total failure")

    section("QualityChange: noise-aware and direction-aware")
    # Real numbers from this project's own runs.
    lossless = QualityChange("math_500", "exact_match", 0.7333, 0.7300, 0.04)
    check("a delta inside the benchmark's resolution is not a finding",
          lossless.within_noise and not lossless.is_regression,
          "the lossless step moved 0.0033 against a 0.04 resolution")
    real = QualityChange("math_500", "exact_match", 0.6633, 0.6380, 0.006)
    check("a delta outside it IS a regression",
          real.is_regression and not real.within_noise,
          "nvfp4 at n=500 lost 0.0253 against a 0.006 spread")
    up = QualityChange("math_500", "exact_match", 0.70, 0.76, 0.04)
    check("an improvement is not reported as a regression",
          not up.is_regression and abs(up.delta - 0.06) < 1e-9)
    lower = QualityChange("asr", "wer", 0.10, 0.14, 0.01)
    check("for a lower-is-better metric, a RISE is the regression",
          lower.is_regression,
          "direction must come from the metric, not from the sign")
    check("...and a fall is not",
          not QualityChange("asr", "wer", 0.14, 0.10, 0.01).is_regression)
    check("str() states the verdict in words", "REGRESSION" in str(real))

    section("Sample keeps the row intact")
    sm = Sample(row={"answer": "42", "subject": "algebra"}, text="\\boxed{42}", index=3)
    check("the dataset row is passed through untouched",
          sm.row["subject"] == "algebra",
          "judges need fields that differ per benchmark; normalising loses them")
    check("the index is kept so verdicts can be zipped back to rows", sm.index == 3)


def _err(fn):
    try:
        fn()
    except Exception as e:
        return e
    return None


# ==========================================================================
def test_strategies():
    """Three search strategies behind one protocol, with a declared asymmetry."""
    section("strategies: the shared protocol")
    from inferopt.strategies import (STRATEGIES, ScreenStrategy, SearchResult,
                                     SequentialStrategy, Strategy, YoloStrategy)

    check("all three are registered",
          set(STRATEGIES) == {"sequential", "yolo", "screen"}, f"{set(STRATEGIES)}")
    for n, cls in STRATEGIES.items():
        check(f"{n}: declares a name matching its key", cls.name == n)
        check(f"{n}: declares whether it chains an incumbent",
              isinstance(cls.chains_incumbent, bool))

    # THE ASYMMETRY IS THE POINT. PB's arithmetic requires every row to share a
    # background -- the difference of means only isolates a factor if it does --
    # so chaining it would destroy the property the method exists for. The
    # protocol declares this rather than forcing one behaviour.
    check("only the sequential walk chains",
          SequentialStrategy.chains_incumbent
          and not YoloStrategy.chains_incumbent
          and not ScreenStrategy.chains_incumbent,
          "chaining PB would break the balance its estimates rest on")

    section("strategies: yolo measures two configs, whatever the budget")
    factors = [{"id": f"f{i}", "on": {f"flag{i}": True}} for i in range(6)]
    seed = {"model": "m", "gpu_memory_utilization": 0.9}
    ev = _Runner({"all_off-rep1": 100.0, "all_off-rep2": 110.0,
                  "all_on-rep1": 300.0, "all_on-rep2": 290.0})
    out = YoloStrategy(factors, repeats=2).search(_ctx(), ev, seed, log=lambda *_: None)
    check("four launches for two cells at two repeats", out.launches == 4, out.launches)
    seen = {json.dumps({k: v for k, v in t.config.items() if k != "model"},
                       sort_keys=True) for t in out.trials}
    check("only TWO distinct configs are ever measured", len(seen) == 2,
          f"{len(seen)} -- yolo cannot attribute a result to any single factor")
    check("it ships the better cell", out.chosen.goodput in (300.0, 290.0),
          f"{out.chosen.goodput}")
    check("the cell's value is the MEAN, not the best launch",
          abs(out.extra["cells"]["all_on"]["mean_goodput"] - 295.0) < 1e-9,
          "taking the max would keep whichever launch drew luckiest")
    check("the lift is reported against all-off",
          abs(out.extra["lift_all_on_vs_all_off"] - (295.0 / 105.0 - 1)) < 1e-9)

    section("strategies: SearchResult counts dead launches")
    r = SearchResult(method="x", trials=[_T(10.0), _T(0.0), _T(5.0)])
    check("a launch with no goodput counts as failed", r.failed_launches == 1,
          f"{r.failed_launches} -- a dead launch cost the same as a live one")


class _T:
    def __init__(self, gp):
        self.goodput = gp


class _Runner:
    """The narrow Measurer a strategy sees: measure(config, label) -> Trial.

    Not ScriptedEvaluator, which implements the RAW evaluator signature with
    probes, benchmarks and levels. The distinction is the point: a strategy
    does not choose those, because letting each one choose is how "goodput"
    came to mean three different things across the three implementations.
    """

    def __init__(self, script: dict, default: float = 10.0):
        self.script, self.default = script, default
        self.calls: list[tuple[str, dict]] = []

    def measure(self, config: dict, label: str):
        from inferopt.traverse import Trial
        self.calls.append((label, dict(config)))
        gp = self.script.get(label, self.default)
        return Trial(node_id=label, config=dict(config), goodput=gp,
                     ttft_p99_ms=100.0, itl_p99_ms=10.0, memory_gb=1.0,
                     slo_ok=bool(gp), concurrency=8)


# ==========================================================================
def test_result_api():
    """Result: what a caller reads, and what it refuses to guess."""
    section("Result: reporting")
    from inferopt.api import Result
    from inferopt.api_types import QualityChange

    class _R:
        def __init__(self, gp, L=8, q=None, node="n", inherited=False):
            self.goodput, self.concurrency = gp, L
            self.quality, self.node_id = q or {}, node
            self.quality_inherited = inherited

    r = Result(model="m", strategy="sequential",
               chosen=_R(100.0), best_seen=_R(120.0),
               baselines={"stock": _R(11.9, 30)},
               provenance={"workload": {"mean_output_tokens": 259.6}})

    check("replicas divides demand by the shipped goodput",
          r.replicas(16) == 42, f"{r.replicas(16)}  (ceil(16*259.6/100))")
    check("replicas can ask the same of a baseline",
          r.replicas(16, config=r.baselines["stock"]) == 350,
          "the report's stock figure is 350 replicas")
    check("replicas returns None without a workload, rather than guessing",
          Result(model="m", strategy="s", chosen=_R(100.0)).replicas(16) is None,
          "a fabricated demand would silently scale every capacity claim")
    check("replicas returns None when nothing was shipped",
          Result(model="m", strategy="s",
                 provenance={"workload": {"mean_output_tokens": 100}}).replicas(16) is None)

    check("a gap between shipped and best-seen is surfaced",
          "above what it ships" in r.summary(),
          "a strategy that walks past something better is worth reporting")

    section("Result: regressions are reported, never acted on")
    r.quality_changes = [
        QualityChange("math_500", "exact_match", 0.7333, 0.7300, 0.04),   # noise
        QualityChange("math_500", "exact_match", 0.6633, 0.6380, 0.006),  # real
        QualityChange("math_500", "exact_match", 0.70, 0.76, 0.006),      # better
    ]
    check("only the real, wrong-direction movement is a regression",
          len(r.regressions) == 1
          and abs(r.regressions[0].delta + 0.0253) < 1e-9,
          f"{[str(c) for c in r.regressions]}")
    check("the noisy one is not reported as a regression",
          all(not c.within_noise for c in r.regressions))
    check("an improvement is not a regression",
          all(c.delta < 0 for c in r.regressions))
    check("regressions do not remove the config from the frontier",
          r.chosen is not None,
          "quality is an axis; dropping the point removes the choice")


# ==========================================================================
def test_judges():
    """Judges return one Verdict per sample, and never confuse their own
    failure with the model's."""
    section("RuleJudge")
    from inferopt.api_types import Sample, Verdict
    from inferopt.judges import LLMJudge, RuleJudge

    j = RuleJudge(lambda s: "42" in s.text)
    vs = j([Sample(row={}, text="it is 42"), Sample(row={}, text="no")])
    check("one verdict per sample, in order", [v.ok for v in vs] == [True, False])
    check("a bool is wrapped into a Verdict", isinstance(vs[0], Verdict))

    boom = RuleJudge(lambda s: 1 / 0)
    got = boom([Sample(row={}, text="a"), Sample(row={}, text="b")])
    check("a judge that raises loses ONE sample, not the run",
          len(got) == 2 and not got[0].ok,
          "one bad row must not discard the other 499")
    check("...and the exception is kept as the reason",
          "ZeroDivisionError" in got[0].reason, got[0].reason)

    section("LLMJudge: parsing a verdict")
    p = LLMJudge._parse
    check("JSON verdict is read", p('{"pass": true, "reason": "ok"}').ok)
    check("...with its reason", p('{"pass": false, "reason": "rude"}').reason == "rude")
    check("a bare affirmative is accepted, and flagged as such",
          p("yes").ok and "not JSON" in p("yes").reason)
    # THE IMPORTANT ONE. Prose is not a False vote -- it is a failure to vote,
    # and scoring it as False reports the model under test got worse when the
    # JUDGE did. That is the shape of the RULER failure.
    check("unparseable prose RAISES rather than voting False",
          raises(lambda: p("Well, it depends on context.")),
          "a judge that failed to answer has not answered 'no'")

    section("LLMJudge: an outage is not a regression")
    import os
    os.environ.pop("INFEROPT_JUDGE_URL", None)
    check("no endpoint refuses to score rather than returning zeros",
          raises(lambda: LLMJudge(model="m", rubric="r")([Sample(row={}, text="x")])),
          "silently scoring 0.0 is indistinguishable from total model failure")
    check("the error says the judge must not be the model under test",
          "under test" in str(_err(
              lambda: LLMJudge(model="m", rubric="r")([Sample(row={}, text="x")]))))

    section("LLMJudge: the cap is reported, never silent")
    j = LLMJudge(model="m", rubric="r", max_samples=2)
    check("nothing skipped reads as complete", "all samples judged" in j.report())
    j.skipped = 5
    check("a truncated benchmark says so",
          "NOT judged" in j.report() and "subset" in j.report(),
          "a truncated score that looks complete makes the quality axis decorative")
    check("temperature is pinned to 0", j.temperature == 0.0,
          "a judge with its own sampling noise adds resolution the caller cannot see")


# ==========================================================================
def test_benchmark_surface():
    """Benchmark carries a Metric with a direction, a role, and opt-in gates."""
    section("Benchmark: metric direction is resolved, not assumed")
    from inferopt.quality import BENCHMARKS, Benchmark

    for n, b in BENCHMARKS.items():
        check(f"{n}: metric resolves to a Metric with a direction",
              b.metric_spec.direction in ("max", "min"), f"{b.metric}")
        check(f"{n}: higher_is_better agrees with the metric",
              b.higher_is_better == (b.metric_spec.direction == "max"))
        check(f"{n}: defaults to an AXIS, not a gate", b.role == "axis")
        check(f"{n}: no gate is set by default",
              b.require is None and b.abandon_below is None,
              "a default that stops the search stops exploration")

    section("Benchmark.builtin: overrides a built-in without redefining it")
    c = Benchmark.builtin("math_500", n_full=100, role="report")
    check("the override applies", c.n_full == 100 and c.role == "report")
    check("the original is untouched", BENCHMARKS["math_500"].n_full == 500,
          "dataclasses.replace must not mutate the registry")
    check("everything else is carried over",
          c.judge is BENCHMARKS["math_500"].judge and c.max_tokens == 1024)
    check("an unknown NAME raises, naming what exists",
          raises(lambda: Benchmark.builtin("exact-match")))
    # A typo'd FIELD would otherwise be swallowed and produce a benchmark that
    # silently ignores the caller's intent.
    check("an unknown FIELD raises rather than being ignored",
          raises(lambda: Benchmark.builtin("math_500", n_ful=100)),
          "a swallowed typo yields a benchmark that ignores what was asked")

    section("Benchmark: gates are opt-in and do not remove points")
    g = Benchmark.builtin("math_500", require=0.70, abandon_below=0.20)
    check("require and abandon_below are set when asked",
          g.require == 0.70 and g.abandon_below == 0.20)
    check("...and still default off elsewhere",
          BENCHMARKS["math_500"].require is None)


def test_slo_attainment():
    """The attainment floor: expressible, off by default, never deletes a point."""
    section("SLO: attainment eligibility")
    from inferopt.fingerprint import SLO

    loose = SLO(ttft_p99_ms=500, itl_p99_ms=250)
    check("with no floor, ANY attainment may ship",
          loose.attainment_ok(0.43) and loose.attainment_ok(0.0),
          "this is the historical behaviour and must not change silently")
    check("...including unmeasured", loose.attainment_ok(None))

    strict = SLO(ttft_p99_ms=500, itl_p99_ms=250, min_slo_attainment=0.95)
    check("at or above the floor ships",
          strict.attainment_ok(0.95) and strict.attainment_ok(0.99))
    # The real case: Qwen3-14B shipped at 76% attainment, TTFT p99 935ms against
    # a 500ms target, because goodput counts only conforming requests.
    check("below the floor does not ship", not strict.attainment_ok(0.76))
    check("UNMEASURED is not the same as passing",
          not strict.attainment_ok(None),
          "an absent measurement must never satisfy a floor")

    check("the floor is bounded to a fraction",
          raises(lambda: SLO(min_slo_attainment=1.5))
          and raises(lambda: SLO(min_slo_attainment=-0.1)))

    section("Result: eligibility is reported, not enforced by deletion")
    from inferopt.api import Result

    class _T:
        def __init__(self, gp, att):
            self.goodput, self.concurrency = gp, 32
            self.diagnostics = {"slo_attainment": att}
            self.quality, self.node_id, self.quality_inherited = {}, "n", False

    r = Result(model="m", strategy="s", chosen=_T(118.7, 0.76), slo=strict,
               frontier=[_T(118.7, 0.76), _T(15.8, 1.0)])
    check("a config below the floor is NOT eligible to ship",
          r.ships_within_slo() is False)
    check("...but is still on the frontier",
          len(r.frontier) == 2,
          "a point that misses is still a measurement someone may want")
    check("the summary warns rather than hiding it",
          "MISSES the attainment floor" in r.summary(), r.summary())

    r2 = Result(model="m", strategy="s", chosen=_T(118.7, 0.76), slo=loose)
    check("with no floor, eligibility is None -- not True",
          r2.ships_within_slo() is None,
          "unset must be distinguishable from passing")
    check("...and a low attainment is still surfaced as a note",
          "76% of requests" in r2.summary(), r2.summary())


# ==========================================================================
def test_run_benchmark_guards():
    """run_benchmark's own guards. Untested until a mutation pass showed the
    earlier 'coverage' was collateral from neighbouring mutations."""
    section("run_benchmark: a broken judge must not look like a broken model")
    import dataclasses

    from inferopt import quality as Q
    from inferopt.api_types import Verdict

    class _Out:
        def __init__(self, t): self.text = t

    def gen(prompts, max_tokens):
        return [_Out("\\boxed{42}") for _ in prompts]

    real = Q.BENCHMARKS["math_500"]

    def with_judge(judge, metric=None):
        b = dataclasses.replace(real, judge=judge,
                                **({"metric": metric} if metric else {}))
        return {**Q.BENCHMARKS, "math_500": b}

    saved = Q.BENCHMARKS
    saved_n = Q.TRAVERSAL_N
    try:
        Q.TRAVERSAL_N = 5

        # THE CENTRAL PROPERTY. A judge that returns nothing has not scored 0.0;
        # it has failed to score, and 0.0 is indistinguishable from a model that
        # got every answer wrong. That confusion is how a broken probe reads as
        # a total collapse.
        Q.BENCHMARKS = with_judge(lambda rows, texts: [])
        check("a judge returning NO verdicts raises rather than scoring 0.0",
              raises(lambda: Q.run_benchmark("math_500", gen)),
              "0.0 from an empty judge looks exactly like total model failure")
        # Metric.compute also refuses an empty list, so the check above passes
        # either way. Pin the MESSAGE: run_benchmark's names the benchmark and
        # the row count, which is what makes the failure actionable.
        msg = str(_err(lambda: Q.run_benchmark("math_500", gen)))
        check("...and the error names the benchmark and the row count",
              "math_500" in msg and "5 rows" in msg, msg[:110])

        # Verdicts are zipped against rows by position, so a short list silently
        # misattributes every verdict after the gap.
        Q.BENCHMARKS = with_judge(lambda rows, texts: [Verdict(True)] * (len(rows) - 1))
        check("a SHORT verdict list raises rather than misaligning",
              raises(lambda: Q.run_benchmark("math_500", gen)),
              "verdicts are matched to rows by position")
        Q.BENCHMARKS = with_judge(lambda rows, texts: [Verdict(True)] * (len(rows) + 3))
        check("a LONG verdict list raises too",
              raises(lambda: Q.run_benchmark("math_500", gen)))

        section("run_benchmark: aggregation goes through the Metric")
        # 3 of 5 pass. pass@1 is 0.6; error_rate over the SAME verdicts is 0.4.
        # Averaging booleans would give 0.6 for both, which is the bug.
        vs = lambda rows, texts: [Verdict(i < 3) for i in range(len(rows))]
        Q.BENCHMARKS = with_judge(vs)
        got = Q.run_benchmark("math_500", gen)
        check("a max-direction metric aggregates as a pass rate",
              abs(got - 0.6) < 1e-9, f"{got}")

        Q.BENCHMARKS = with_judge(vs, metric="error_rate")
        got = Q.run_benchmark("math_500", gen)
        check("a metric with a different rule is NOT averaged as booleans",
              abs(got - 0.4) < 1e-9,
              f"{got} -- error_rate is the complement, not the mean of verdicts")

        section("run_benchmark: bool judges still work")
        Q.BENCHMARKS = with_judge(lambda rows, texts: [True, False, True, True, False])
        got = Q.run_benchmark("math_500", gen)
        check("a judge returning plain bools is unaffected",
              abs(got - 0.6) < 1e-9,
              f"{got} -- Verdict.__bool__ keeps every existing judge working")

        section("run_benchmark: the record keeps the reason")
        import json
        import tempfile
        Q.BENCHMARKS = with_judge(
            lambda rows, texts: [Verdict(False, reason="did not parse")] * len(rows))
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "g.jsonl"
            Q.run_benchmark("math_500", gen, record=f)
            rows = [json.loads(l) for l in open(f)]
            check("every generation is recorded", len(rows) == 5, f"{len(rows)}")
            check("...with the judge's reason, not just the verdict",
                  rows[0]["reason"] == "did not parse", rows[0])
            check("...and the prompt and output", rows[0]["output"] and rows[0]["prompt"])
    finally:
        Q.BENCHMARKS = saved
        Q.TRAVERSAL_N = saved_n


# ==========================================================================
def test_legality():
    """Combinations vLLM refuses, caught before a launch is spent on them."""
    section("legality: the combination that cost 11 launches")
    from inferopt.legality import illegal, repair

    # Individually every flag here is accepted. Together vLLM raises at
    # SchedulerConfig validation, and an automated generator produces this
    # pairing constantly: it is in 3 of 12 PB design rows by construction.
    bad = {"max_num_batched_tokens": 2048, "max_model_len": 6144,
           "enable_chunked_prefill": False}
    why = illegal(bad)
    check("the pair is detected", len(why) == 1, f"{why}")
    check("...and the reason names both values",
          "2048" in why[0] and "6144" in why[0], why[0])

    ok = {**bad, "enable_chunked_prefill": True}
    check("chunked prefill makes it legal", not illegal(ok),
          "a long prompt is split across batches, so it need not fit one")
    check("equal is legal, not just greater",
          not illegal({**bad, "max_num_batched_tokens": 6144}))
    check("a config missing either field is not judged",
          not illegal({"enable_chunked_prefill": False}),
          "absent is not the same as violating")

    section("legality: repair is minimal and always reported")
    fixed, notes = repair(bad, log=lambda *_: None)
    check("the repaired config is legal", not illegal(fixed), f"{fixed}")
    check("it raises the token budget", fixed["max_num_batched_tokens"] == 6144)
    # THE IMPORTANT CHOICE. Switching chunked prefill on would also make it
    # legal and would silently add a SECOND technique to the configuration --
    # which in a screening design means the row stops measuring what the design
    # says it measures.
    check("it does NOT switch chunked prefill on",
          fixed["enable_chunked_prefill"] is False,
          "that would add a second technique and void the row")
    check("the change is reported, never silent", notes and "6144" in notes[0], notes)
    check("a legal config is returned unchanged with no notes",
          repair(ok, log=lambda *_: None) == (ok, []))

    section("legality: the pinning path, which poisoned all 8 stage-2 cells")
    # Stage 2 pins non-survivors by the sign of a confounded estimate, so one
    # bad pin lands in EVERY cell of the factorial.
    pinned = {"max_num_batched_tokens": 2048, "max_model_len": 6144,
              "enable_chunked_prefill": False, "enable_prefix_caching": False}
    cells = []
    for bits in range(8):
        c = dict(pinned)
        for i, k in enumerate(("enable_prefix_caching", "enforce_eager", "x")):
            if bits >> i & 1:
                c[k] = True
        cells.append(repair(c, log=lambda *_: None)[0])
    check("every factorial cell is legal after repair",
          not any(illegal(c) for c in cells),
          "one bad pin previously killed all eight launches")
    check("...and the varied factors are untouched by the repair",
          sum(1 for c in cells if c.get("enable_prefix_caching")) == 4,
          "repair must fix the background, never the thing under test")

    section("legality: TP divisibility")
    check("a TP that does not divide the heads is flagged",
          illegal({"tensor_parallel_size": 3, "_n_attention_heads": 40}))
    check("...and one that does is not",
          not illegal({"tensor_parallel_size": 8, "_n_attention_heads": 40}))
    check("TP=1 is never flagged",
          not illegal({"tensor_parallel_size": 1, "_n_attention_heads": 40}))


# ==========================================================================
def test_percentile_stability():
    """p95 beside p99, and the sample count both are drawn from."""
    section("summarize: a p99 over a handful is a maximum, not a percentile")
    from inferopt.evaluator import Req, summarize
    from inferopt.fingerprint import SLO

    def measure(n_slow, n=400):
        reqs = []
        for i in range(n):
            r = Req()
            r.start, r.ok, r.n_out = i * 0.05, True, 10
            r.ttft = 0.05 + (2.0 if i >= n - n_slow else 0.0)
            r.latency = r.ttft + 0.5
            r.token_times = [r.start + r.ttft + j * 0.05 for j in range(10)]
            reqs.append(r)
        return summarize(reqs, 0.0, 30.0, SLO(ttft_p99_ms=500, itl_p99_ms=250))

    clean = measure(0)
    check("the sample count is reported", clean["ttft_n"] == 400, clean["ttft_n"])
    check("with no outliers p95 and p99 agree",
          abs(clean["ttft_p99_ms"] - clean["ttft_p95_ms"]) < 1e-6)

    # THE MEASURED INSTABILITY. Across three identical launches of one config,
    # TTFT p99 varied 6.63x at L=64 (149/160/988ms) while goodput varied 1.06x.
    # A 45s window at L=128 completes ~384 requests, so p99 is the slowest ~4.
    few = measure(10)
    check("TEN slow requests in 400 move p99",
          few["ttft_p99_ms"] > 1000, f"{few['ttft_p99_ms']:.0f}ms")
    check("...and do NOT move p95",
          few["ttft_p95_ms"] < 100,
          f"{few['ttft_p95_ms']:.0f}ms -- p95 needs ~20 of 400, p99 needs ~4")
    many = measure(25)
    check("twenty-five DO move p95", many["ttft_p95_ms"] > 1000)

    section("summarize: the search is unaffected")
    # goodput counts SLO-conforming requests across ALL completions, which is
    # why it is stable at 1.0-1.06x where p99 swings 6.6x -- and goodput is what
    # keep/revert runs on.
    check("goodput barely moves on the same outliers",
          abs(measure(10)["goodput"] / clean["goodput"] - 1) < 0.05,
          "if goodput moved with p99, every keep/revert decision would be noise")
    check("itl gets the same treatment",
          "itl_p95_ms" in clean and "itl_p99_ms" in clean)

    section("the percentiles survive aggregate() and the diagnostics whitelist")
    import inspect
    from inferopt.evaluator import LOWER_IS_BETTER, VllmEvaluator, aggregate
    for k in ("ttft_p95_ms", "itl_p95_ms"):
        check(f"{k} aggregates to the WORST pass", k in LOWER_IS_BETTER)
    check("ttft_n aggregates to the FEWEST samples", "ttft_n" not in LOWER_IS_BETTER,
          "more samples is the stronger claim, so the min is the honest one")
    agg = aggregate([{"ttft_p95_ms": 10.0, "ttft_n": 400, "goodput": 100.0},
                     {"ttft_p95_ms": 90.0, "ttft_n": 300, "goodput": 200.0}])
    check("...and it does", agg["ttft_p95_ms"] == 90.0 and agg["ttft_n"] == 300, agg)
    # Turn 4 recorded p95 nan / n=0 on all three launches: summarize() emitted
    # them and the Trial's diagnostics dict is a whitelist that did not.
    src = inspect.getsource(VllmEvaluator.measure)
    for k in ("ttft_p95_ms", "itl_p95_ms", "ttft_n"):
        check(f"{k} is in the diagnostics whitelist", src.count(f'"{k}"') >= 2,
              f"counted {src.count(chr(34) + k + chr(34))} -- both the sweep and the "
              f"fixed-concurrency path build their own dict")


# ==========================================================================
def test_closed_loop_stagger():
    """The load driver must pipeline, not convoy."""
    section("_closed_loop: workers are dephased")
    import asyncio, inspect
    from inferopt import evaluator as E

    src = inspect.getsource(E._closed_loop)
    check("stagger is a parameter", "stagger_s" in inspect.signature(E._closed_loop).parameters)
    check("it defaults to OFF so callers opt in",
          inspect.signature(E._closed_loop).parameters["stagger_s"].default == 0.0)
    check("the offset is per-worker, not global", "slot" in src.split("stagger_s > 0")[1][:220])
    check("and capped at settle so no worker starts inside the window",
          "min(stagger_s, settle_s)" in src)

    # MEASURED, turn 3: locked L=128 gave goodput 205.1 and 2181.6 across two
    # identical drives (10.6x); dephased gave 1891.0 and 1893.8 (1.00x), with
    # arrivals peaking at 10.5-14.9x the mean rate locked vs 1.52x dephased.
    started = []

    async def fake_one(client, base_url, model, prompt, max_tokens):
        r = E.Req()
        r.start = asyncio.get_event_loop().time()
        started.append(r.start)
        await asyncio.sleep(0.20)          # every request the same length --
        r.ttft, r.n_out, r.ok = 0.01, 4, True   # which is what locks the convoy
        r.latency = 0.20
        r.token_times = [r.start + 0.05 * j for j in range(4)]
        return r

    def spread(stagger):
        started.clear()
        orig = E._one
        E._one = fake_one
        try:
            asyncio.run(E._closed_loop("http://x", "m", ["p"] * 64, 4, 16,
                                       0.30, 0.40, stagger_s=stagger))
        finally:
            E._one = orig
        base = min(started)
        bins = {}
        for t in started:
            bins[round((t - base) * 20)] = bins.get(round((t - base) * 20), 0) + 1
        return max(bins.values()), len(started)

    locked_peak, locked_n = spread(0.0)
    deph_peak, deph_n = spread(0.20)
    check("locked, every worker fires in the same instant",
          locked_peak >= 16, f"peak {locked_peak} arrivals in one 50ms bin of {locked_n}")
    check("staggered, arrivals spread out",
          deph_peak < locked_peak,
          f"peak {deph_peak} vs {locked_peak} -- the convoy is what made goodput "
          f"swing 10.6x across identical drives")
    check("the stagger does not starve the loop",
          deph_n >= locked_n * 0.5, f"{deph_n} vs {locked_n} requests issued")

    section("VllmEvaluator: the estimate survives the settle clamp")
    src = inspect.getsource(E.VllmEvaluator.__init__)
    check("one_request_s is stored, not recovered from settle_s",
          "self.one_request_s = one_request_s" in src,
          "settle_s is clamped at both ends; dividing by 1.2 recovers the floor, "
          "not the estimate")
    check("and the sweep path passes it",
          "stagger_s=getattr(self, \"one_request_s\"" in inspect.getsource(E.VllmEvaluator._point))


# ==========================================================================
def test_replay_lengths():
    """Every request carries the trace's OWN output length, not the mean."""
    section("_mt: scalar for probes, per-request for load")
    import asyncio, inspect, json, tempfile
    from pathlib import Path
    from inferopt import evaluator as E

    check("a scalar still resolves", E._mt(259, 7) == 259)
    check("a sequence indexes", E._mt([10, 20, 30], 1) == 20)
    check("...and wraps with the prompt cursor", E._mt([10, 20, 30], 4) == 20)
    check("an empty sequence cannot return 0 tokens", E._mt([], 0) == 1)

    section("the driver pairs each prompt with ITS OWN length")
    seen = []

    async def fake_one(client, base_url, model, prompt, max_tokens, stream=True):
        seen.append((prompt, max_tokens))
        r = E.Req(start=asyncio.get_event_loop().time())
        await asyncio.sleep(0.01 * max_tokens)
        r.ttft, r.n_out, r.ok, r.latency = 0.005, max_tokens, True, 0.01 * max_tokens
        r.token_times = [r.start + 0.005 * j for j in range(max_tokens)]
        return r

    prompts = [f"p{i}" for i in range(8)]
    lengths = [3, 9, 3, 9, 3, 9, 3, 9]
    orig = E._one
    E._one = fake_one
    try:
        asyncio.run(E._closed_loop("http://x", "m", prompts, lengths, 4, 0.05, 0.15))
    finally:
        E._one = orig
    check("something ran", len(seen) >= 4, f"{len(seen)} requests")
    paired = {p: mt for p, mt in seen}
    check("prompt p1 always got length 9", paired.get("p1") == 9, paired)
    check("prompt p0 always got length 3", paired.get("p0") == 3, paired)
    check("the served lengths VARY", len({mt for _, mt in seen}) > 1,
          "a constant here is the original bug: trace_shared.jsonl has sd 167.7 "
          "on output_tokens and all of it collapsed to int(mean)=259")

    section("replay_lengths: clamped to the context actually served")

    class Fake(E.VllmEvaluator):
        def __init__(self):
            self.prompts = ["a", "b", "c"]
            self.out_tokens = [100, 1464, 200]
            self.in_tokens = [10, 4431, 20]
            self.max_tokens = 259

    f = Fake()
    check("unclamped when no server has been launched", f.replay_lengths() == [100, 1464, 200])
    f._served_max_len = 6144
    got = f.replay_lengths()
    check("a length that fits is untouched", got[0] == 100 and got[2] == 200, got)
    check("the 4431+1464 row is clamped under the context",
          got[1] + 4431 + E.CONTEXT_MARGIN_TOKENS <= 6144, got)
    f._served_max_len = 2048
    got = f.replay_lengths()
    check("a prompt that alone overflows still yields a legal request",
          all(g >= 1 for g in got), got)
    check("...and shrinks rather than 400s",
          got[1] == 1, "_one swallows HTTP 400 into ok=False, so an over-length "
                       "request would vanish from the window instead of failing")

    section("a subclass that never set out_tokens still works")

    class Bare(E.VllmEvaluator):
        def __init__(self):
            self.prompts, self.max_tokens = ["a", "b"], 8

    check("falls back to the mean", Bare().replay_lengths() == [8, 8])

    section("the load paths do NOT send the mean")
    for name in ("serving_metrics", "_point", "measure"):
        src = inspect.getsource(getattr(E.VllmEvaluator, name))
        if "_load(" in src or "_closed_loop(" in src:
            check(f"{name} passes replay_lengths()", "self.replay_lengths()" in src)
            check(f"{name} no longer passes self.max_tokens as the load length",
                  "self.prompts, self.max_tokens" not in src)

    section("out_tokens stays aligned with prompts")
    rows = [{"prompt": "keep", "output_tokens": 11, "input_tokens": 3},
            {"output_tokens": 999, "input_tokens": 3},          # no prompt: DROPPED
            {"prompt": "keep2", "output_tokens": 22, "input_tokens": 4}]
    d = Path(tempfile.mkdtemp()) / "t.jsonl"
    d.write_text("".join(json.dumps(r) + "\n" for r in rows))
    src = inspect.getsource(E.VllmEvaluator.__init__)
    check("prompts and lengths are built from ONE filtered list",
          src.count("for r in replay") >= 2 and "if r.get(\"prompt\")" in src,
          "two independent comprehensions over `rows` would silently offset the "
          "lengths by every prompt-less row")


# ==========================================================================
def test_slo_explore():
    """The SLO must be movable after the run, not only before it."""
    section("dump_requests -> recompute reproduces summarize EXACTLY")
    import random, tempfile
    from pathlib import Path
    from inferopt import evaluator as E
    from inferopt.fingerprint import SLO
    from inferopt.slo_explore import load, recompute

    slo = SLO(ttft_p99_ms=500, itl_p99_ms=250)
    rnd = random.Random(7)
    t0, t1 = 100.0, 145.0
    reqs = []
    for i in range(300):
        r = E.Req()
        # Deliberately messy: some start BEFORE the window (summarize counts
        # their tokens but not their percentiles), some fail outright, and the
        # TTFTs straddle the bound so attainment is neither 0 nor 1.
        r.start = t0 - 3.0 + i * 0.16
        r.ok = i % 37 != 0
        r.ttft = rnd.uniform(0.05, 1.2)
        r.n_out = rnd.randint(2, 40)
        r.latency = r.ttft + r.n_out * rnd.uniform(0.02, 0.4)
        r.token_times = [r.start + r.ttft + j * 0.03 for j in range(r.n_out)]
        if not r.ok:
            # A request that failed produced no tokens. Leaving them attached
            # made throughput count work that never happened, which is why the
            # loose-bound check below could not hold.
            r.ttft, r.n_out, r.token_times = None, 0, []
        reqs.append(r)

    want = E.summarize(reqs, t0, t1, slo)

    class Fake(E.VllmEvaluator):
        def __init__(self, d):
            self.run_dir, self.slo, self.log = Path(d), slo, lambda *a: None

    d = tempfile.mkdtemp()
    f = Fake(d)
    f.dump_requests(reqs, t0, t1, node_id="n", concurrency=64, phase="closed_loop")
    pts = load(d)
    check("one measurement point was written", len(pts) == 1, len(pts))
    check("every request is kept, in-window or not",
          len(pts[0][1]) == len(reqs), f"{len(pts[0][1])} of {len(reqs)}")

    got = recompute(pts[0][0], pts[0][1], slo.ttft_p99_ms, slo.itl_p99_ms)
    for k in ("goodput", "throughput", "goodput_req_s", "throughput_req_s",
              "slo_attainment", "ttft_p99_ms", "ttft_p95_ms", "itl_p99_ms",
              "itl_p95_ms", "ttft_n", "completed"):
        a, b = want[k], got[k]
        close = abs(a - b) <= max(1e-6, abs(a) * 1e-4)
        check(f"{k} reproduced from disk", close, f"summarize {a!r} vs recompute {b!r}")

    section("...and MOVES when the bound moves")
    tight = recompute(pts[0][0], pts[0][1], 100.0, 250.0)
    loose = recompute(pts[0][0], pts[0][1], 5000.0, 5000.0)
    check("a tighter TTFT bound lowers attainment",
          tight["slo_attainment"] < got["slo_attainment"],
          f"{tight['slo_attainment']:.3f} vs {got['slo_attainment']:.3f}")
    check("a tighter bound lowers goodput", tight["goodput"] < got["goodput"])
    check("an unreachable-loose bound makes goodput == throughput",
          abs(loose["goodput"] - loose["throughput"]) < 1e-6,
          "every completion conforms, so nothing is discounted")
    check("throughput is INVARIANT to the SLO",
          abs(tight["throughput"] - loose["throughput"]) < 1e-6,
          "the server did the same work; only what counts changed")

    section("replicas and cost fall out of goodput_req_s")
    r16 = recompute(pts[0][0], pts[0][1], 500.0, 250.0, demand_qps=16.0,
                    gpu_hourly_usd=3.0)
    check("a replica count is produced", r16["replicas"] >= 1, r16["replicas"])
    check("it is a CEILING, not a rounding",
          r16["replicas"] >= 16.0 / r16["goodput_req_s"], r16)
    tighter = recompute(pts[0][0], pts[0][1], 100.0, 250.0, demand_qps=16.0,
                        gpu_hourly_usd=3.0)
    check("a tighter SLO needs at least as many replicas",
          tighter["replicas"] >= r16["replicas"],
          f"{tighter['replicas']} vs {r16['replicas']} -- this is the whole point "
          f"of the slider: what does the promise cost")
    check("price per hour follows the replica count",
          tighter["usd_per_hour"] >= r16["usd_per_hour"])
    check("no run supplies the GPU price", "gpu_hourly_usd" not in str(pts[0][0]),
          "it is the operator's number, passed in, never measured")

    section("_meets stays in step with Req.meets")
    from inferopt.slo_explore import _meets
    for r in reqs[:60]:
        row = {"k": r.ok, "t": (r.ttft * 1e3 if r.ttft is not None else None),
               "l": r.latency * 1e3, "n": r.n_out, "s": 0.0, "w": 0}
        check_quiet = _meets(row, slo.ttft_p99_ms, slo.itl_p99_ms) == r.meets(slo)
        if not check_quiet:
            break
    check("the two predicates agree request by request", check_quiet,
          "meets() uses MEAN itl (latency-ttft)/(n_out-1); a reader using max "
          "would silently disagree on exactly the borderline requests")

    section("a run without the file says so")
    empty = tempfile.mkdtemp()
    try:
        load(empty)
        check("missing capture raises", False)
    except FileNotFoundError as e:
        check("missing capture raises, and explains why a p99 cannot substitute",
              "p99" in str(e))


# ==========================================================================
def test_review_fixes():
    """Defects found by the line-by-line review of the measurement path."""
    import inspect, statistics, tempfile
    from pathlib import Path
    from inferopt import evaluator as E
    from inferopt.fingerprint import SLO

    section("summarize: a request that FAILED is a miss, not an exclusion")
    slo = SLO(ttft_p99_ms=500, itl_p99_ms=250)

    def mk(n_fail, n=100):
        out = []
        for i in range(n):
            r = E.Req()
            r.start, r.ok = 100.0 + i * 0.1, i >= n_fail
            if r.ok:
                r.ttft, r.n_out = 0.05, 10
                r.latency = r.ttft + 0.5
                r.token_times = [r.start + r.ttft + j * 0.05 for j in range(10)]
            else:
                r.error = "HTTP 400"
            out.append(r)
        return E.summarize(out, 100.0, 145.0, slo)

    clean, broken = mk(0), mk(20)
    check("with no failures attainment is 1.0", abs(clean["slo_attainment"] - 1.0) < 1e-9)
    check("20 failures of 100 make attainment 0.80",
          abs(broken["slo_attainment"] - 0.80) < 1e-9,
          f"{broken['slo_attainment']:.3f} -- dividing by `done` would report 1.000 "
          f"while a fifth of requests errored")
    check("the failure count is still reported", broken["failed"] == 20)
    check("and WHY they failed", broken["failure_reasons"] == {"HTTP 400": 20},
          broken["failure_reasons"])
    section("_one records WHY, from the real exception path")
    import asyncio, httpx
    async def probe(url):
        async with httpx.AsyncClient() as c:
            return await E._one(c, url, "m", "hello", 4)
    # Nothing listening: the request raises inside _one and used to be swallowed
    # by a bare `except Exception: pass`, leaving ok=False and no reason.
    r = asyncio.run(probe("http://127.0.0.1:1"))
    check("a dead endpoint is not ok", not r.ok)
    check("and says why", bool(r.error),
          "a bare except left every failure mode identical: an HTTP 400 from an "
          "over-length generation looked exactly like flaky networking")
    check("the reason names the exception type", ":" in r.error, r.error)

    class Resp:
        status_code = 400
        async def aiter_lines(self):
            if False:
                yield ""
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
    class Client:
        def stream(self, *a, **k): return Resp()
    r2 = asyncio.run(E._one(Client(), "http://x", "m", "p", 4))
    check("a non-200 records its status", r2.error == "HTTP 400", r2.error)

    section("gauges peak, counters delta")
    check("the two are kept apart",
          "kv_cache_util" in E.VllmEvaluator._GAUGES
          and "preemptions" in E.VllmEvaluator._COUNTERS)

    class G(E.VllmEvaluator):
        def __init__(s, series):
            s._series, s._i = series, 0
        def sample_gauges(s):
            v = s._series[min(s._i, len(s._series) - 1)]
            s._i += 1
            return v

    import threading
    # vllm:num_preemptions_total is CUMULATIVE. The server had already recorded
    # 500 before this window opened; 3 happened during it.
    ev = G([{"kv_cache_util": 0.1, "preemptions": 500.0},
            {"kv_cache_util": 0.9, "preemptions": 501.0},
            {"kv_cache_util": 0.4, "preemptions": 503.0}])
    out, stop = {}, threading.Event()
    t = threading.Thread(target=ev._gauge_watch, args=(stop, out, 0.001))
    t.start(); time_waited = 0
    while len(out) < 2 or ev._i < 3:
        stop.wait(0.005)
        time_waited += 1
        if time_waited > 200:
            break
    stop.set(); t.join(timeout=5)
    check("the gauge keeps its PEAK", abs(out.get("kv_cache_util", 0) - 0.9) < 1e-9, out)
    check("the counter reports the DELTA, not 503",
          out.get("preemptions") == 3.0,
          f"{out.get('preemptions')} -- max() of a monotone counter is the count "
          f"since the server booted, which made the curve a staircase that read "
          f"as 'preemptions rise with L'")

    src = inspect.getsource(E.VllmEvaluator.measure)
    check("across levels the gauge is maxed", "for k in self._GAUGES" in src)
    check("...and the counter is SUMMED", "diag[k] = sum(vals)" in src,
          "per-level deltas add up; taking their max would under-report")

    section("the per-request dump names the NODE")
    psrc = inspect.getsource(E.VllmEvaluator._point)
    check("_point takes a node_id", "node_id" in inspect.signature(E.VllmEvaluator._point).parameters)
    check("and writes it, not the sweep label", 'node_id=node_id or "unknown"' in psrc)
    check("never the label", 'node_id=label' not in psrc,
          "one requests.jsonl.gz holds every node in the run; labelling them all "
          "'sweep' makes the file unattributable")
    check("every _point call site passes it", src.count("node_id=node_id") >= 4,
          f"{src.count('node_id=node_id')} of 4 sweep/extend/repeat call sites")

    section("one implementation of the operating point")
    csrc = inspect.getsource(E.VllmEvaluator.capacity)
    check("capacity delegates to peak()", "self.peak(curve)" in csrc)
    check("and does not re-implement it", 'max(curve, key=' not in csrc,
          "its own docstring warns that a duplicated selection is how two "
          "answers come to disagree, and it contained one")

    section("no dead re-selection after the bracket")
    check("the always-false cand block is gone", "cand = max(pts" not in src,
          "cand was max(pts,...) recomputed on an unchanged pts, so the branch "
          "could never fire -- and if pts had ever gained the repeat point it "
          "would have silently burned an extra launch")


# ==========================================================================
def test_pb_spare_contrasts():
    """A 12-run design estimates 11 contrasts; six factors used six."""
    section("spare_contrasts: the columns no factor was assigned to")
    from inferopt.pb_screen import (_pb_full, balance_warning, effects, lenth,
                                    pb_design, spare_contrasts)

    rows, n = _pb_full(6)
    check("six factors buy a 12-run design", n == 12 and len(rows) == 12)
    check("which estimates 11 contrasts", len(rows[0]) == 11)
    check("pb_design still hands back only the assigned six",
          len(pb_design(6)[0][0]) == 6)

    # One real effect on column 0, plus deterministic noise -- a noiseless
    # design gives PSE 0 and the margin comparison below has nothing to compare.
    import random
    rnd = random.Random(3)
    res = [1000.0 + (500 if r[0] else -500) + rnd.gauss(0, 40) for r in rows]
    sp = spare_contrasts(6, res)
    check("five columns are recovered", len(sp) == 5, len(sp))
    check("a factor loaded on column 0 does not leak into them",
          max(abs(x) for x in sp) < 400,
          f"max spare {max(abs(x) for x in sp):.1f} against a 1000 main effect")

    section("...and they are what makes Lenth usable")
    main = [x["effect"] for x in
            effects([r[:6] for r in rows], [{"id": f"f{i}"} for i in range(6)], res)
            if x.get("effect") is not None]
    six, eleven = lenth(main), lenth(main + sp)
    check("six contrasts give df=2", abs(six["df"] - 2.0) < 1e-9)
    check("eleven give df=3.67", abs(eleven["df"] - 11 / 3) < 1e-9)
    check("which is a smaller t multiplier and a tighter margin",
          eleven["me"] < six["me"],
          f"ME {six['me']:.1f} -> {eleven['me']:.1f}. On the real 1.7B screen "
          f"df=2 put t at 4.303 and the margin at 1275 tok/s, so prefix_caching "
          f"-- +929.9 tok/s, +99%, 3.1x the PSE -- was reported NOT active")

    section("balance_warning: failed rows break orthogonality silently")
    check("a complete design is quiet", balance_warning([r[:6] for r in rows], res) is None)
    holed = list(res)
    for i in (0, 3, 5):
        holed[i] = None
    w = balance_warning([r[:6] for r in rows], holed)
    check("a holed design is not", w is not None)
    check("it names the rows", w and "1, 4, 6" in w, w)
    check("and says what it costs",
          w and "confounded with each other" in w,
          "losing rows confounds main effects with EACH OTHER, not merely with "
          "interactions -- the 1.7B screen shipped 5-vs-4 columns and said nothing")
    e = effects([r[:6] for r in rows], [{"id": f"f{i}"} for i in range(6)], holed)
    check("effects() still computes on the unbalanced design",
          e[0].get("effect") is not None,
          "which is why the warning has to exist -- the arithmetic does not object")


# ==========================================================================
def test_parse_metrics_granularity():
    """A Prometheus name is not one number."""
    section("parse_prometheus: every series, labels intact")
    import json as _json
    from inferopt.evaluator import VllmEvaluator as V

    TEXT = """
# HELP vllm:kv_cache_usage_perc GPU KV-cache usage.
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{engine="0",model_name="Qwen/Qwen3-1.7B"} 0.5
vllm:kv_cache_usage_perc{engine="1",model_name="Qwen/Qwen3-1.7B"} 0.4
vllm:num_preemptions_total{engine="0",model_name="Qwen/Qwen3-1.7B"} 7
vllm:num_preemptions_total{engine="1",model_name="Qwen/Qwen3-1.7B"} 5
vllm:prefix_cache_hits_total{engine="0"} 30
vllm:prefix_cache_queries_total{engine="0"} 100
vllm:time_to_first_token_seconds_bucket{le="0.1"} 12
""".strip()

    ser = V.parse_prometheus(TEXT)
    check("both KV series survive parsing", len(ser["vllm:kv_cache_usage_perc"]) == 2)
    check("labels are kept, not discarded",
          ser["vllm:kv_cache_usage_perc"][0][0]["engine"] == "0",
          ser["vllm:kv_cache_usage_perc"][0][0])
    check("a label value containing a slash is intact",
          ser["vllm:kv_cache_usage_perc"][0][0]["model_name"] == "Qwen/Qwen3-1.7B")
    check("comments are skipped", "# HELP" not in "".join(ser))
    check("nothing is summed at parse time",
          [v for _, v in ser["vllm:kv_cache_usage_perc"]] == [0.5, 0.4],
          "reducing here destroys the evidence that a reduction was needed")

    section("_parse_metrics: reduced per METRIC TYPE, not blanket-summed")
    m = V._parse_metrics(V, TEXT)
    check("a fraction takes the MAX across engines",
          abs(m["kv_cache_util"] - 0.5) < 1e-9,
          f"{m['kv_cache_util']} -- summing gave 0.9, and two engines at 0.5 "
          f"each used to report 1.0: a full cache on a half-empty server")
    check("a counter still SUMS", m["preemptions"] == 12.0, m["preemptions"])
    check("ratios are built from summed counters",
          abs(m["prefix_hit_rate"] - 0.30) < 1e-9, m["prefix_hit_rate"])

    section("...and says what it collapsed")
    check("the granular series travel with the scalars", "series" in m)
    check("each carries its labels and value",
          m["series"]["vllm:kv_cache_usage_perc"][1] == {
              "labels": {"engine": "1", "model_name": "Qwen/Qwen3-1.7B"}, "value": 0.4},
          m["series"]["vllm:kv_cache_usage_perc"][1])
    check("a metric with more than one series is FLAGGED",
          "vllm:kv_cache_usage_perc" in m["multi_series"],
          "on a single-engine run this is empty; when it is not, a reduction "
          "was actually exercised and is worth looking at")
    check("a single-series metric is not flagged",
          "vllm:prefix_cache_hits_total" not in m.get("multi_series", []))
    check("histogram buckets are NOT carried into the record",
          "vllm:time_to_first_token_seconds_bucket" not in m.get("series", {}),
          "the full scrape would bloat every trial by orders of magnitude for "
          "data nothing reads")
    check("the whole thing still serialises", bool(_json.dumps(m)))

    section("single engine: unchanged from before")
    one = V._parse_metrics(V, 'vllm:kv_cache_usage_perc{engine="0"} 0.42')
    check("one series reduces to itself", abs(one["kv_cache_util"] - 0.42) < 1e-9)
    check("and is not flagged", "multi_series" not in one)

    section("the launch tag is deterministic")
    import inspect
    src = inspect.getsource(V.measure)
    check("sha256, not hash()", "hashlib.sha256" in src)
    check("hash() is gone", "abs(hash(" not in src,
          "Python randomises string hashing per process, so the same config "
          "produced a different launch directory on every invocation")


# ==========================================================================
def test_resume():
    """Resume a run from its own journal, or refuse to."""
    import json, tempfile
    from pathlib import Path
    from inferopt import resume

    section("resume.plan: fresh, resume, conflict")
    d = Path(tempfile.mkdtemp())
    check("no journal is a fresh run", resume.plan(d, {"a": 1}).mode == "fresh")
    (d / "trials.jsonl").write_text("")
    check("an empty journal is a fresh run", resume.plan(d, {"a": 1}).mode == "fresh")

    stamp = {"model": "m", "gpu": "g", "trace_sha": "abc", "slo": "500/250"}
    rows = [
        {"node_id": "prefix_caching", "config": {"x": 1}, "goodput": 100.0,
         "provenance": stamp},
        {"node_id": "chunked_prefill", "config": {"x": 1}, "goodput": 110.0,
         "provenance": stamp},
        # SAME node, DIFFERENT value: a sweep records both under one id, so a
        # key of node_id alone would replay the first in place of the second.
        {"node_id": "chunked_prefill", "config": {"x": 2}, "goodput": 120.0,
         "provenance": stamp},
    ]
    (d / "trials.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    p = resume.plan(d, stamp)
    check("a matching stamp resumes", p.resuming, p.mode)
    check("the config is part of the key", len(p.cache) == 3, len(p.cache))
    check("and the two sweep values are distinct entries",
          resume.key("chunked_prefill", {"x": 1}) in p.cache
          and resume.key("chunked_prefill", {"x": 2}) in p.cache)
    check("key is order independent",
          resume.key("n", {"a": 1, "b": 2}) == resume.key("n", {"b": 2, "a": 1}))

    section("resume.plan: a different job is not a run to resume")
    other = {**stamp, "slo": "200/100"}
    q = resume.plan(d, other)
    check("a differing stamp is a CONFLICT, not a resume", q.conflict, q.mode)
    check("and the reason names the field that differs", "slo" in q.reason, q.reason)
    check("it does not silently replay", not q.cache)
    # Goodput counts only requests that met the SLO, so the same config against
    # two targets gives numbers that neither compare nor average. Replaying
    # across that boundary is the exact failure trial_stamp exists to stop.
    check("nor silently truncate: the caller decides",
          "--restart" in q.reason and "--run-dir" in q.reason)

    section("resume.plan: a torn last line is what a kill leaves behind")
    (d / "trials.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows) + '{"node_id": "half')
    r2 = resume.plan(d, stamp)
    check("the torn line is skipped, the rest still resumes",
          r2.resuming and len(r2.cache) == 3, len(r2.cache))

    section("duplicates keep the FIRST, so a replay is deterministic")
    (d / "trials.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows + [
        {"node_id": "prefix_caching", "config": {"x": 1}, "goodput": 999.0,
         "provenance": stamp}]))
    r3 = resume.plan(d, stamp)
    check("the duplicate is counted", r3.n_duplicates == 1, r3.n_duplicates)
    check("and the first value wins",
          r3.cache[resume.key("prefix_caching", {"x": 1})]["goodput"] == 100.0)

    section("the evaluator replays instead of launching")
    from inferopt import evaluator as E

    class Fake(E.VllmEvaluator):
        def __init__(self):
            self.replay = p.cache
            self.log = lambda *a: None
        def _serve(self, *a, **k):
            raise AssertionError("a replayed measurement must not launch a server")

    t = Fake().measure({"x": 2}, probes=["goodput"], benchmarks=[],
                       node_id="chunked_prefill")
    check("the cached trial comes back", t.goodput == 120.0, t.goodput)
    check("marked replayed, so callers do not re-append it",
          (t.diagnostics or {}).get("replayed") is True)

    section("launches already spent are credited, not forgotten")
    import inspect
    from inferopt.methods import MethodRunner
    src = inspect.getsource(MethodRunner.__init__)
    check("a resumed runner seeds its trial list from the journal",
          "self.trials.append(resume.to_trial(" in src,
          "otherwise 'pb took 16 launches' stops being true across a restart")
    check("and does not truncate what it is about to replay",
          "if self.plan.resuming" in src and "else:" in src)
    check("measure skips re-appending a replayed trial",
          '"replayed"' in inspect.getsource(MethodRunner.measure))


# ==========================================================================
def test_seed_provenance():
    """A head start must never be invisible."""
    import inspect, json, tempfile
    from pathlib import Path
    from inferopt import api, strategies
    from inferopt.provenance import seed_fingerprint

    section("seed_fingerprint: the starting config is part of the identity")
    a = seed_fingerprint({"max_num_seqs": 256, "enforce_eager": True})
    b = seed_fingerprint({"enforce_eager": True, "max_num_seqs": 256})
    check("key order does not change the digest", a == b, (a, b))
    c = seed_fingerprint({"max_num_seqs": 512, "enforce_eager": True})
    check("a different starting config is a different digest", a["seed_sha"] != c["seed_sha"])
    w = seed_fingerprint({"max_num_seqs": 256}, "runs/previous")
    check("a warm start records where it came from", w["seed_from"] == "runs/previous")
    check("a default seed records no origin", "seed_from" not in a)
    check("an empty seed produces nothing to record", seed_fingerprint(None) == {})

    section("the comments no longer contradict the code")
    # search() takes the seed for EVERY strategy, yolo builds both cells from
    # dict(seed) and the screen builds all twelve rows from it, so a claim that
    # --seed-from-run is sequential-only was false wherever it appeared.
    # The phrase still appears, in a sentence saying the claim was false. What
    # must not survive is the ASSERTION, so check the correction is there rather
    # than that the words are absent.
    # Whitespace-normalised: the docstring is hard wrapped, so a raw substring
    # spanning a line break never matches and the check would fail on formatting
    # rather than on content.
    doc = " ".join(strategies.__doc__.split())
    check("the old claim is marked as history, not stated",
          "used to say the flag applies to the sequential walk only" in doc
          and "It never did" in doc,
          "the screen and yolo both relocate with the seed")
    check("and says what it actually does", "relocates all three" in doc)
    src = inspect.getsource(api.optimize)
    check("api.py no longer claims only the chaining strategy can use it",
          "only the chaining strategy can use" not in src)
    check("api.py says it applies to all three", "ALL THREE" in src)

    section("the stamp carries it, on every trial")
    check("optimize stamps the seed before any measurement",
          "runner.stamp.update(seed_fingerprint(" in src,
          "the stamp is copied onto each trial inside measure(), so it has to be "
          "set before the first one")
    i_stamp = src.index("runner.stamp.update(seed_fingerprint(")
    i_search = src.index("strat.search(")
    check("...and before search runs", i_stamp < i_search)
    rsrc = Path("src/inferopt/run.py").read_text()
    check("the DAG entry point stamps it too",
          "stamp.update(seed_fingerprint(" in rsrc)

    section("a warm start from a different environment warns")
    d = Path(tempfile.mkdtemp())
    (d / "run_meta.json").write_text(json.dumps({
        "environment": {"vllm": "0.24.0"},
        "fingerprint": {"hw": {"gpu_name": "NVIDIA GB10"}}}))
    said = []
    api._warn_if_stale_seed(d, {"vllm": "0.26.0", "gpu": "NVIDIA GB10"}, said.append)
    joined = " ".join(said)
    check("it warns when vLLM changed", "WARNING" in joined, joined[:90])
    check("and names the change", "0.24.0 -> 0.26.0" in joined, joined[:120])
    check("it warns rather than refusing", "Drop the flag" in joined,
          "after an upgrade is exactly when you re-optimize, so refusing would "
          "block the case the flag exists for")
    quiet = []
    api._warn_if_stale_seed(d, {"vllm": "0.24.0", "gpu": "NVIDIA GB10"}, quiet.append)
    check("and is silent when nothing changed", not quiet, quiet)
    gone = []
    api._warn_if_stale_seed(Path(tempfile.mkdtemp()), {"vllm": "0.26.0"}, gone.append)
    check("a missing run_meta is not fatal", not gone)


# ==========================================================================
def test_dag_file():
    section("dag/llm.json: structural invariants")
    d = json.loads(_DAG.read_text())
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
    from inferopt.quality import BENCHMARKS
    for n in d["nodes"]:
        for b in (n.get("quality_benchmarks") or []):
            check(f"{n['id']} asks for a registered benchmark: {b}", b in BENCHMARKS,
                  f"not in quality.py ({sorted(BENCHMARKS)})")
    for b in d.get("benchmarks", []):
        check(f"declared benchmark {b['id']} is registered", b["id"] in BENCHMARKS,
              f"declared in the DAG but absent from quality.py")

    # Every predicate parses AND type-checks against the real schema.
    from inferopt.predicates import Predicate
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
    d = json.loads(_DAG.read_text())
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
    d = json.loads(_DAG.read_text())
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
               test_trial_axes, test_frontier, test_pb_design, test_replay, test_moe_backend_and_int_flags,
               test_qps_source, test_methods_comparable, test_doe_analysis, test_seed_from_run, test_api_types, test_judges, test_legality, test_percentile_stability, test_closed_loop_stagger, test_replay_lengths, test_slo_explore, test_review_fixes, test_pb_spare_contrasts, test_parse_metrics_granularity, test_resume, test_seed_provenance, test_benchmark_surface, test_run_benchmark_guards, test_slo_attainment, test_strategies, test_result_api, test_dag_file,
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
