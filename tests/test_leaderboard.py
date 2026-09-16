"""Unit tests for the leaderboard integration.

No GPU, no network, no lm_eval. The subprocess is intercepted so the test can
read the config that WOULD have been sent, which is where the interesting
decisions live: which token budget each benchmark dictates, whether a
multiple-choice task was wrongly handed a generation budget, and whether a run
that overrode anything is still being called leaderboard parity.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from inferopt import leaderboard as lb

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


class Capture:
    """Stand in for subprocess.run and keep the config that was written."""

    def __init__(self, results: dict | None = None, returncode: int = 0):
        self.cfgs: list[dict] = []
        self.results = results if results is not None else {}
        self.returncode = returncode
        self.argv: list[list[str]] = []

    def __call__(self, cmd, **kw):
        self.argv.append(list(cmd))
        cfg_path, res_path = Path(cmd[-2]), Path(cmd[-1])
        self.cfgs.append(json.loads(cfg_path.read_text()))
        if self.returncode == 0:
            res_path.write_text(json.dumps({"results": self.results}))
        return subprocess.CompletedProcess(cmd, self.returncode, "", "boom")


def run_with(capture, *a, **kw):
    orig = subprocess.run
    subprocess.run = capture
    try:
        return lb.run(*a, python="/nonexistent/python", log=lambda *_: None, **kw)
    finally:
        subprocess.run = orig


# --------------------------------------------------------------------------
def test_registry():
    section("the six benchmarks, as lm_eval defines them")
    check("all six are present", len(lb.TASKS) == 6, f"{sorted(lb.TASKS)}")
    for want in ("leaderboard_bbh", "leaderboard_gpqa", "leaderboard_mmlu_pro",
                 "leaderboard_musr", "leaderboard_math_hard", "leaderboard_ifeval"):
        check(f"{want} registered", want in lb.TASKS)

    for name, t in lb.TASKS.items():
        if t.generative:
            check(f"{name} carries its own generation budget",
                  isinstance(t.max_gen_toks, int) and t.max_gen_toks > 0,
                  f"got {t.max_gen_toks}")
        else:
            check(f"{name} is multiple-choice and has NO generation budget",
                  t.max_gen_toks is None,
                  "a logprob-scored task generates nothing, so a cap there is a lie")

    # The values that came from lm_eval's own YAMLs. If lm_eval changes them,
    # this fails and we find out rather than silently drifting off parity.
    check("MATH stops at 1024 generated tokens",
          lb.TASKS["leaderboard_math_hard"].max_gen_toks == 1024)
    check("IFEval stops at 1280", lb.TASKS["leaderboard_ifeval"].max_gen_toks == 1280)
    check("MMLU-Pro is 5-shot", lb.TASKS["leaderboard_mmlu_pro"].num_fewshot == 5)
    check("BBH is 3-shot", lb.TASKS["leaderboard_bbh"].num_fewshot == 3)
    check("GPQA and MuSR are zero-shot",
          lb.TASKS["leaderboard_gpqa"].num_fewshot == 0
          and lb.TASKS["leaderboard_musr"].num_fewshot == 0)
    check("the budgets genuinely differ, so one global cap could not serve both",
          lb.TASKS["leaderboard_math_hard"].max_gen_toks
          != lb.TASKS["leaderboard_ifeval"].max_gen_toks)

    check("GENERATIVE names exactly the tasks that decode",
          set(lb.GENERATIVE) == {"leaderboard_math_hard", "leaderboard_ifeval"},
          f"{lb.GENERATIVE}")
    check("metric_key matches lm_eval's result spelling",
          lb.TASKS["leaderboard_bbh"].metric_key() == "acc_norm,none")
    check("describe() renders every task", all(k in lb.describe() for k in lb.TASKS))


def test_preflight():
    section("preflight refuses before a GPU is held, not after")
    p = lb.preflight()
    check("gpqa is flagged as gated", "gated" in p["leaderboard_gpqa"].lower())
    check("the gpqa message says what to actually do",
          "Accept the terms" in p["leaderboard_gpqa"]
          and "huggingface.co/datasets/Idavidrein/gpqa" in p["leaderboard_gpqa"])
    check("a valid token alone is called out as insufficient",
          "not enough" in p["leaderboard_gpqa"])
    check("ungated tasks are clear",
          all(p[k] == "" for k in lb.ALL if not lb.TASKS[k].gated))
    check("an unknown task is named as unknown",
          "unknown task" in lb.preflight(["nope"])["nope"])

    try:
        lb.run(["leaderboard_gpqa"], "http://x", "m", python="/nonexistent")
        check("run refuses a gated task", False, "it did not raise")
    except lb.LeaderboardError as e:
        check("run refuses a gated task before spawning anything", "gated" in str(e))


def test_budgets_are_per_benchmark():
    section("each benchmark dictates its own output token limit")
    cap = Capture({"leaderboard_math_hard": {"exact_match,none": 0.4}})
    run_with(cap, ["leaderboard_math_hard"], "http://h:8000", "m")
    check("MATH is sent its own 1024", cap.cfgs[0]["gen_kwargs"] == {"max_gen_toks": 1024},
          str(cap.cfgs[0]["gen_kwargs"]))

    cap = Capture({"leaderboard_ifeval": {"prompt_level_strict_acc,none": 0.3}})
    run_with(cap, ["leaderboard_ifeval"], "http://h:8000", "m")
    check("IFEval is sent its own 1280", cap.cfgs[0]["gen_kwargs"] == {"max_gen_toks": 1280})

    cap = Capture({"leaderboard_musr": {"acc_norm,none": 0.5}})
    run_with(cap, ["leaderboard_musr"], "http://h:8000", "m",
             gen_toks={"leaderboard_musr": 4096})
    check("a multiple-choice task is never sent a generation budget, even if asked",
          cap.cfgs[0]["gen_kwargs"] is None,
          f"got {cap.cfgs[0]['gen_kwargs']}; it scores logprobs and generates nothing")

    cap = Capture({"leaderboard_math_hard": {"exact_match,none": 0.5}})
    out = run_with(cap, ["leaderboard_math_hard"], "http://h:8000", "m",
                   gen_toks={"leaderboard_math_hard": 8192})
    check("an override reaches the harness",
          cap.cfgs[0]["gen_kwargs"] == {"max_gen_toks": 8192})
    check("the override is reported back", out["leaderboard_math_hard"]["max_gen_toks"] == 8192)


def test_parity_flag():
    section("parity is tracked, so a tuned number cannot pose as a leaderboard score")
    res = {"leaderboard_math_hard": {"exact_match,none": 0.4}}
    out = run_with(Capture(res), ["leaderboard_math_hard"], "http://h", "m")
    check("defaults are parity", out["leaderboard_math_hard"]["parity"] is True)

    out = run_with(Capture(res), ["leaderboard_math_hard"], "http://h", "m",
                   gen_toks={"leaderboard_math_hard": 8192})
    check("raising the token budget breaks parity",
          out["leaderboard_math_hard"]["parity"] is False)

    out = run_with(Capture(res), ["leaderboard_math_hard"], "http://h", "m",
                   num_fewshot={"leaderboard_math_hard": 0})
    check("changing the shot count breaks parity",
          out["leaderboard_math_hard"]["parity"] is False)

    out = run_with(Capture(res), ["leaderboard_math_hard"], "http://h", "m", limit=10)
    check("scoring a subset breaks parity",
          out["leaderboard_math_hard"]["parity"] is False,
          "a limit is a different measurement from the full set")


def test_request_shape():
    section("what actually gets sent")
    cap = Capture({"leaderboard_musr": {"acc_norm,none": 0.5}})
    run_with(cap, ["leaderboard_musr"], "http://host:8100", "Qwen/Qwen3-1.7B",
             max_length=9000, concurrent=8, seed=3)
    c = cap.cfgs[0]
    check("the completions path is appended once",
          c["base_url"] == "http://host:8100/v1/completions", c["base_url"])
    check("an already-complete url is not doubled",
          run_with(Capture({"leaderboard_musr": {"acc_norm,none": 0.1}}),
                   ["leaderboard_musr"], "http://host:8100/v1/completions", "m")
          is not None)
    check("model travels", c["model"] == "Qwen/Qwen3-1.7B")
    check("max_length travels", c["max_length"] == 9000)
    check("concurrency travels", c["concurrent"] == 8)
    check("seed travels", c["seed"] == 3)
    check("the task's own shot count is left alone (None means do not override)",
          c["num_fewshot"] is None)

    # The runner must be invoked by PATH. `-m inferopt.lmeval_runner` imports the
    # inferopt package, whose __init__ needs pydantic, which the lm_eval
    # environment has no reason to have.
    argv = cap.argv[0]
    check("the runner is invoked by path, never with -m",
          "-m" not in argv and argv[1].endswith("lmeval_runner.py"), " ".join(argv[:3]))
    check("the runner file exists where run() points", Path(argv[1]).exists(), argv[1])


def test_failure_is_not_zero():
    section("a benchmark that could not run is None, never 0.0")
    out = run_with(Capture(returncode=1), ["leaderboard_musr"], "http://h", "m")
    row = out["leaderboard_musr"]
    check("score is None on failure", row["score"] is None, str(row))
    check("the error is kept", bool(row.get("error")), str(row))
    check("a failure is never reported as a zero score", row["score"] != 0.0)

    # A task that ran but whose metric is absent must also not become 0.0.
    out = run_with(Capture({"leaderboard_musr": {"something_else,none": 0.9}}),
                   ["leaderboard_musr"], "http://h", "m")
    check("a missing metric key yields None, not 0.0",
          out["leaderboard_musr"]["score"] is None)


def test_missing_harness_is_actionable():
    section("no lm_eval anywhere: say how to fix it")
    orig = lb.lmeval_python
    lb.lmeval_python = lambda: None
    try:
        lb.run(["leaderboard_musr"], "http://h", "m")
        check("run raises when the harness is absent", False, "it did not raise")
    except lb.LeaderboardError as e:
        msg = str(e)
        check("the error explains the install", "python -m venv" in msg)
        check("the error names the env var", "INFEROPT_LMEVAL_PYTHON" in msg)
        check("the error says why it is not a dependency",
              "pinned vLLM" in msg or "do not belong" in msg)
    finally:
        lb.lmeval_python = orig


def test_gate_tolerates_unscored():
    section("the quality gate must not crash or reject on an unscored benchmark")
    import inspect
    from inferopt import traverse
    src = inspect.getsource(traverse)
    check("the gate skips a None score",
          "if ref is None or v is None:" in src,
          "delta = ref - v would raise TypeError on None")
    check("the tolerance re-anchor skips a None score",
          "if ref is not None and v is not None:" in src)
    check("a None score cannot overwrite a real baseline",
          "if v is not None}" in src.replace(" ", "").replace("\n", "")
          or "items() if v is not None}" in src)


# ==========================================================================
def main() -> int:
    for fn in (test_registry, test_preflight, test_budgets_are_per_benchmark,
               test_parity_flag, test_request_shape, test_failure_is_not_zero,
               test_missing_harness_is_actionable, test_gate_tolerates_unscored):
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
    print("  all leaderboard unit checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
