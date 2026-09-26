"""A journal from another workload is set aside, not a reason to exit.

The control plane hands one directory per job and the trace legitimately
changes when it is re-measured; the first DiffusionGemma rerun died at this
check with a SystemExit nobody reported."""

import json

from inferopt import prompting
from inferopt.methods import MethodRunner
from test_dag_unit import _ctx


def _trace(tmp_path):
    path = tmp_path / "trace.jsonl"
    path.write_text(json.dumps({"prompt": "2+2?", "input_tokens": 5, "output_tokens": 8}) + "\n")
    return str(path)


def test_a_journal_from_another_workload_is_kept_aside_and_the_run_starts_fresh(tmp_path, monkeypatch):
    monkeypatch.setattr(prompting.AutoTokenizer, "from_pretrained",
                        staticmethod(lambda *a, **k: (_ for _ in ()).throw(OSError("offline"))))
    prompting._FORMATTERS.clear()
    ctx = _ctx()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    foreign = {"node_id": "incumbent", "config": {}, "goodput": 2.3, "ttft_p99_ms": 1.0,
               "itl_p99_ms": 1.0, "memory_gb": 1.0,
               "provenance": {"model": "test/model", "trace": "trace.jsonl",
                              "trace_sha": "000000000000", "trace_rows": 1}}
    (run_dir / "trials.jsonl").write_text(json.dumps(foreign) + "\n")
    notes = []

    runner = MethodRunner("sequential", ctx.fingerprint, ctx.slo, _trace(tmp_path), run_dir,
                          log=notes.append)

    assert runner.plan.mode == "fresh" and runner.trials == []
    kept = list(run_dir.glob("trials.*.superseded.jsonl"))
    assert len(kept) == 1 and json.loads(kept[0].read_text())["goodput"] == 2.3
    assert (run_dir / "trials.jsonl").read_text() == ""
    assert any("trace_sha differ" in n for n in notes)
    assert any("another workload" in n and "starting fresh" in n for n in notes)
