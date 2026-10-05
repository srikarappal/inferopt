"""Quantization calibrates only on the prompts it is handed, when the caller
grades on other rows of the same eval (4 Oct 2026: calibrated on the trace, a
variant's scales were fit on questions it was then graded on)."""

import json
import types

from inferopt import quantize


def a_fingerprint():
    return types.SimpleNamespace(model=types.SimpleNamespace(id="org/model", decoding="autoregressive"))


def produced(monkeypatch, tmp_path):
    """ensure_variant with the producer faked: every conversion writes a
    config.json and is recorded with the calibration file it was given."""
    runs = []

    def fake_run(command, **kwargs):
        # The conversion job: python, job.py, model, kind, out, calibration, ...
        if len(command) > 5 and str(command[1]).endswith(".job.py"):
            (quantize.Path(command[4]) / "config.json").write_text("{}")
            runs.append(quantize.Path(command[5]).read_text())
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(quantize, "artifacts", lambda name: tmp_path / "artifacts" / name)
    monkeypatch.setattr(quantize, "producer_available", lambda: True)
    monkeypatch.setattr(quantize, "sensitivity_ignore", lambda fp, log=print: [])
    monkeypatch.setattr(quantize.subprocess, "run", fake_run)
    (tmp_path / "artifacts").mkdir()
    return runs


def rows(path, prompts):
    path.write_text("".join(json.dumps({"prompt": prompt}) + "\n" for prompt in prompts))
    return str(path)


def test_the_quantizer_reads_the_calibration_set_and_not_the_trace(monkeypatch, tmp_path):
    runs = produced(monkeypatch, tmp_path)
    trace = rows(tmp_path / "trace.jsonl", [f"graded {i}" for i in range(8)] + ["calibrate a", "calibrate b"])
    calibration = rows(tmp_path / "calibration.jsonl", ["calibrate a", "calibrate b"])
    notes = []
    quantize.ensure_variant(a_fingerprint(), "w4a16", trace, log=notes.append, calibration=calibration)
    assert "graded" not in runs[0] and runs[0].count("calibrate") == 2
    assert any("2 prompts from the calibration set, none of them graded" in note for note in notes)


def test_without_a_calibration_set_the_trace_is_the_source_as_for_real_traffic(monkeypatch, tmp_path):
    runs = produced(monkeypatch, tmp_path)
    trace = rows(tmp_path / "trace.jsonl", ["a", "b", "c"])
    quantize.ensure_variant(a_fingerprint(), "w4a16", trace, log=lambda *a: None)
    assert runs[0].count("prompt") == 3


def test_a_checkpoint_calibrated_on_other_prompts_is_produced_again_and_the_same_one_reused(monkeypatch, tmp_path):
    """The checkpoint's name carries no calibration set, so one calibrated on
    the whole trace before this would otherwise be reused as it was."""
    runs = produced(monkeypatch, tmp_path)
    trace = rows(tmp_path / "trace.jsonl", ["graded", "calibrate a"])
    calibration = rows(tmp_path / "calibration.jsonl", ["calibrate a"])
    quantize.ensure_variant(a_fingerprint(), "w4a16", trace, log=lambda *a: None)
    notes = []
    quantize.ensure_variant(a_fingerprint(), "w4a16", trace, log=notes.append, calibration=calibration)
    assert len(runs) == 2 and "graded" not in runs[1]
    assert any("calibrated on other prompts" in note for note in notes)
    quantize.ensure_variant(a_fingerprint(), "w4a16", trace, log=notes.append, calibration=calibration)
    quantize.ensure_variant(a_fingerprint(), "w4a16", trace, log=notes.append)
    assert len(runs) == 2, "the same calibration set, or none asked for: reused"
