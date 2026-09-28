"""A conversion is bounded: past INFEROPT_QUANTIZE_TIMEOUT_S it is stopped and
its variant skipped, rather than holding the search (and a rented host)."""

import subprocess
import types

import pytest

from inferopt import quantize


def test_a_conversion_past_its_time_is_stopped_and_named(monkeypatch, tmp_path):
    monkeypatch.setenv("INFEROPT_QUANTIZE_TIMEOUT_S", "7")
    monkeypatch.setattr(quantize, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(quantize, "HERE", tmp_path)
    monkeypatch.setattr(quantize, "producer_available", lambda: True)
    monkeypatch.setattr(quantize, "_write_calibration", lambda trace, calib: 4)
    monkeypatch.setattr(quantize, "sensitivity_ignore", lambda fp, log=print: [])
    seen = {}

    def slow(argv, **kwargs):
        seen.update(kwargs)
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
    monkeypatch.setattr(quantize.subprocess, "run", slow)
    fp = types.SimpleNamespace(model=types.SimpleNamespace(id="Qwen/Qwen3-1.7B", decoding="autoregressive",
                                                           dllm_block_size=0))
    with pytest.raises(RuntimeError, match="did not finish in 0 minutes"):
        quantize.ensure_variant(fp, "autoquant@6.0", "trace.jsonl", log=lambda *a: None)
    assert seen["timeout"] == 7.0
    assert not (tmp_path / "Qwen__Qwen3-1.7B--autoquant_6.0").exists(), "a half written artifact is removed"
    monkeypatch.delenv("INFEROPT_QUANTIZE_TIMEOUT_S")
    assert quantize.quantize_timeout_s() == 1800
