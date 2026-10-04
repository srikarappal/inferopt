"""A card the table does not know takes its bandwidth from the caller: the
control plane rented an RTX 3080 for its bandwidth and the search refused it
for having none on record (28 Sep 2026)."""

import subprocess
import types

import pytest

from inferopt import request


def fake_smi(name):
    def run(command, **kwargs):
        if command[0] == "nvidia-smi":
            return types.SimpleNamespace(stdout=f"{name}, 8.6, 10240\n", returncode=0)
        if command[0] == "lscpu":
            return types.SimpleNamespace(stdout="", returncode=0)
        raise AssertionError(command)
    return run


def a_linux_box(monkeypatch, name):
    monkeypatch.setattr(subprocess, "run", fake_smi(name))
    # /proc/meminfo is read by name; a Mac running the tests has none.
    monkeypatch.setattr(request, "open", lambda path: iter(["MemTotal: 32000000 kB\n"]), raising=False)


def a_request(tmp_path):
    trace = tmp_path / "trace.jsonl"
    trace.write_text('{"prompt": "hi", "output_tokens": 4}\n')
    return request.InferOptRequest.model_construct(model="Qwen/Qwen3-1.7B", trace=str(trace),
                                                   override_memory_bandwidth_gb_s=None)


def test_the_environment_supplies_a_card_the_table_does_not_know(tmp_path, monkeypatch):
    a_linux_box(monkeypatch, "NVIDIA GeForce RTX 3080")
    monkeypatch.setenv(request.BANDWIDTH_ENV, "760")
    assert request.detect_hardware(a_request(tmp_path)).memory_bandwidth_gb_s == 760.0


def test_without_it_an_unknown_card_is_still_refused_loudly(tmp_path, monkeypatch):
    a_linux_box(monkeypatch, "NVIDIA GeForce RTX 3080")
    monkeypatch.delenv(request.BANDWIDTH_ENV, raising=False)
    with pytest.raises(RuntimeError, match=request.BANDWIDTH_ENV):
        request.detect_hardware(a_request(tmp_path))
