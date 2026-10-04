"""A run handed several cards splits its model across all of them: the
platform hands it the fewest cards that hold the model, so stock vLLM's single
card would not load it (3 Oct 2026)."""

import subprocess
import types

import pytest

from inferopt import engines, request
from inferopt.fingerprint import Fingerprint


def a_fingerprint(gpu_count, dense=True, unified=False):
    return types.SimpleNamespace(
        hw=types.SimpleNamespace(gpu_count=gpu_count, unified_memory=unified, sm_major=9),
        model=types.SimpleNamespace(is_dense=dense, decoding="autoregressive"))


def four_cards(command, **kwargs):
    if command[0] == "nvidia-smi":
        return types.SimpleNamespace(stdout="NVIDIA H100 80GB HBM3, 9.0, 81559\n" * 4, returncode=0)
    raise AssertionError(command)


def a_request(tmp_path):
    trace = tmp_path / "trace.jsonl"
    trace.write_text('{"prompt": "hi", "output_tokens": 4}\n')
    return request.InferOptRequest.model_construct(model="Qwen/Qwen3-32B", trace=str(trace),
                                                   override_memory_bandwidth_gb_s=None)


def test_one_card_keeps_stock_and_several_split_the_model():
    assert engines.split_across(a_fingerprint(1)) == {}
    assert engines.split_across(a_fingerprint(4)) == {"tensor_parallel_size": 4}


def test_both_engines_put_the_split_under_every_launch(monkeypatch):
    monkeypatch.setattr(engines.Engine, "installed_flags", lambda self: set())
    assert engines.VllmEngine().defaults(a_fingerprint(2))["tensor_parallel_size"] == 2
    assert engines.SglangEngine().defaults(a_fingerprint(2))["tensor_parallel_size"] == 2
    assert "tensor_parallel_size" not in engines.VllmEngine().defaults(a_fingerprint(1))


def test_a_run_counts_only_the_cards_it_was_handed(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", four_cards)
    monkeypatch.setattr(request, "open", lambda path: iter(["MemTotal: 32000000 kB\n"]), raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert request.detect_hardware(a_request(tmp_path)).gpu_count == 4
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    assert request.detect_hardware(a_request(tmp_path)).gpu_count == 2
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,7")
    assert request.detect_hardware(a_request(tmp_path)).gpu_count == 4, "a list naming cards it lacks is ignored"


def test_a_model_no_one_card_holds_is_checked_against_every_card_it_was_handed():
    """3 Oct 2026, live: Qwen3-8B's 16.4 GB was refused on four 12 GB cards as
    exceeding device memory, before vLLM was ever asked to split it."""
    model = types.SimpleNamespace(max_model_len=40960, kv_bytes_per_token=147456, weight_gb=16.4)
    workload = types.SimpleNamespace(p999_input_tokens=400, p99_output_tokens=600)
    four = types.SimpleNamespace(memory_gb=12.0, gpu_count=4)
    one = types.SimpleNamespace(memory_gb=12.0, gpu_count=1)
    assert Fingerprint.model_construct(model=model, hw=four, workload=workload)._cross() is not None
    with pytest.raises(ValueError, match="across 1 card"):
        Fingerprint.model_construct(model=model, hw=one, workload=workload)._cross()
