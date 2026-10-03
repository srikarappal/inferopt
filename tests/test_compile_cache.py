"""Engines on one host share what they compile. See evaluator.compile_cache_env."""

import inspect
import subprocess

import pytest

from inferopt import evaluator

CACHE_VARIABLES = ("VLLM_CACHE_ROOT", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "FLASHINFER_WORKSPACE_BASE")


class SmiAnswer:
    """subprocess.run as nvidia-smi answers it: `out`, or no binary at all."""

    def __init__(self, out=None):
        self.out = out

    def __call__(self, *args, **kwargs):
        if self.out is None:
            raise FileNotFoundError("nvidia-smi")
        return subprocess.CompletedProcess(args, 0, stdout=self.out, stderr="")


@pytest.fixture
def fresh_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("INFEROPT_HOME", str(tmp_path))
    for name in CACHE_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    evaluator.compile_cache_env.cache_clear()
    yield tmp_path
    evaluator.compile_cache_env.cache_clear()


def test_every_launch_on_a_host_shares_one_cache_keyed_by_stack_and_card(monkeypatch, fresh_cache):
    monkeypatch.setattr(evaluator, "_compute_capability", lambda: "sm121")
    monkeypatch.setattr(evaluator, "_installed", lambda package: "1.0")
    env = evaluator.child_env(CUDA_VISIBLE_DEVICES="0")
    root = fresh_cache / "compile-cache" / "vllm1.0-torch1.0-triton1.0-flashinfer1.0" / "sm121"
    assert env["VLLM_CACHE_ROOT"] == str(root / "vllm")
    assert env["TRITON_CACHE_DIR"] == str(root / "triton")
    assert env["TORCHINDUCTOR_CACHE_DIR"] == str(root / "inductor")
    assert env["FLASHINFER_WORKSPACE_BASE"] == str(root / "flashinfer")
    assert env["CUDA_VISIBLE_DEVICES"] == "0"


def test_a_directory_the_operator_set_still_wins(monkeypatch, fresh_cache):
    monkeypatch.setattr(evaluator, "_compute_capability", lambda: "sm90")
    monkeypatch.setenv("TRITON_CACHE_DIR", "/operator/choice")
    env = evaluator.child_env()
    assert env["TRITON_CACHE_DIR"] == "/operator/choice"
    assert env["VLLM_CACHE_ROOT"].endswith("/sm90/vllm")


def test_the_card_is_read_from_nvidia_smi_and_a_host_without_one_still_gets_a_cache(monkeypatch):
    monkeypatch.setattr(evaluator.subprocess, "run", SmiAnswer("12.1\n"))
    assert evaluator._compute_capability() == "sm121"
    monkeypatch.setattr(evaluator.subprocess, "run", SmiAnswer())
    assert evaluator._compute_capability() == "nogpu"


def test_no_launch_keeps_its_compile_cache_in_the_run_directory():
    assert "VLLM_CACHE_ROOT" not in inspect.getsource(evaluator.VllmEvaluator._serve), \
        "a run directory is deleted when the search leaves the host"
