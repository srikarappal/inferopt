"""Guards that skipped what mattered for short, compute-bound chat traffic:
Qwen3-1.7B on an RTX 4090, 70-token prompts, 50 ms ITL (28 Sep 2026)."""

import json
from importlib import resources

import pytest

from inferopt.fingerprint import HardwareFingerprint


def nodes():
    dag = json.loads((resources.files("inferopt") / "dag" / "llm.json").read_text())
    found = {}

    def walk(node):
        if isinstance(node, dict):
            if "id" in node:
                found[node["id"]] = node
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(dag)
    return found


@pytest.mark.parametrize("capability, fp8", [("8.6", False), ("8.9", True), ("9.0", True), ("12.1", True)])
def test_ada_has_fp8(capability, fp8):
    card = HardwareFingerprint(gpu_name="G", compute_capability=capability, memory_gb=24,
                               memory_bandwidth_gb_s=1008, system_ram_gb=64, cpu_cores=16)
    assert card.supports_fp8 is fp8


def test_chunked_prefill_and_prefix_caching_run_whenever_there_is_a_target():
    dag = nodes()
    assert "slo.itl_p99_ms" in dag["chunked_prefill"]["applicable_when"]
    assert "slo.ttft_p99_ms" in dag["prefix_caching"]["applicable_when"]
    assert "measurements.chunked_prefill.kept" in dag["max_num_batched_tokens"]["applicable_when"]


def test_plain_fp8_is_measured_beside_the_bit_budgets():
    node = nodes()["weight_autoquantize"]
    assert {"quantize": "fp8"} in node["sweep"]
    assert node["cost_launches"] == len(node["sweep"])
