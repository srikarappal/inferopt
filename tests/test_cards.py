"""Estimate-only AIConfigurator systems for the cards it has not measured."""

import pytest

from inferopt import cards, predictor
from inferopt.fingerprint import SLO


def test_the_system_files_are_what_the_table_writes(tmp_path):
    """The table is the source; a file edited by hand, or a card added without
    running tools/write_aic_systems.py, fails here."""
    written = cards.write_systems(tmp_path)
    for system in written:
        committed = cards.systems_dir / f"{system}.yaml"
        assert committed.is_file(), f"{system} was never written"
        assert committed.read_text() == (tmp_path / f"{system}.yaml").read_text(), f"{system} is stale"


@pytest.mark.parametrize("name, system", [
    ("NVIDIA GeForce RTX 3080 Ti", "rtx_3080_ti"),
    ("NVIDIA RTX3080 10GB", "rtx_3080"),
    ("NVIDIA GeForce RTX 4070 Ti SUPER", "rtx_4070_ti_super"),
    ("NVIDIA GB10", "gb10"),
    ("NVIDIA DGX Spark (GB10 Grace Blackwell)", "gb10"),
    ("NVIDIA A10", "a10"),
    ("NVIDIA A40", "a40"),
    ("NVIDIA GH200 480GB", "gh200"),
])
def test_a_card_resolves_to_its_own_estimate_with_a_donor_of_its_generation(name, system):
    resolved = cards.system_for(name)
    assert resolved is not None and resolved[0] == system
    donor = resolved[1]
    assert donor and cards.measured(donor), "the efficiency has to come from measurements"
    assert abs(cards.card_spec(donor)["sm_version"] - cards.card_spec(system)["sm_version"]) <= 6


@pytest.mark.parametrize("name, system", [
    ("h100_sxm", "h100_sxm"), ("NVIDIA H100 80GB HBM3", "h100_sxm"), ("NVIDIA A100-SXM4-80GB", "a100_sxm"),
    ("NVIDIA L40S", "l40s"), ("NVIDIA RTX PRO 6000 Blackwell Server Edition", "rtx_pro_6000_server"),
])
def test_a_measured_system_needs_no_donor(name, system):
    if not cards.measured(system):
        pytest.skip("aiconfigurator is not installed here")
    assert cards.system_for(name) == (system, "")


def test_a_shipped_system_without_measurements_borrows_like_ours():
    if cards.shipped_systems_dir() is None:
        pytest.skip("aiconfigurator is not installed here")
    assert cards.system_for("NVIDIA L4") == ("l4", "l40s")


def test_a_card_we_have_no_figures_for_is_none():
    assert cards.system_for("Tesla T4") is None
    assert cards.system_for("") is None


def test_geforce_bf16_is_the_fp32_accumulate_rate():
    """vLLM's bf16 matmuls accumulate in FP32, which GeForce runs at half the
    rate on the box: 165.2 for an RTX 4090, never 330.3."""
    spec = cards.card_spec("rtx_4090")
    assert spec["bf16_tflops"] == pytest.approx(165.2)
    assert spec["bandwidth_gb_s"] == pytest.approx(1008)


def test_a_derated_frontier_is_slower_everywhere_and_judged_again():
    row = {"tokens_s_gpu": 1000.0, "tokens_s_user": 40.0, "req_s": 250.0, "ttft_ms": 100.0,
           "tpot_ms": 25.0, "request_latency_ms": 200.0, "meets_slo": True}
    [derated] = predictor.derate([row], 0.4, SLO(ttft_p99_ms=1000, itl_p99_ms=50))
    assert derated["tokens_s_gpu"] == 400.0 and derated["tokens_s_user"] == 16.0
    assert derated["tpot_ms"] == 62.5 and derated["meets_slo"] is False, "25 ms at 40% is 62.5, over the 50 ms target"


def test_efficiency_is_measured_over_speed_of_light_and_never_above_one():
    assert predictor.efficiency_of({"tokens_s_gpu": 600}, {"tokens_s_gpu": 1000}) == 0.6
    assert predictor.efficiency_of({"tokens_s_gpu": 1200}, {"tokens_s_gpu": 1000}) == 1.0
    assert predictor.efficiency_of({}, {"tokens_s_gpu": 1000}) == 0.0
