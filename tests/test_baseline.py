"""The baseline is stock `vllm serve`: the engine's own defaults plus what the
hardware needs to start. See run.seed_config and engines.VllmEngine.stock.

On Qwen3-8B on the GB10 the old conservative seed measured 181 tok/s, stock
vLLM 711 and the walk's answer from that seed 476, below stock: the reported
lift was a lift over a handicap (1 Oct 2026).
"""

import inspect
import json
from pathlib import Path

import pytest

from inferopt import api, engines, lines, run, traverse
from inferopt.evaluator import SWEEP_LEVELS, VllmEvaluator
from inferopt.fingerprint import (SLO, Context, Fingerprint, HardwareFingerprint, LoraFingerprint, ModelFingerprint,
                                  NodeMeasurement, WorkloadFingerprint)

dag_path = Path(__file__).resolve().parent.parent / "src" / "inferopt" / "dag" / "llm.json"
gigabyte = 1024 ** 3


def a_fingerprint(gpu_name="NVIDIA H100 80GB HBM3", memory_gb=79.6, unified=False):
    model = ModelFingerprint(
        id="Qwen/Qwen3-8B", architecture="Qwen3ForCausalLM", n_params_b=8.2, n_layers=36, hidden_size=4096,
        n_heads=32, n_kv_heads=8, attention_type="gqa", max_model_len=40960, bytes_per_param=2.0)
    hw = HardwareFingerprint(
        gpu_name=gpu_name, gpu_count=1, compute_capability="9.0", memory_gb=memory_gb,
        memory_bandwidth_gb_s=273.0 if unified else 3350.0, unified_memory=unified, system_ram_gb=memory_gb if unified else 200.0,
        cpu_cores=32)
    workload = WorkloadFingerprint(
        n_requests=500, mean_input_tokens=620.0, p99_input_tokens=2660, p999_input_tokens=4380,
        mean_output_tokens=260.0, p99_output_tokens=804, request_rate_qps=1.0, max_concurrency=32, burstiness=1.0,
        prefix_overlap=0.31, prefix_overlap_per_adapter=0.31, multi_turn=False, greedy=True, temperature=0.0,
        top_p=1.0, structured_generation=0.0, trace_ref="x")
    return Fingerprint(model=model, hw=hw, workload=workload, lora=LoraFingerprint())


# ================================================================  stock ===

def test_the_seed_is_stock_vllm_serve_written_out():
    seed = run.seed_config(a_fingerprint())
    assert seed == {"enable_prefix_caching": True, "enable_chunked_prefill": True, "enforce_eager": False,
                    "max_model_len": 40960, "max_num_seqs": 1024, "max_num_batched_tokens": 8192,
                    "gpu_memory_utilization": 0.90}


@pytest.mark.parametrize("gpu_name, memory_gb, unified, limits", [
    ("NVIDIA H100 80GB HBM3", 79.6, False, (1024, 8192)),
    ("NVIDIA A100-SXM4-80GB", 79.2, False, (256, 2048)),
    ("NVIDIA B200", 178.4, False, (1024, 16384)),
    ("NVIDIA GeForce RTX 4090", 23.6, False, (256, 2048)),
    ("NVIDIA L40S", 44.4, False, (256, 2048)),
    ("NVIDIA GB10", 121.7, True, (1024, 8192)),
])
def test_the_batch_limits_are_the_ones_vllm_picks_by_device_memory(gpu_name, memory_gb, unified, limits):
    """vLLM 0.29, EngineArgs._set_default_max_num_seqs_and_batched_tokens_args,
    for the API server: 160 GiB and up, 70 GiB and up but not an A100, else."""
    stock = engines.engine_for(a_fingerprint(gpu_name, memory_gb, unified)).stock(
        a_fingerprint(gpu_name, memory_gb, unified))
    assert (stock["max_num_seqs"], stock["max_num_batched_tokens"]) == limits


def test_the_screen_keeps_every_factor_off_in_its_base_row():
    base = run.factors_off_config(a_fingerprint())
    assert (base["enable_prefix_caching"], base["enable_chunked_prefill"], base["enforce_eager"]) == \
        (False, False, True)


# =======================================================  unified memory ===

def meminfo(available_gib, total_gib=121.7):
    return f"MemTotal: {int(total_gib * 1024 * 1024)} kB\nMemAvailable: {int(available_gib * 1024 * 1024)} kB\n"


@pytest.mark.parametrize("available, share", [
    (121.0, 0.71),     # an idle box
    (115.0, 0.66),     # the GB10 with nothing else running
    (102.1, 0.56),     # another service holding 16 GB
    (30.0, 0.30),      # nearly full: the floor, and vLLM says plainly it does not fit
])
def test_a_unified_memory_box_gives_a_server_what_is_free_less_what_is_kept(available, share):
    assert engines.unified_fraction(meminfo(available)) == share


def test_without_meminfo_the_old_share_stands():
    assert engines.unified_fraction("nothing here") == engines.UNIFIED_CEILING


def test_the_fraction_leaves_the_oom_guard_its_share():
    total = 121.7
    for available in (60.0, 80.0, 102.1, 110.0):
        share = engines.unified_fraction(meminfo(available, total))
        left = available - share * total
        assert share == engines.UNIFIED_FLOOR or left >= engines.KEEP_FREE_SHARE * total, (available, share)


# ===================================================  nodes that change nothing ===

class Recorder:
    """An evaluator that measures every config the same, noting each launch."""

    replay = None

    def __init__(self):
        self.launched = []

    def measure(self, config, *, probes, benchmarks, node_id, **_):
        self.launched.append((node_id, dict(config)))
        return traverse.Trial(node_id=node_id, config=dict(config), goodput=100.0, ttft_p99_ms=100.0,
                              itl_p99_ms=20.0, memory_gb=10.0, slo_ok=True, concurrency=16)


def walk_from_stock():
    fingerprint = a_fingerprint()
    seed = run.seed_config(fingerprint)
    context = Context(fingerprint=fingerprint, slo=SLO(ttft_p99_ms=500, itl_p99_ms=250, quality_budget=0.1,
                                                      lossless_quality_budget=0.03), incumbent=seed)
    context.incumbent_metrics = NodeMeasurement(goodput=100.0, ttft_p99_ms=100.0, itl_p99_ms=20.0, quality={},
                                                config=seed)
    said, recorder = [], Recorder()
    result = traverse.traverse(json.loads(dag_path.read_text()), context, recorder, log=said.append,
                               lossless_only=True)
    return seed, said, recorder, result


def test_a_node_that_sets_only_what_stock_has_is_skipped_not_launched():
    seed, said, recorder, result = walk_from_stock()
    launched = {node_id for node_id, _config in recorder.launched}
    assert "graph_capture" not in launched and "prefix_caching" not in launched
    assert lines.skip("graph_capture", "nothing to change: the incumbent already has it") in said
    assert all(config != seed for node_id, config in recorder.launched if node_id != "lossless_complete"), \
        "no launch of the baseline itself, which would measure only noise"


def test_a_sweep_measures_only_the_values_stock_does_not_already_have():
    seed, said, recorder, result = walk_from_stock()
    chunked = [config for node_id, config in recorder.launched if node_id == "chunked_prefill"]
    assert [config["max_num_batched_tokens"] for config in chunked] == [2048], "8192 is stock on this card"


def test_a_node_whose_launch_is_the_point_is_still_measured():
    seed, said, recorder, result = walk_from_stock()
    assert "lossless_complete" in {node_id for node_id, _config in recorder.launched}


# ======================================================  the baseline's sweep ===

class LadderRecorder:
    """An evaluator that names its baseline ladder and notes how each config
    was asked to be measured; the baseline peaks at 128 in flight."""

    replay = None
    baseline_levels = (4, 8, 16, 32, 64, 128, 256)

    def __init__(self):
        self.asked = []

    def measure(self, config, *, probes, benchmarks, node_id, concurrency=None, levels=None,
                fixed_concurrency=None):
        self.asked.append((node_id, concurrency, levels))
        peak = 128 if node_id == "incumbent" else (concurrency or 16)
        return traverse.Trial(node_id=node_id, config=dict(config), goodput=711.0, ttft_p99_ms=100.0,
                              itl_p99_ms=20.0, memory_gb=10.0, slo_ok=True, concurrency=peak)


def walk_measuring_its_own_baseline(fixed_concurrency=None):
    fingerprint = a_fingerprint()
    context = Context(fingerprint=fingerprint, slo=SLO(ttft_p99_ms=500, itl_p99_ms=250, quality_budget=0.1,
                                                      lossless_quality_budget=0.03),
                      incumbent=run.seed_config(fingerprint))
    recorder = LadderRecorder()
    traverse.traverse(json.loads(dag_path.read_text()), context, recorder, log=list().append, lossless_only=True,
                      concurrency=16, fixed_concurrency=fixed_concurrency)
    return recorder.asked


def test_a_baseline_the_walk_measures_itself_is_swept_across_the_whole_ladder():
    """It was bracketed from the workload's load like a node: on Qwen3-8B it
    stopped at 16 in flight (151) while stock vLLM peaks at 128 (711), and the
    first node, bracketed upward from 16, was kept for +372% it did not earn."""
    asked = walk_measuring_its_own_baseline()
    assert asked[0] == ("incumbent", 16, LadderRecorder.baseline_levels)


def test_the_baselines_peak_is_where_the_first_node_starts():
    asked = walk_measuring_its_own_baseline()
    first_node = next(entry for entry in asked if entry[0] != "incumbent")
    assert first_node[1] == 128 and first_node[2] is None, "a node is bracketed, around the baseline's peak"


def test_a_walk_at_a_fixed_load_measures_its_baseline_there_too():
    asked = walk_measuring_its_own_baseline(fixed_concurrency=30)
    assert asked[0][2] is None


def test_the_vllm_evaluator_sweeps_a_baseline_from_4_to_256():
    assert VllmEvaluator.baseline_levels == SWEEP_LEVELS == (4, 8, 16, 32, 64, 128, 256)


# ==================================================  stock and the prediction ===

class TwoStarts:
    """An evaluator for a stock baseline and one predicted shape (512
    sequences): each measures as told; every node measures like its base."""

    replay = None
    baseline_levels = (4, 8, 16, 32, 64, 128, 256)

    def __init__(self, stock, predicted, predicted_meets_targets=True):
        self.goodput = {"stock": stock, "predicted": predicted}
        self.meets = predicted_meets_targets
        self.asked = []

    def measure(self, config, *, probes, benchmarks, node_id, concurrency=None, levels=None,
                fixed_concurrency=None):
        which = "predicted" if config.get("max_num_seqs") == 512 else "stock"
        self.asked.append((node_id, which, concurrency, levels))
        return traverse.Trial(node_id=node_id, config=dict(config), goodput=self.goodput[which],
                              ttft_p99_ms=100.0, itl_p99_ms=20.0, memory_gb=10.0,
                              slo_ok=self.meets or which == "stock", concurrency=192 if which == "predicted" else 128)


def walk_from_both(stock, predicted, predicted_meets_targets=True):
    fingerprint = a_fingerprint()
    seed = run.seed_config(fingerprint)
    start = run.predicted_start(fingerprint, {"max_num_seqs": 512})
    context = Context(fingerprint=fingerprint, slo=SLO(ttft_p99_ms=500, itl_p99_ms=250, quality_budget=0.1,
                                                      lossless_quality_budget=0.03), incumbent=seed)
    evaluator, said = TwoStarts(stock, predicted, predicted_meets_targets), []
    result = traverse.traverse(json.loads(dag_path.read_text()), context, evaluator, log=said.append,
                               lossless_only=True, concurrency=16, starts=[start])
    return evaluator.asked, said, result


def test_stock_and_the_prediction_are_both_measured_across_the_whole_ladder():
    asked, said, result = walk_from_both(700.0, 900.0)
    assert asked[:2] == [("incumbent", "stock", 16, TwoStarts.baseline_levels),
                         ("predicted", "predicted", 128, TwoStarts.baseline_levels)]


def test_a_prediction_that_serves_better_is_where_the_walk_starts():
    asked, said, result = walk_from_both(700.0, 900.0)
    assert lines.predicted(900.0, True) in said
    first_node = next(entry for entry in asked if entry[0] not in ("incumbent", "predicted"))
    assert first_node[1:3] == ("predicted", 192), "built on the prediction, from its peak"
    assert [trial.node_id for trial in result.trials[:2]] == ["incumbent", "predicted"], "stock stays first"
    assert result.trials[1].kept


def test_a_prediction_within_the_band_leaves_stock_as_the_start():
    asked, said, result = walk_from_both(700.0, 720.0)
    assert lines.predicted(720.0, False) in said
    first_node = next(entry for entry in asked if entry[0] not in ("incumbent", "predicted"))
    assert first_node[1:3] == ("stock", 128)


def test_a_prediction_that_misses_the_targets_is_never_the_start():
    asked, said, result = walk_from_both(700.0, 2000.0, predicted_meets_targets=False)
    assert lines.predicted(2000.0, False) in said and not result.trials[1].kept


def test_a_prediction_that_is_stock_already_is_not_measured_again():
    fingerprint = a_fingerprint()
    assert run.predicted_start(fingerprint, {}) == run.seed_config(fingerprint)


def test_the_hardware_rails_win_over_a_predicted_shape():
    fingerprint = a_fingerprint(gpu_name="NVIDIA GB10", memory_gb=121.7, unified=True)
    start = run.predicted_start(fingerprint, {"gpu_memory_utilization": 0.95, "tensor_parallel_size": 2})
    assert start["gpu_memory_utilization"] <= engines.UNIFIED_CEILING and "tensor_parallel_size" not in start


def test_the_platform_path_fetches_the_weights_before_any_launch():
    """Only the command line did: a 30B search through api.optimize downloaded
    inside its first launch, silently, and lost its baseline to the hang
    detector (2 Oct 2026)."""
    source = inspect.getsource(api.optimize)
    assert source.index("prefetch_weights(fp.model.id") < source.index("strat.search(")
