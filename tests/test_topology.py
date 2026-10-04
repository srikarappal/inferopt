"""What a run's cards sit on (inferopt.topology): every multi card host used to
be written down as NVLink, including four RTX 3060s on PCIe without peer to
peer (4 Oct 2026). The nvidia-smi outputs below are its own formats."""

import subprocess
import types

from inferopt import request, topology

PCIE_MATRIX = (
    "\t\x1b[4mGPU0\tGPU1\tGPU2\tGPU3\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID\x1b[0m\n"
    "GPU0\t X \tPHB\tSYS\tSYS\t0-31\t0\t\tN/A\n"
    "GPU1\tPHB\t X \tSYS\tSYS\t0-31\t0\t\tN/A\n"
    "GPU2\tSYS\tSYS\t X \tPHB\t32-63\t1\t\tN/A\n"
    "GPU3\tSYS\tSYS\tPHB\t X \t32-63\t1\t\tN/A\n"
    "\nLegend:\n\n  X    = Self\n  SYS  = Connection traversing PCIe as well as the SMP interconnect\n"
)
PCIE_P2P = (
    "\t\x1b[4mGPU0\tGPU1\tGPU2\tGPU3\x1b[0m\n"
    " GPU0\tX\tCNS\tCNS\tCNS\n GPU1\tCNS\tX\tCNS\tCNS\n GPU2\tCNS\tCNS\tX\tCNS\n GPU3\tCNS\tCNS\tCNS\tX\n"
    "\nLegend:\n\n  X    = Self\n  OK   = Status Ok\n  CNS  = Chipset not supported\n"
)
NVLINK_MATRIX = (
    "\t\x1b[4mGPU0\tGPU1\tCPU Affinity\tNUMA Affinity\x1b[0m\n"
    "GPU0\t X \tNV12\t0-47\t0\nGPU1\tNV12\t X \t0-47\t0\n"
)
NVLINK_P2P = "\tGPU0\tGPU1\nGPU0\tX\tOK\nGPU1\tOK\tX\n"
CARDS = ("0, 1, 4, 8, 16, 170.00, 170.00, 595.71.05\n1, 1, 4, 8, 16, 170.00, 170.00, 595.71.05\n"
         "2, 1, 4, 4, 16, 170.00, 170.00, 595.71.05\n3, 1, 4, 4, 16, 150.00, 170.00, 595.71.05\n")


class Smi:
    """subprocess.run for nvidia-smi: one answer per kind of question."""

    def __init__(self, topo, p2p, cards, names="NVIDIA GeForce RTX 3060, 8.6, 12288\n" * 4):
        self.answers = {"topo-m": topo, "topo-p2p": p2p, "query-cards": cards, "query-names": names}

    def __call__(self, command, **kwargs):
        if command[:3] == ["nvidia-smi", "topo", "-m"]:
            key = "topo-m"
        elif command[:3] == ["nvidia-smi", "topo", "-p2p"]:
            key = "topo-p2p"
        elif "pcie.link.gen.current" in " ".join(command):
            key = "query-cards"
        else:
            key = "query-names"
        return types.SimpleNamespace(stdout=self.answers[key], returncode=0)


def test_pcie_cards_without_peer_to_peer_are_recorded_as_they_are(monkeypatch):
    monkeypatch.setattr(subprocess, "run", Smi(PCIE_MATRIX, PCIE_P2P, CARDS))
    found = topology.read([0, 1, 2, 3])
    assert found["interconnect"] == "pcie" and found["p2p"] is False
    assert found["gpu_paths"] == {"0-1": "PHB", "0-2": "SYS", "0-3": "SYS", "1-2": "SYS", "1-3": "SYS", "2-3": "PHB"}
    assert [(link["width"], link["width_max"], link["gen_max"]) for link in found["card_links"]] == \
        [(8, 16, 4), (8, 16, 4), (4, 16, 4), (4, 16, 4)]
    assert found["card_links"][3]["power_w"] == 150 and found["driver_version"] == "595.71.05"


def test_nvlink_with_peer_to_peer_is_nvlink(monkeypatch):
    monkeypatch.setattr(subprocess, "run", Smi(NVLINK_MATRIX, NVLINK_P2P, CARDS))
    found = topology.read([0, 1])
    assert found["interconnect"] == "nvlink" and found["p2p"] is True and found["gpu_paths"] == {"0-1": "NV12"}


def test_one_card_has_no_paths_and_an_unanswered_matrix_is_none_not_a_guess(monkeypatch):
    monkeypatch.setattr(subprocess, "run", Smi("", "", CARDS))
    assert topology.read([0])["interconnect"] is None
    found = topology.read([0, 1])
    assert found["interconnect"] is None and found["gpu_paths"] is None and found["p2p"] is None


def test_the_run_records_and_says_what_its_cards_sit_on(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", Smi(PCIE_MATRIX, PCIE_P2P, CARDS))
    monkeypatch.setattr(request, "open", lambda path: iter(["MemTotal: 32000000 kB\n"]), raising=False)
    monkeypatch.setattr(topology, "cpu_model", lambda: "AMD EPYC 7402 24-Core Processor")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv(request.BANDWIDTH_ENV, "360")
    trace = tmp_path / "trace.jsonl"
    trace.write_text('{"prompt": "hi", "output_tokens": 4}\n')
    hw = request.detect_hardware(request.InferOptRequest.model_construct(
        model="Qwen/Qwen3-8B", trace=str(trace), override_memory_bandwidth_gb_s=None))
    assert hw.gpu_count == 4 and hw.interconnect == "pcie" and hw.p2p is False
    assert hw.cpu_model.startswith("AMD EPYC") and hw.model_dump()["gpu_paths"]["2-3"] == "PHB"
    assert topology.describe(hw) == (
        "cards     4 x NVIDIA GeForce RTX 3060, over pcie (PHB, SYS), peer to peer off, "
        "PCIe x4 of x16, x8 of x16 Gen4, 150 W of 170 W, 170 W of 170 W, driver 595.71.05, "
        "AMD EPYC 7402 24-Core Processor")


class Lscpu:
    def __call__(self, command, **kwargs):
        assert command == ["lscpu"]
        return types.SimpleNamespace(stdout="Architecture:  aarch64\nVendor ID:  ARM\nModel name:  Cortex-X925\n"
                                            "Model name:  Cortex-A725\n", returncode=0)


def test_an_arm_cpu_is_named_by_its_cores_not_its_implementer_code(monkeypatch, tmp_path):
    """4 Oct 2026, on the DGX: /proc/cpuinfo named only "0x41"."""
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text("processor\t: 0\nCPU implementer\t: 0x41\nCPU part\t: 0xd85\n")
    real_open = open
    monkeypatch.setattr("builtins.open", lambda path, *a, **k: real_open(cpuinfo if path == "/proc/cpuinfo" else path, *a, **k))
    monkeypatch.setattr(subprocess, "run", Lscpu())
    assert topology.cpu_model() == "Cortex-X925 + Cortex-A725"


def test_a_unified_parts_placeholder_pcie_link_stays_out_of_the_line():
    hw = types.SimpleNamespace(gpu_count=1, gpu_name="NVIDIA GB10", unified_memory=True, interconnect=None,
                               gpu_paths=None, p2p=None, driver_version="580.173.02", cpu_model="Cortex-X925",
                               card_links=[{"index": 0, "gen": 1, "gen_max": 1, "width": 1, "width_max": 16}])
    assert topology.describe(hw) == "cards     1 x NVIDIA GB10, driver 580.173.02, Cortex-X925"
