"""GPUs AIConfigurator has no measured database for, as estimate-only systems.

AIConfigurator predicts from kernels it measured on a handful of systems. Every
other card, the GB10 we own and the consumer and workstation cards a market
rents, gets a system file here: its datasheet figures, and the measured system
of the same generation whose efficiency it borrows (the donor).

    card at speed-of-light (SOL, its own datasheet)
      x  donor's measured / SOL throughput for the SAME model and traffic
      =  a LOWER estimate than the roofline, per job

The files are written by tools/write_aic_systems.py into aic_systems/ and loaded
beside AIConfigurator's own, without touching its package.

TENSOR RATES ARE DENSE, AT THE ACCUMULATE PRECISION vLLM USES. Its bf16 matmuls
accumulate in FP32, and GeForce cards run FP32 accumulate at half their FP16
accumulate rate: an RTX 4090 is 165.2 TFLOPS for this, not the 330.3 on the box.
Where a card's sheet lists only "AI TOPS" (FP4 with sparsity), bf16 is that over
16 on GeForce and over 8 on workstation parts, the ratio every listed card obeys.
Memory variants take the smaller part, so an estimate is never the better one.
"""

import os
import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import yaml

systems_dir = Path(__file__).parent / "aic_systems"

# The measured system each generation borrows its efficiency from.
DONOR_BY_SM = {80: "a100_sxm", 86: "a100_sxm", 89: "l40s", 90: "h100_sxm",
               120: "rtx_pro_6000_server", 121: "rtx_pro_6000_server"}


@dataclass(frozen=True)
class Card:
    system: str
    names: tuple        # plain-name tokens that identify it, see plain()
    sm_version: int
    memory_gb: float
    bandwidth_gb_s: float
    bf16_tflops: float  # dense, FP32 accumulate
    fp8_tflops: float   # dense, 0 where the card has no FP8 tensor path
    power_w: int
    pcie_gen: int
    source: str


CARDS = (
    # Ampere GA10x: SMs x 1024 FP16 FLOP/clk x boost = FP16-accumulate dense;
    # GeForce halves it for FP32 accumulate, workstation parts do not.
    Card("rtx_3060", ("rtx3060",), 86, 12, 360, 25.5, 0, 170, 4, "28 SM x 1.777 GHz, GeForce half rate"),
    Card("rtx_3060_ti", ("rtx3060ti",), 86, 8, 448, 32.4, 0, 200, 4, "38 SM x 1.665 GHz, GeForce half rate"),
    Card("rtx_3070", ("rtx3070",), 86, 8, 448, 40.6, 0, 220, 4, "46 SM x 1.725 GHz, GeForce half rate"),
    Card("rtx_3070_ti", ("rtx3070ti",), 86, 8, 608, 43.5, 0, 290, 4, "48 SM x 1.77 GHz, GeForce half rate"),
    Card("rtx_3080", ("rtx3080",), 86, 10, 760, 59.5, 0, 320, 4, "GA102 whitepaper, 10 GB part"),
    Card("rtx_3080_ti", ("rtx3080ti",), 86, 12, 912, 68.2, 0, 350, 4, "80 SM x 1.665 GHz, GeForce half rate"),
    Card("rtx_3090", ("rtx3090",), 86, 24, 936, 71.2, 0, 350, 4, "GA102 whitepaper"),
    Card("rtx_3090_ti", ("rtx3090ti",), 86, 24, 1008, 80.0, 0, 450, 4, "84 SM x 1.86 GHz, GeForce half rate"),
    Card("rtx_a4000", ("rtxa4000", "a4000"), 86, 16, 448, 76.7, 0, 140, 4, "datasheet 153.4 sparse"),
    Card("rtx_a4500", ("rtxa4500", "a4500"), 86, 20, 640, 94.6, 0, 200, 4, "datasheet 189.2 sparse"),
    Card("rtx_a5000", ("rtxa5000", "a5000"), 86, 24, 768, 111.1, 0, 230, 4, "datasheet 222.2 sparse"),
    Card("rtx_a6000", ("rtxa6000", "a6000"), 86, 48, 768, 154.8, 0, 300, 4, "datasheet 309.7 sparse"),
    Card("a10", ("a10",), 86, 24, 600, 125.0, 0, 150, 4, "datasheet BF16 125 dense"),
    Card("a40", ("a40",), 86, 48, 696, 149.7, 0, 300, 4, "datasheet BF16 149.7 dense"),
    # Ada: SMs x 1024 x boost, halved for FP32 accumulate on GeForce and on L40.
    Card("rtx_4060", ("rtx4060",), 89, 8, 272, 30.2, 60.4, 115, 4, "24 SM x 2.46 GHz, GeForce half rate"),
    Card("rtx_4060_ti", ("rtx4060ti",), 89, 8, 288, 44.1, 88.3, 160, 4, "34 SM x 2.535 GHz, GeForce half rate"),
    Card("rtx_4070", ("rtx4070",), 89, 12, 504, 58.3, 116.6, 200, 4, "46 SM x 2.475 GHz, GeForce half rate"),
    Card("rtx_4070_super", ("rtx4070super", "rtx4070s"), 89, 12, 504, 71.0, 141.9, 220, 4, "56 SM x 2.475 GHz"),
    Card("rtx_4070_ti", ("rtx4070ti",), 89, 12, 504, 80.2, 160.4, 285, 4, "60 SM x 2.61 GHz, GeForce half rate"),
    Card("rtx_4070_ti_super", ("rtx4070tisuper", "rtx4070sti"), 89, 16, 672, 88.2, 176.4, 285, 4, "66 SM x 2.61 GHz"),
    Card("rtx_4080", ("rtx4080",), 89, 16, 717, 97.5, 194.9, 320, 4, "AD103 whitepaper"),
    Card("rtx_4080_super", ("rtx4080super", "rtx4080s"), 89, 16, 736, 104.4, 208.9, 320, 4, "80 SM x 2.55 GHz"),
    Card("rtx_4090", ("rtx4090",), 89, 24, 1008, 165.2, 330.3, 450, 4, "AD102 whitepaper"),
    Card("rtx_6000_ada", ("rtx6000ada",), 89, 48, 960, 364.2, 728.5, 300, 4, "datasheet 728.5 sparse"),
    Card("l40", ("l40",), 89, 48, 864, 181.0, 362.1, 300, 4, "datasheet FP16 181.05 dense"),
    # Hopper parts AIConfigurator has not measured.
    Card("h100_nvl", ("h100nvl",), 90, 94, 3900, 835.5, 1671.0, 400, 5, "datasheet 1671 sparse"),
    Card("gh200", ("gh200",), 90, 96, 4000, 989.0, 1979.0, 700, 5, "H100 die, 96 GB HBM3 part"),
    # Blackwell with SM 12.x. AI TOPS is FP4 with sparsity: bf16 is /16 on
    # GeForce, /8 on workstation parts.
    Card("rtx_5060", ("rtx5060",), 120, 8, 448, 38.4, 76.8, 145, 5, "614 AI TOPS"),
    Card("rtx_5060_ti", ("rtx5060ti",), 120, 8, 448, 47.4, 94.9, 180, 5, "759 AI TOPS"),
    Card("rtx_5070", ("rtx5070",), 120, 12, 672, 61.8, 123.5, 250, 5, "988 AI TOPS"),
    Card("rtx_5070_ti", ("rtx5070ti",), 120, 16, 896, 87.9, 175.8, 300, 5, "1406 AI TOPS"),
    Card("rtx_5080", ("rtx5080",), 120, 16, 960, 112.6, 225.1, 360, 5, "1801 AI TOPS"),
    Card("rtx_5090", ("rtx5090",), 120, 32, 1792, 209.5, 419.0, 575, 5, "3352 AI TOPS"),
    Card("rtx_pro_4500", ("rtxpro4500", "pro4500"), 120, 32, 896, 210.9, 421.8, 200, 5, "1687 AI TOPS"),
    Card("rtx_pro_5000", ("rtxpro5000", "pro5000"), 120, 48, 1344, 258.0, 516.0, 300, 5, "2064 AI TOPS"),
    # GB10: 1000 AI TOPS. Taken at the GeForce ratio, 62.5 bf16, not the 125 a
    # workstation ratio gives, because an estimate here has to err low.
    Card("gb10", ("gb10", "dgxspark"), 121, 128, 273, 62.5, 125.0, 140, 5,
         "1000 AI TOPS at the GeForce ratio; LPDDR5X 273 GB/s unified"),
)

# Names that are AIConfigurator's own systems under another spelling.
SHIPPED_ALIASES = {
    "h100sxm": "h100_sxm", "h100hbm3": "h100_sxm", "h100": "h100_sxm", "h800": "h100_sxm",
    "h100pcie": "h100_pcie", "h200": "h200_sxm", "a100pcie": "a100_pcie", "a100": "a100_sxm",
    "a800": "a100_sxm", "a30": "a30", "l4": "l4", "l40s": "l40s", "b200": "b200_sxm",
    "b300": "b300_sxm", "gb200": "gb200", "gb300": "gb300", "rtxpro6000": "rtx_pro_6000_server",
    "pro6000": "rtx_pro_6000_server",
}

_TOKENS = sorted([(token, card.system) for card in CARDS for token in card.names]
                 + list(SHIPPED_ALIASES.items()), key=lambda pair: len(pair[0]), reverse=True)


def plain(name):
    """'NVIDIA GeForce RTX 3080 Ti' -> 'rtx3080ti': no vendor, no memory size,
    letters and digits only."""
    text = re.sub(r"\d+\s*gb\b", "", (name or "").lower())
    text = text.replace("nvidia", "").replace("geforce", "").replace("_", "")
    return re.sub(r"[^a-z0-9]", "", text)


def shipped_systems_dir():
    """AIConfigurator's own systems, or None where it is not installed (a
    rented host runs searches, never predictions)."""
    try:
        return Path(str(resources.files("aiconfigurator_core") / "systems"))
    except ModuleNotFoundError:
        return None


def spec_path(system):
    """The system file, ours or AIConfigurator's, or None."""
    for directory in (systems_dir, shipped_systems_dir()):
        if directory is not None and (directory / f"{system}.yaml").is_file():
            return directory / f"{system}.yaml"
    return None


def measured(system):
    """Whether AIConfigurator ships kernel measurements for this system."""
    shipped = shipped_systems_dir()
    return shipped is not None and (shipped / "data" / system / "gemm").is_dir()


def system_for(gpu_name):
    """(system, donor) for a card's name, or None. donor is "" when the system
    has its own measurements; otherwise it is the measured system of the same
    generation whose efficiency the estimate borrows."""
    name = plain(gpu_name)
    system = next((target for token, target in _TOKENS if token in name), None)
    if system is None or spec_path(system) is None:
        return None
    if measured(system):
        return system, ""
    return system, DONOR_BY_SM.get(card_spec(system)["sm_version"], "")


def card_spec(system):
    """The figures a roofline needs, read from the system file."""
    path = spec_path(system)
    gpu = yaml.safe_load(path.read_text())["gpu"]
    return {
        "system": system,
        "bandwidth_gb_s": gpu["mem_bw"] / 1e9,
        "bf16_tflops": gpu["bfloat16_tc_flops"] / 1e12,
        "fp8_tflops": gpu.get("fp8_tc_flops", 0) / 1e12,
        "memory_gb": gpu["mem_capacity"] / 2 ** 30,
        "sm_version": int(gpu.get("sm_version") or 0),
    }


def system_yaml(card):
    """The AIConfigurator system file for one card. data_dir names a directory
    that does not exist, which is what makes it estimate-only."""
    pcie_bw = 32_000_000_000 if card.pcie_gen == 4 else 64_000_000_000
    gpu = {
        "mem_bw": int(card.bandwidth_gb_s * 1e9),
        "mem_bw_empirical_scaling_factor": 0.8,
        "mem_empirical_constant_latency": 0.000003,
        "mem_capacity": int(card.memory_gb * 2 ** 30),
        "bfloat16_tc_flops": int(card.bf16_tflops * 1e12),
        "int8_tc_flops": int(card.bf16_tflops * 2e12),
    }
    if card.fp8_tflops:
        gpu["fp8_tc_flops"] = int(card.fp8_tflops * 1e12)
    gpu.update(power=card.power_w, sm_version=card.sm_version)
    document = {
        "data_dir": f"data/{card.system}",
        "gpu": gpu,
        "node": {"num_gpus_per_node": 8, "inter_node_bw": 25_000_000_000, "intra_node_bw": pcie_bw,
                 "pcie_bw": pcie_bw, "p2p_latency": 0.00001},
        "misc": {"nccl_mem": {1: 0, 2: 358612992, 4: 411041792, 8: 411041792},
                 "other_mem": 3758096384, "nccl_version": "2.27.3"},
    }
    header = (f"# Estimate-only system for {card.system}, written by tools/write_aic_systems.py.\n"
              f"# Dense bf16 {card.bf16_tflops} TFLOPS at FP32 accumulate ({card.source}),\n"
              f"# {card.bandwidth_gb_s} GB/s, {card.memory_gb} GB. Efficiency borrowed from "
              f"{DONOR_BY_SM[card.sm_version]}.\n")
    return header + yaml.safe_dump(document, sort_keys=False)


def write_systems(directory=systems_dir):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for card in CARDS:
        (directory / f"{card.system}.yaml").write_text(system_yaml(card))
    return [card.system for card in CARDS]


def register_with_aiconfigurator():
    """Put our systems beside AIConfigurator's own, once per process."""
    from aiconfigurator.sdk.perf_database import get_systems_paths, set_systems_paths

    paths = get_systems_paths()
    if str(systems_dir) not in paths:
        set_systems_paths([*paths, str(systems_dir)])
        os.environ["AICONFIGURATOR_SYSTEMS_PATH"] = os.pathsep.join([*paths, str(systems_dir)])
