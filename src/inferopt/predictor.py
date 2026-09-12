"""Stage 1.2 -- predict a good config before touching a GPU.

    pred = predict(fingerprint, slo)
    pred.seed_config     -> what to hand the DAG
    pred.feasible        -> False when the SLO is unreachable on this hardware

AIConfigurator has no calibrated database for every GPU, so an unsupported part
is mapped to the nearest supported member of its architecture family and the
result is corrected. Two rules make that honest:

  RANK on the proxy      config rankings survive a monotone rescaling -- a config
                         that batches better or preempts less wins on both parts,
                         for the same reason. So the proxy picks the SHAPE.

  SCALE with a roofline  absolute numbers do not survive. GB10 has ~273 GB/s
                         against B200's ~8000, a 29x gap, so a proxy's tokens/s
                         is meaningless as a forecast. The floor comes from
                         physics instead: a decode step must read every weight,
                         so ITL >= weight_bytes / bandwidth, whatever any
                         database says.

That second rule is the valuable half. It answers "is this SLO even reachable on
this hardware" in milliseconds, which no amount of measurement can do faster.

HISTORY -- getting a predictor to run at all

  GB10 has no calibrated kernel database. AIConfigurator supports h100_sxm,
  h200_sxm, b200_sxm, gb200 and a100_sxm. The workaround is to rank on the
  nearest family member and rescale with a roofline: config RANKINGS survive a
  monotone rescaling, absolute numbers do not (GB10's ~273 GB/s against B200's
  ~8000 is a 29x gap). The roofline half is the valuable half -- it answers "is
  this SLO reachable at all" in milliseconds, and it was validated against
  measurement: Qwen3.5-9B's 66ms floor against 74-197ms measured, never below.

  numpy 1.26.4 against vLLM's 2.3.5. AIConfigurator can never share the serving
  environment. This is the origin of the --target + PYTHONPATH subprocess
  pattern that quantize.py later reused.

  A venv could not be relocated. Bind-mounting an environment at a different
  path breaks console-script shebangs, the bin/python symlink, and pyvenv.cfg.
  `pip install --target` plus PYTHONPATH has no absolute paths to break.

  SSL CERTIFICATE_VERIFY_FAILED. aiconfigurator uses urllib, which reads the
  system CA store rather than certifi's. Fixed with certifi env vars AND a local
  config directory, which removes the network call entirely -- verified against
  a blocked proxy.
"""

from __future__ import annotations

import importlib.metadata
import json
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from inferopt.fingerprint import SLO, Fingerprint

# Unsupported GPU -> nearest supported member of the same architecture family.
# Same tensor-core generation and kernel shapes, so the RANKING transfers; the
# memory system does not, which is what the roofline correction is for.
PROXY_SYSTEM = {
    "NVIDIA GB10": ("b200_sxm", "same Blackwell tensor cores; GB10's unified LPDDR5X "
                                "is ~29x slower than B200 HBM3e, so only the ranking transfers"),
    "NVIDIA GH200": ("h200_sxm", "same Hopper generation"),
    "NVIDIA RTX PRO 6000": ("b200_sxm", "same Blackwell generation"),
}
NATIVE_SYSTEMS = {"h100_sxm", "h200_sxm", "b200_sxm", "gb200", "a100_sxm",
                  "h100_pcie", "a100_pcie", "l4", "a30"}


@dataclass
class Prediction:
    system_used: str
    is_proxy: bool
    proxy_note: str = ""
    seed_config: dict = field(default_factory=dict)
    predicted: dict = field(default_factory=dict)     # on the proxy, uncorrected
    corrected: dict = field(default_factory=dict)     # scaled to the real hardware
    feasible: bool = True
    infeasible_reason: str = ""
    remedies: list[str] = field(default_factory=list)
    frontier: list[dict] = field(default_factory=list)
    """Every config aiconfigurator returned, not only the best.

    The CLI path asked for five and read one. These cost nothing extra and are a
    predicted frontier to set beside the measured one."""


def roofline_itl_ms(fp: Fingerprint, weight_gb: float | None = None) -> float:
    """Minimum inter-token latency: one decode step reads every active weight.

    A hard floor. No batching, scheduler or kernel choice moves it -- batching
    raises THROUGHPUT by amortising the same read across more sequences, but the
    per-token latency each user sees is bounded by this.
    """
    w = weight_gb if weight_gb is not None else fp.model.active_weight_gb
    return w / fp.hw.memory_bandwidth_gb_s * 1000.0


def _feasibility(fp: Fingerprint, slo: SLO) -> tuple[bool, str, list[str]]:
    if not slo.itl_p99_ms:
        return True, "", []
    floor = roofline_itl_ms(fp)
    if floor <= slo.itl_p99_ms:
        return True, "", []

    reason = (f"ITL floor is {floor:.0f}ms on this hardware but the SLO asks for "
              f"{slo.itl_p99_ms:.0f}ms. A decode step must read all "
              f"{fp.model.active_weight_gb:.1f}GB of weights at "
              f"{fp.hw.memory_bandwidth_gb_s:.0f} GB/s, so no serving configuration "
              f"can reach it -- this is arithmetic, not tuning.")
    rem = []
    for label, factor in (("FP8", 0.5), ("INT4/AWQ", 0.25)):
        got = roofline_itl_ms(fp, fp.model.active_weight_gb * factor)
        mark = "meets" if got <= slo.itl_p99_ms else "still misses"
        rem.append(f"{label} weights -> ITL floor {got:.0f}ms ({mark} the SLO)")
    smaller = fp.model.active_weight_gb * slo.itl_p99_ms / floor
    rem.append(f"or a model under ~{smaller:.0f}GB of active weights "
               f"(~{smaller/2:.0f}B params at bf16)")
    rem.append(f"or relax itl_p99 to >= {floor:.0f}ms")
    return False, reason, rem


def _local_config_dir(model_id: str) -> Path | None:
    """A directory holding just this model's config.json, from the local cache.

    aiconfigurator only needs config.json to derive layers/heads/dims, so
    handing it a local path avoids the download entirely.
    """
    try:
        src = Path(model_id) / "config.json"
        if not src.exists():
            from huggingface_hub import hf_hub_download
            src = Path(hf_hub_download(model_id, "config.json"))
        d = Path(tempfile.mkdtemp(prefix="inferopt-cfg-"))
        shutil.copy(src, d / "config.json")
        return d
    except Exception:
        return None


def environment_warning() -> str:
    """Non-empty when installing the predictor has broken the server.

    `pip install aiconfigurator` in a vLLM environment resolves numpy~=1.26.4
    and downgrades it. vLLM and torch then fail at import or at launch, with an
    error that says nothing about aiconfigurator, and the person debugging it
    has no reason to connect the two. This turns that into one sentence at the
    point of use.

    Checked rather than assumed: a serving environment is one where vLLM is
    importable, so the pairing that matters is numpy 1.x beside vLLM.
    """
    try:
        numpy_version = importlib.metadata.version("numpy")
    except importlib.metadata.PackageNotFoundError:
        return ""
    if not numpy_version.startswith("1."):
        return ""
    try:
        importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        return ""
    return (f"numpy is {numpy_version} in an environment that also has vLLM. "
            f"Installing aiconfigurator without --no-deps downgrades numpy to "
            f"satisfy its numpy~=1.26.4 pin, which breaks torch and vLLM. Repair "
            f"with: pip install --no-deps -U 'numpy>=2' , then reinstall "
            f"aiconfigurator with --no-deps, or run ./setup.sh which does both.")


def _frontier(fp: Fingerprint, slo: SLO, system: str) -> tuple[dict, list[dict]]:
    """Call aiconfigurator and return (top config, the whole frontier).

    THE PYTHON API, NOT THE CLI. This used to shell out to
    `python -m aiconfigurator.main cli default` in a subprocess and parse row 1
    out of the ASCII table with a regex over column positions. Two things were
    wrong with that. The regex broke silently on any layout change, with
    `_parse` returning None as the only signal, indistinguishable from an
    unsupported model. And it threw away the other four configs: --top-n 5 was
    already being paid for and only row 1 was read.

    The subprocess existed for a real reason that has since been removed.
    aiconfigurator pins numpy~=1.26.4 against the serving environment's 2.3.5,
    so it could not share the process. The pin turns out to be conservative:
    forced to numpy 2.3.5 it returns byte-identical numbers, verified on both.
    The one genuine incompatibility was plotext, which needs <6 and which
    nothing else in the serving environment required. So it is installed there
    with --no-deps and called in-process.

    tpot is floored at 30ms deliberately. The proxy is a faster machine, and
    filtering on the real ITL target would discard configs that rank well and
    fail only on the hardware gap the roofline correction already accounts for.
    """
    from aiconfigurator.cli import cli_default

    model_arg = str(_local_config_dir(fp.model.id) or fp.model.id)
    # top_n is explicit although it already defaults to 5: the CLI path passed
    # it and the frontier this returns is now used, so it is a choice rather
    # than an inherited default. Note that the SLO filters below cut it further,
    # so one row on a single GPU is a filtered frontier, not a truncated one.
    kwargs = {"model_path": model_arg, "total_gpus": fp.hw.gpu_count,
              "system": system, "isl": int(fp.workload.mean_input_tokens),
              "osl": int(fp.workload.mean_output_tokens), "top_n": 5}
    if slo.ttft_p99_ms:
        kwargs["ttft"] = slo.ttft_p99_ms
    if slo.itl_p99_ms:
        kwargs["tpot"] = max(slo.itl_p99_ms, 30.0)

    result = cli_default(**kwargs)
    frame = (result.best_configs or {}).get("agg")
    if frame is None or not len(frame):
        return {}, []

    rows = [_row(frame.iloc[i]) for i in range(len(frame))]
    return rows[0], rows


def _row(row) -> dict:
    """One frontier entry, in the names the rest of this module uses.

    Named lookups rather than column positions: a renamed column raises here
    instead of quietly shifting every field one to the left, which is what a
    positional regex over a printed table does.
    """
    def num(*names, default=0.0):
        for n in names:
            if n in row.index:
                try:
                    return float(row[n])
                except (TypeError, ValueError):
                    pass
        return default

    return {
        "backend": str(row.get("backend", "")),
        "tokens_s_gpu": num("tokens/s/gpu", "tokens/s"),
        "tokens_s_user": num("tokens/s/user"),
        "req_s": num("seq/s/gpu", "request_rate"),
        "ttft_ms": num("ttft"),
        "tpot_ms": num("tpot"),
        "request_latency_ms": num("ttft") + num("tpot") * 0.0,
        "concurrency": int(num("concurrency")),
        "tp": int(num("tp", default=1)),
        "pp": int(num("pp", default=1)),
        "batch_size": int(num("bs", "global_bs", default=1)),
        "memory_gb": num("memory"),
        "power_w": num("power_w"),
    }


def predict(fp: Fingerprint, slo: SLO, *, log=print) -> Prediction:
    feasible, reason, remedies = _feasibility(fp, slo)

    system, is_proxy, note = fp.hw.gpu_name, False, ""
    guess = fp.hw.gpu_name.lower().replace("nvidia ", "").replace(" ", "_")
    if guess in NATIVE_SYSTEMS:
        system = guess
    elif fp.hw.gpu_name in PROXY_SYSTEM:
        system, note = PROXY_SYSTEM[fp.hw.gpu_name]
        is_proxy = True
    else:
        return Prediction(system_used="none", is_proxy=False,
                          proxy_note=f"no proxy on record for {fp.hw.gpu_name!r}",
                          feasible=feasible, infeasible_reason=reason, remedies=remedies)

    top, rows = _frontier(fp, slo, system)
    if not top:
        return Prediction(system_used=system, is_proxy=is_proxy, proxy_note=note,
                          feasible=feasible, infeasible_reason=reason,
                          remedies=remedies)

    # Correct to the real hardware. Decode is memory-bound, so scale by the
    # bandwidth ratio, then floor at the roofline -- the proxy cannot predict a
    # latency faster than physics allows on the target.
    proxy_bw = {"b200_sxm": 8000.0, "gb200": 8000.0, "h200_sxm": 4800.0,
                "h100_sxm": 3350.0, "a100_sxm": 2039.0}.get(system, fp.hw.memory_bandwidth_gb_s)
    ratio = fp.hw.memory_bandwidth_gb_s / proxy_bw
    floor = roofline_itl_ms(fp)
    corrected = {
        "tokens_s_gpu": round(top["tokens_s_gpu"] * ratio, 1),
        "itl_ms": round(max(1000.0 / top["tokens_s_user"] if top["tokens_s_user"] else floor,
                            floor), 1),
        "ttft_ms": round(top["ttft_ms"] / ratio, 1),
        "bandwidth_ratio": round(ratio, 4),
        "roofline_itl_ms": round(floor, 1),
    }

    seed = {"max_num_seqs": top["batch_size"], "tensor_parallel_size": top["tp"]}
    if top["pp"] > 1:
        seed["pipeline_parallel_size"] = top["pp"]
    return Prediction(system_used=system, is_proxy=is_proxy, proxy_note=note,
                      seed_config=seed, predicted=top, corrected=corrected,
                      feasible=feasible, infeasible_reason=reason, remedies=remedies,
                      frontier=rows)


def describe(p: Prediction, log=print) -> None:
    warning = environment_warning()
    if warning:
        log(f"  stage 1.2 WARNING: {warning}")
    if p.is_proxy:
        log(f"  stage 1.2 predicted on PROXY system {p.system_used}")
        log(f"            {p.proxy_note}")
    else:
        log(f"  stage 1.2 predicted on {p.system_used}")
    if p.predicted:
        log(f"    on proxy  bs={p.predicted['batch_size']} tp{p.predicted['tp']}pp{p.predicted['pp']}  "
            f"{p.predicted['tokens_s_gpu']:,.0f} tok/s  ttft {p.predicted['ttft_ms']:.0f}ms")
        log(f"    corrected {p.corrected['tokens_s_gpu']:,.0f} tok/s  "
            f"itl >= {p.corrected['roofline_itl_ms']:.0f}ms  "
            f"(bandwidth ratio {p.corrected['bandwidth_ratio']:.3f})")
        log(f"    -> ranking is trustworthy, absolute numbers are not; "
            f"stage 1.3 measures the truth")
    if not p.feasible:
        log(f"\n  INFEASIBLE: {p.infeasible_reason}")
        for r in p.remedies:
            log(f"    - {r}")
