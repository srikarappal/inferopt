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

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from fingerprint import SLO, Fingerprint

# NOT a venv. A venv bakes absolute paths into its shebang, its python symlink
# and pyvenv.cfg, so one created on the host is broken inside a container that
# mounts the same files at a different path -- which is exactly what happened.
# `pip install --target` produces a plain directory with no absolute paths, and
# PYTHONPATH is computed relative to this file, so it works from either side.
AIC_PKGS = Path(__file__).resolve().parent / ".aic-pkgs"

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
    raw: str = ""


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


def _run_aic(fp: Fingerprint, slo: SLO, system: str, backend: str = "vllm") -> str:
    if not (AIC_PKGS / "aiconfigurator").is_dir():
        raise RuntimeError(
            f"{AIC_PKGS} not found. AIConfigurator pins numpy~=1.26.4, which can never "
            f"coexist with vLLM's <2.4, so it is installed beside the code and imported "
            f"only in a subprocess:\n"
            f"    pip install --target .aic-pkgs aiconfigurator 'plotext<6'")
    # Run with THIS interpreter plus the target dir on PYTHONPATH: the subprocess
    # gets numpy 1.26.4, the parent keeps its own. No shebangs, no symlinks.
    env = {**os.environ, "PYTHONPATH": str(AIC_PKGS)}
    # aiconfigurator fetches config.json with urllib, which reads the SYSTEM CA
    # store -- and inside a container that often is not there, while
    # huggingface_hub works because it uses certifi. Point urllib at certifi too.
    try:
        import certifi
        env.setdefault("SSL_CERT_FILE", certifi.where())
        env.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
    except ImportError:
        pass

    # Better still: give it a LOCAL directory. We already resolved config.json
    # to build the fingerprint, so the subprocess needs no network at all --
    # which removes a whole class of failure rather than working around it.
    model_arg = fp.model.id
    local = _local_config_dir(fp.model.id)
    if local:
        model_arg = str(local)

    cmd = [sys.executable, "-m", "aiconfigurator.main",
           "cli", "default", "--model", model_arg, "--system", system,
           "--backend", backend, "--isl", str(int(fp.workload.mean_input_tokens)),
           "--osl", str(int(fp.workload.mean_output_tokens)),
           "--total-gpus", str(fp.hw.gpu_count), "--no-color", "--top-n", "5"]
    if slo.ttft_p99_ms:
        cmd += ["--ttft", str(slo.ttft_p99_ms)]
    if slo.itl_p99_ms:
        # The proxy is a faster machine; filtering on the real SLO would discard
        # configs that rank well and only fail because of the hardware gap the
        # roofline already accounts for.
        cmd += ["--tpot", str(max(slo.itl_p99_ms, 30.0))]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=env)
    if not r.stdout.strip():
        raise RuntimeError(f"aiconfigurator produced no output:\n{r.stderr[-800:]}")
    return r.stdout


_ROW = re.compile(
    r"\|\s*1\s*\|\s*(\w+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|"
    r"\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*(\d+)[^|]*\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|"
    r"\s*tp(\d+)pp(\d+)\s*\|\s*(\d+)\s*\|")


def _parse(out: str) -> dict | None:
    m = _ROW.search(out)
    if not m:
        return None
    b, tps_gpu, tps_user, reqs, ttft, lat, conc, tp, pp, bs = m.groups()
    return {"backend": b, "tokens_s_gpu": float(tps_gpu), "tokens_s_user": float(tps_user),
            "req_s": float(reqs), "ttft_ms": float(ttft), "request_latency_ms": float(lat),
            "concurrency": int(conc), "tp": int(tp), "pp": int(pp), "batch_size": int(bs)}


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

    out = _run_aic(fp, slo, system)
    top = _parse(out)
    if not top:
        return Prediction(system_used=system, is_proxy=is_proxy, proxy_note=note,
                          feasible=feasible, infeasible_reason=reason,
                          remedies=remedies, raw=out[-1500:])

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
                      raw=out[-1500:])


def describe(p: Prediction, log=print) -> None:
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
