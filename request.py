"""What a user actually hands inferopt, and how it becomes a Fingerprint.

    inferopt optimize \\
        --model Qwen/Qwen3.5-9B \\
        --trace prod_traffic.jsonl \\
        --ttft-p99 500 --itl-p99 30 \\
        --allow-loss 0.01

The user supplies six things at most. Everything else in the fingerprint --
head counts, KV bytes per token, compute capability, length distributions --
is DETECTED, because a field a human types is a field a human can get wrong,
and a wrong fingerprint optimizes for a machine or a workload that isn't there.

Detection sources:
    model     config.json of the checkpoint
    hardware  nvidia-smi, /proc/meminfo, os.cpu_count
    workload  the trace file, via WorkloadFingerprint.from_trace
    lora      adapter_config.json of each adapter

HISTORY -- detection bugs, all of which produced believable wrong answers

  A swallowed exception reported 16.45B for a 14.8B model. Parameter count fell
  back to arithmetic over config fields when the checkpoint index could not be
  read. An hf_hub_download out of scope raised NameError, which a bare
  `except Exception: return None` swallowed, and the fallback formula assumed
  MHA -- over-counting every grouped-query model by 11%. Nobody would notice,
  because the number looks reasonable. Now: the index is authoritative, failures
  are PRINTED with their reason, and the fallback is GQA-aware.

  Multimodal configs nest the language model under text_config. Reading the top
  level finds nothing and raises KeyError: num_attention_heads.

  GB10 reports memory.total as [N/A]. It is a unified-memory part: CPU and GPU
  share one pool, so the real budget is system RAM and the host competes for it.

  nvidia-smi does not report memory bandwidth, and it bounds every decode
  estimate. MEMORY_BANDWIDTH_GB_S fails LOUDLY on an unknown part rather than
  defaulting: guessing an HBM figure for a unified-memory part is a ~30x error
  in exactly the direction that makes everything look fine.

  The quantized-checkpoint lookup was deleted, not fixed. It searched the Hub
  for first-party -FP8/-AWQ repos. Even done carefully (same org only, verified
  against the candidate's own quantization_config) it answers the wrong
  question -- see the note in fingerprint.py. detect_quantization_capability
  replaced it.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from huggingface_hub import hf_hub_download

from pydantic import BaseModel, ConfigDict, Field, model_validator

from fingerprint import (
    DTYPE_BYTES,
    SLO,
    Fingerprint,
    HardwareFingerprint,
    LoraFingerprint,
    ModelFingerprint,
    WorkloadFingerprint,
)

# nvidia-smi does not report memory bandwidth, and it is the single number that
# drives every decode estimate. A lookup table with a loud failure beats a
# plausible default: guessing an HBM figure for a unified-memory part is a ~30x
# error in exactly the direction that makes everything look fine.
MEMORY_BANDWIDTH_GB_S = {
    "NVIDIA GB10": 273.0,
    "NVIDIA H100 80GB HBM3": 3350.0,
    "NVIDIA H100 PCIe": 2000.0,
    "NVIDIA H200": 4800.0,
    "NVIDIA A100-SXM4-80GB": 2039.0,
    "NVIDIA A100-SXM4-40GB": 1555.0,
    "NVIDIA B200": 8000.0,
    "NVIDIA L40S": 864.0,
    "NVIDIA L4": 300.0,
    "NVIDIA GeForce RTX 4090": 1008.0,
}

# Unified-memory parts: CPU and GPU share one pool, so "device memory" is system
# memory and the host competes for it.
UNIFIED_MEMORY_PARTS = ("GB10", "GH200", "GB200")


class InferOptRequest(BaseModel):
    """The whole user-facing input surface."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model: str = Field(description="HF id or local path of the model to optimize")
    trace: str = Field(description="JSONL workload trace. Required, and deliberately so -- "
                                   "summary statistics can describe a workload but not reproduce "
                                   "one, and a guessed length distribution optimizes for traffic "
                                   "that does not exist.")

    ttft_p99_ms: float | None = Field(None, description="latency SLO; requests slower than this earn no goodput")
    itl_p99_ms: float | None = Field(None, description="inter-token latency SLO")
    throughput_target_tok_s: float | None = Field(None, description="minimum acceptable throughput, if any")

    allow_loss: float | None = Field(None, ge=0.0, le=1.0,
        description="quality budget for the lossy branch. None means explore it anyway and "
                    "return the frontier for you to pick from.")

    adapters: list[str] = Field(default_factory=list,
        description="LoRA adapter paths. One is merged upstream; several enable the multi-LoRA subtree.")
    eval_set: str | None = Field(None,
        description="custom eval set. Without one, public benchmarks are used at their full size.")
    runtimes: list[str] | None = Field(None,
        description="restrict candidate backends, e.g. ['vllm']. Default: whatever supports the model.")
    budget_minutes: int = Field(180, description="wall-clock ceiling for the whole run")

    # Escape hatches. Detection is right almost always; when it is not, the fix
    # should be a named override rather than editing a detected value by hand.
    override_memory_bandwidth_gb_s: float | None = None
    override_max_model_len: int | None = None

    @model_validator(mode="after")
    def _check(self):
        if not Path(self.trace).exists():
            raise ValueError(f"trace not found: {self.trace}")
        if self.allow_loss == 0.0:
            raise ValueError(
                "allow_loss=0 means the lossless branch only -- omit the flag instead. "
                "Exactly zero measured loss is not something a benchmark can certify."
            )
        return self


# --------------------------------------------------------------------------
# detection
# --------------------------------------------------------------------------

def detect_hardware(req: InferOptRequest) -> HardwareFingerprint:
    q = ["nvidia-smi", "--query-gpu=name,compute_cap,memory.total", "--format=csv,noheader,nounits"]
    out = subprocess.run(q, capture_output=True, text=True, timeout=20).stdout.strip()
    if not out:
        raise RuntimeError("nvidia-smi returned no GPUs")
    rows = [r.split(",") for r in out.splitlines()]
    name, cc, mem = rows[0][0].strip(), rows[0][1].strip(), rows[0][2].strip()

    sys_ram_gb = 0.0
    for line in open("/proc/meminfo"):
        if line.startswith("MemTotal:"):
            sys_ram_gb = float(line.split()[1]) / 1024 / 1024
            break

    unified = any(p in name for p in UNIFIED_MEMORY_PARTS)
    # Unified parts report memory.total as [N/A]; the real budget is system RAM.
    try:
        mem_gb = float(mem)
    except ValueError:
        if not unified:
            raise RuntimeError(f"{name}: memory.total unreadable ({mem!r}) and not a known unified part")
        mem_gb = sys_ram_gb

    bw = req.override_memory_bandwidth_gb_s or MEMORY_BANDWIDTH_GB_S.get(name)
    if bw is None:
        raise RuntimeError(
            f"no memory bandwidth on record for {name!r}. This number bounds every decode "
            f"estimate, so it is not defaulted. Add it to MEMORY_BANDWIDTH_GB_S or pass "
            f"--override-memory-bandwidth-gb-s."
        )

    return HardwareFingerprint(
        gpu_name=name, gpu_count=len(rows), compute_capability=cc,
        memory_gb=round(mem_gb, 1), memory_bandwidth_gb_s=bw, unified_memory=unified,
        interconnect="nvlink" if len(rows) > 1 else None,
        system_ram_gb=round(sys_ram_gb, 1), cpu_cores=os.cpu_count() or 1,
    )


def _hf_config(model: str) -> dict:
    local = Path(model) / "config.json"
    if local.exists():
        return json.loads(local.read_text())
    return json.loads(Path(hf_hub_download(model, "config.json")).read_text())


def _checkpoint_bytes(model: str) -> float | None:
    """Bytes of weights from the checkpoint index -- the number the memory
    budget actually cares about.

    Reads only the index (a few KB), never a shard. Failures are REPORTED, not
    swallowed: falling back silently to arithmetic over config fields produced
    16.45B for a 14.8B model here, an 11% overestimate that no one would notice
    because the number looks reasonable.
    """
    try:
        idx = Path(model) / "model.safetensors.index.json"
        raw = idx.read_text() if idx.exists() else \
            Path(hf_hub_download(model, "model.safetensors.index.json")).read_text()
        return float(json.loads(raw)["metadata"]["total_size"])
    except Exception:
        pass

    # No index means a SINGLE-FILE checkpoint, not a missing one. Small models
    # ship one model.safetensors with no index at all, and treating that as a
    # failure sent every one of them down the arithmetic fallback.
    try:
        local = Path(model) / "model.safetensors"
        if local.exists():
            return float(local.stat().st_size)
        from huggingface_hub import HfApi
        info = HfApi().model_info(model, files_metadata=True)
        sizes = [f.size for f in info.siblings
                 if f.rfilename.endswith(".safetensors") and f.size]
        if sizes:
            return float(sum(sizes))
        raise FileNotFoundError("no .safetensors with a readable size")
    except Exception as e:
        print(f"    could not read the checkpoint index ({type(e).__name__}: {e}); "
              f"falling back to arithmetic over config fields, which over-counts "
              f"tied embeddings and ignores GQA")
        return None


def detect_quantization_capability(sm_major: int, *, log=print) -> dict[str, bool]:
    """Which quantized variants this pipeline can PRODUCE from the served model.

    Deliberately NOT "which published checkpoints exist on the Hub". A customer
    brings their own fine-tuned model; there is no `customer/their-model-AWQ`
    to download, and measuring `Qwen/Qwen3-14B-AWQ` would benchmark Qwen's
    quantization job rather than ours. The pipeline has to do the conversion
    itself or the result does not transfer to a real deployment.

    Capability is (hardware supports the format) AND (a producer exists):

      fp8       no producer needed. vLLM quantizes bf16 weights to FP8 during
                model load with `--quantization fp8` -- no artifact, no
                calibration, no conversion time. Needs cc >= 8.9.
      int4_awq  needs llmcompressor to run activation-aware calibration and
                write a checkpoint. awq_marlin kernels need cc >= 8.0.
      nvfp4     Blackwell-native FP4; needs cc >= 10.0 and llmcompressor.
    """
    from quantize import producer_available
    have_producer = producer_available()
    caps = {
        "fp8": sm_major >= 8,          # 8.9+ in practice; cc_major==8 covers Ada
        "int4_awq": sm_major >= 8 and have_producer,
        "nvfp4": sm_major >= 10 and have_producer,
    }
    if not have_producer:
        log(f"  quant     no local producer (.quant-pkgs missing) -- int4_awq and "
            f"nvfp4 will skip. Build it with: python quantize.py --setup")
    log(f"  quant     can produce: {', '.join(k for k, v in caps.items() if v) or 'nothing'}")
    return caps


def detect_model(req: InferOptRequest) -> ModelFingerprint:
    top = _hf_config(req.model)
    # Multimodal configs nest the language model under text_config. Reading the
    # top level finds nothing and raises KeyError on num_attention_heads.
    c = top.get("text_config") or top
    is_mm = bool(top.get("vision_config") or top.get("audio_config"))

    # Hybrid attention: count the layers that actually keep a growing KV cache.
    n_layers = c["num_hidden_layers"]
    full_attn = None
    if isinstance(c.get("layer_types"), list):
        full = [t for t in c["layer_types"] if "full" in str(t) or "attention" == str(t)]
        full_attn = len(full) or None
    if full_attn is None and c.get("full_attention_interval"):
        full_attn = max(1, n_layers // int(c["full_attention_interval"]))

    n_heads = c["num_attention_heads"]
    n_kv = c.get("num_key_value_heads", n_heads)
    attn = "mha" if n_kv == n_heads else ("mqa" if n_kv == 1 else "gqa")

    n_experts = c.get("num_experts") or c.get("num_local_experts") or 0
    n_active = c.get("num_experts_per_tok") or 0

    # Prefer the real parameter count from the checkpoint index over arithmetic
    # on config fields, which silently omits embeddings and expert weights.
    ck_bytes = _checkpoint_bytes(req.model)
    dtype = c.get("torch_dtype") or c.get("dtype") or "bfloat16"
    if ck_bytes:
        n_params_b = ck_bytes / DTYPE_BYTES.get(dtype, 2) / 1e9
    else:
        h, L, V = c["hidden_size"], n_layers, c.get("vocab_size", 0)
        inter, kvh = c.get("intermediate_size", 4 * h), c.get("num_key_value_heads", c["num_attention_heads"])
        hd = c.get("head_dim") or h // c["num_attention_heads"]
        # q + o are full width; k + v are narrowed by GQA. The naive 4*h*h
        # assumes MHA and over-counts every grouped-query model.
        # attn_params, NOT attn. This used to reuse the name of the attention
        # TYPE computed above, silently replacing "gqa" with a parameter count.
        # It only fired when the checkpoint index was unreadable -- i.e. on any
        # single-file model -- and surfaced as a pydantic literal_error naming an
        # integer, which reads like a schema problem and is not.
        attn_params = 2 * h * (c["num_attention_heads"] * hd) + 2 * h * (kvh * hd)
        tied = c.get("tie_word_embeddings", False)
        n_params_b = (L * (attn_params + 3 * h * inter) + (1 if tied else 2) * V * h) / 1e9

    active_b = None
    if n_experts:
        active_b = n_params_b * (n_active / n_experts) if n_experts else n_params_b

    return ModelFingerprint(
        id=req.model,
        architecture=(top.get("architectures") or ["unknown"])[0],
        is_dense=n_experts == 0,
        n_params_b=round(n_params_b, 2),
        n_layers=n_layers, hidden_size=c["hidden_size"],
        n_heads=n_heads, n_kv_heads=n_kv, attention_type=attn,
        native_dtype=dtype, checkpoint_bytes=ck_bytes,
        full_attention_layers=full_attn,
        explicit_head_dim=c.get("head_dim"),
        is_multimodal=is_mm,
        has_mtp=bool(c.get("mtp_num_hidden_layers")),
        n_experts=n_experts, n_active_experts=n_active,
        active_params_b=round(active_b, 2) if active_b else None,
        max_model_len=req.override_max_model_len or c.get("max_position_embeddings", 32768),
        supported_runtimes=req.runtimes or ["vllm", "sglang", "trtllm"],
    )


def detect_lora(req: InferOptRequest) -> LoraFingerprint:
    if not req.adapters:
        return LoraFingerprint()
    ranks, targets = [], set()
    for a in req.adapters:
        cfg = json.loads((Path(a) / "adapter_config.json").read_text())
        ranks.append(cfg.get("r", 0))
        targets |= set(cfg.get("target_modules") or [])
    return LoraFingerprint(
        n_adapters=len(req.adapters), max_rank=max(ranks),
        target_modules=sorted(targets), adapter_retained=True,
    )


def build_fingerprint(req: InferOptRequest) -> tuple[Fingerprint, SLO]:
    """The whole stage-0 step: six user inputs in, a full fingerprint out."""
    hw = detect_hardware(req)
    model = detect_model(req)

    # Which quantized variants can be PRODUCED here -- a property of this
    # machine's hardware and toolchain, not of the model, so it is resolved
    # after both are known.
    caps = detect_quantization_capability(hw.sm_major)
    model.can_quantize_fp8 = caps["fp8"]
    model.can_quantize_int4_awq = caps["int4_awq"]
    model.can_quantize_nvfp4 = caps["nvfp4"]

    fp = Fingerprint(
        model=model, hw=hw,
        workload=WorkloadFingerprint.from_trace(req.trace),
        lora=detect_lora(req),
    )
    slo = SLO(ttft_p99_ms=req.ttft_p99_ms, itl_p99_ms=req.itl_p99_ms,
              quality_budget=req.allow_loss)
    return fp, slo
