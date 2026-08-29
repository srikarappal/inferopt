"""The stage-0 fingerprint: the contract every DAG predicate reads.

    from fingerprint import Fingerprint, Context

Each field below is referenced by at least one `applicable_when` expression in
dag/*.json. Defining them here turns "predicate silently evaluated false because
the field was missing" -- a node that quietly never runs and nobody notices --
into a startup error.

Every field carries how it is OBTAINED, because they are not equally cheap:

    config    read from the model's config.json or the launch args      free
    derived   computed from other fingerprint fields                    free
    hardware  queried from the device                                   free
    registry  looked up in a checkpoint/draft-model index               seconds
    trace     computed from a real workload trace                       needs a trace
    probe     needs a measurement on the target hardware                needs a GPU

The `trace` fields are the ones with no shortcut. A fingerprint built from
guessed length distributions optimizes for a workload that does not exist.

HISTORY -- what this schema learned the hard way

  Hybrid attention was a 4x KV error. The first version multiplied KV bytes by
  every layer. Modern architectures interleave full attention with linear/SSM
  layers, and only the full-attention layers keep a cache that GROWS with
  context. Qwen3.5-9B has full_attention_interval=4, so 8 of 32 layers cache:
  128KB/token predicted against 32KB actual. At 256k context that is 34.4GB vs
  8.6GB -- the difference between "will not run" and "fits comfortably". Hence
  full_attention_layers and the kv_layers property.

  MoE conflated two different weights. Every expert is RESIDENT (memory) but
  only the routed ones are READ (bandwidth). One field for both sizes a
  235B-A22B model as either 22B of memory or 235B of bandwidth -- wrong in
  opposite directions. weight_gb and active_weight_gb are separate for that
  reason.

  The quantization fields meant the wrong thing. They were originally
  has_fp8_checkpoint / has_awq_checkpoint: "a published checkpoint of this model
  exists on the Hub". That is the wrong question. A customer brings their own
  fine-tune, which has no published quantization, and measuring somebody else's
  -AWQ repo benchmarks THEIR calibration job. Replaced with can_quantize_*:
  can this pipeline PRODUCE the variant here. See quantize.py.
"""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

AttentionType = Literal["mha", "gqa", "mqa", "mla"]
DTYPE_BYTES = {"float32": 4, "bfloat16": 2, "float16": 2, "fp8": 1, "int8": 1, "int4": 0.5}


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------

class ModelFingerprint(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    id: str = Field(description="config | HF id or local path of the served model")
    architecture: str = Field(description="config | e.g. Qwen3ForCausalLM")
    is_dense: bool = Field(True, description="config | False for MoE; gates the expert-placement subtree")
    n_params_b: float = Field(description="config | parameter count in billions")
    n_layers: int = Field(description="config | num_hidden_layers")
    hidden_size: int = Field(description="config | hidden_size")
    n_heads: int = Field(description="config | num_attention_heads")
    n_kv_heads: int = Field(description="config | num_key_value_heads; equals n_heads for MHA")
    attention_type: AttentionType = Field(description="derived | from the head ratio")
    native_dtype: str = Field("bfloat16", description="config | torch_dtype")

    # MoE. All experts are RESIDENT (memory) but only k are ACTIVE (compute) --
    # two different numbers used for two different things, and conflating them
    # is how a 235B-A22B model gets sized as if it were 22B.
    n_experts: int = Field(0, ge=0, description="config | 0 for dense")
    n_active_experts: int = Field(0, ge=0, description="config | experts routed to per token")
    active_params_b: float | None = Field(None,
        description="config | params actually multiplied per token. Equals n_params_b when dense. "
                    "Drives compute/roofline; n_params_b drives memory.")

    supported_runtimes: list[str] = Field(default_factory=lambda: ["vllm"],
        description="registry | which of vllm/sglang/trtllm can serve this architecture. "
                    "A hard constraint on what stage 1.2 is allowed to recommend.")
    max_model_len: int = Field(description="config | max_position_embeddings, or the launch override")

    # Hybrid attention. Newer architectures interleave full attention with
    # linear/SSM layers: only the full-attention layers keep a KV cache that
    # GROWS with context, while linear layers hold a fixed-size state. Assuming
    # every layer caches overstates KV per token by the interleave factor --
    # 4x on a model with full_attention_interval=4 -- which mis-sizes context,
    # memory headroom and every decision that depends on them.
    full_attention_layers: int | None = Field(None,
        description="config | layers with a growing KV cache. None means all of them.")
    checkpoint_bytes: float | None = Field(None,
        description="registry | weight bytes from the safetensors index. Preferred over "
                    "params x dtype: checkpoints mix precisions (fp32 norms, tied embeddings) "
                    "and the memory budget cares about bytes, not parameters.")
    explicit_head_dim: int | None = Field(None,
        description="config | head_dim when the config states it rather than implying "
                    "hidden_size/n_heads; the two disagree on some architectures")
    is_multimodal: bool = Field(False,
        description="config | a vision/audio tower is present. A text-only workload leaves it "
                    "idle, but it occupies memory and dag/vlm.json owns its tuning.")
    has_mtp: bool = Field(False,
        description="config | native multi-token prediction, i.e. speculative decoding built "
                    "into the model. Competes with the spec_decode nodes and usually wins.")

    # Which quantized variants this pipeline can PRODUCE from the served model.
    # Not "which exist on the Hub": a customer's fine-tune has no published
    # quantization, and benchmarking someone else's would measure their
    # conversion job rather than ours.
    can_quantize_fp8: bool = Field(False, description="capability | vLLM can quantize weights to FP8 at load time (no artifact)")
    can_quantize_int4_awq: bool = Field(False, description="capability | llmcompressor can produce an INT4-AWQ checkpoint here")
    can_quantize_nvfp4: bool = Field(False, description="capability | llmcompressor can produce an NVFP4 checkpoint here")
    has_compatible_draft: bool = Field(False, description="registry | a draft model sharing this tokenizer exists")
    draft_model: str | None = Field(None, description="registry | its id")

    @computed_field
    @property
    def head_dim(self) -> int:
        """derived | the config's head_dim when stated, else hidden_size / n_heads."""
        return self.explicit_head_dim or (self.hidden_size // self.n_heads)

    @computed_field
    @property
    def kv_layers(self) -> int:
        """derived | layers that actually keep a growing KV cache."""
        return self.full_attention_layers or self.n_layers

    @computed_field
    @property
    def weight_gb(self) -> float:
        """derived | RESIDENT weight bytes. For MoE this is every expert, not
        just the routed ones -- they all have to be in memory. Referenced by
        lora_serve_strategy to decide whether N merged replicas fit."""
        if self.checkpoint_bytes:
            return self.checkpoint_bytes / 1e9
        return self.n_params_b * DTYPE_BYTES.get(self.native_dtype, 2)

    @computed_field
    @property
    def active_weight_gb(self) -> float:
        """derived | weight bytes actually READ per token. This, not weight_gb,
        is what bounds memory-bound decode -- an MoE model reads only its routed
        experts even though all of them occupy memory."""
        active = self.active_params_b if self.active_params_b is not None else self.n_params_b
        return active * DTYPE_BYTES.get(self.native_dtype, 2)

    @computed_field
    @property
    def kv_bytes_per_token(self) -> int:
        """derived | 2 (K and V) x layers x kv_heads x head_dim x dtype bytes.

        The number that decides how much context fits in a given KV budget, and
        why GQA models tolerate long context so much better than MHA ones.
        """
        return int(2 * self.kv_layers * self.n_kv_heads * self.head_dim
                   * DTYPE_BYTES.get(self.native_dtype, 2))

    @model_validator(mode="after")
    def _check(self):
        if self.has_compatible_draft and not self.draft_model:
            raise ValueError("has_compatible_draft=True but draft_model is unset")
        if self.n_kv_heads > self.n_heads:
            raise ValueError(f"n_kv_heads {self.n_kv_heads} > n_heads {self.n_heads}")
        if not self.is_dense and self.n_experts == 0:
            raise ValueError("is_dense=False requires n_experts")
        if self.n_experts and self.active_params_b is None:
            raise ValueError(
                "MoE requires active_params_b -- without it the model is sized for "
                "compute as if every expert ran on every token"
            )
        if self.active_params_b and self.active_params_b > self.n_params_b:
            raise ValueError("active_params_b exceeds n_params_b")
        return self


# --------------------------------------------------------------------------
# hardware
# --------------------------------------------------------------------------

class HardwareFingerprint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gpu_name: str = Field(description="hardware | nvidia-smi --query-gpu=name")
    gpu_count: int = Field(1, description="hardware | devices available to this run")
    compute_capability: str = Field(description="hardware | e.g. '12.1'; decides FP8/FP4 support")
    memory_gb: float = Field(description="hardware | usable device memory. On unified-memory parts this is SYSTEM memory, and the CPU competes for it")
    memory_bandwidth_gb_s: float = Field(description="hardware | peak; the binding constraint for decode")
    unified_memory: bool = Field(False, description="hardware | True on GB10/Grace-class parts where CPU and GPU share one pool")
    interconnect: str | None = Field(None, description="hardware | nvlink/pcie/none; irrelevant at gpu_count=1")
    system_ram_gb: float = Field(description="hardware | host RAM. On unified-memory parts this IS memory_gb "
                                             "and the CPU competes for it; also bounds the CPU-side adapter cache.")
    cpu_cores: int = Field(description="hardware | the load generator is async Python -- too few cores and the "
                                       "benchmark measures its own client rather than the server")
    # Deliberately NOT modelled: NUMA topology and NIC/network. Both are
    # single-node-irrelevant here and only start to matter for disaggregated
    # prefill/decode serving, which is not in any current DAG.

    @computed_field
    @property
    def sm_major(self) -> int:
        return int(self.compute_capability.split(".")[0])

    @computed_field
    @property
    def supports_fp8(self) -> bool:
        """derived | Hopper (9.x) and later have FP8 tensor cores."""
        return self.sm_major >= 9

    @computed_field
    @property
    def supports_fp4(self) -> bool:
        """derived | Blackwell (10.x+) adds FP4. Gates the NVFP4 node."""
        return self.sm_major >= 10

    @model_validator(mode="after")
    def _check(self):
        if self.unified_memory and self.memory_bandwidth_gb_s > 2000:
            raise ValueError(
                f"unified_memory=True with {self.memory_bandwidth_gb_s} GB/s looks like an "
                f"HBM figure. Unified parts are bandwidth-limited (GB10 is ~273 GB/s) and "
                f"that number drives every decode estimate."
            )
        return self


# --------------------------------------------------------------------------
# workload -- the fields with no shortcut
# --------------------------------------------------------------------------

class WorkloadFingerprint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_requests: int = Field(description="trace | requests in the sample")
    mean_input_tokens: float = Field(description="trace |")
    p99_input_tokens: int = Field(description="trace | gates chunked_prefill (>1024) and kv_block_size (>4096)")
    p999_input_tokens: int = Field(description="trace | drives max_model_len right-sizing; the tail sets the ceiling")
    mean_output_tokens: float = Field(description="trace |")
    p99_output_tokens: int = Field(description="trace |")
    request_rate_qps: float = Field(description="trace | sustained arrival rate")
    max_concurrency: int = Field(description="trace | peak in-flight requests")
    burstiness: float = Field(1.0, description="trace | peak qps / mean qps; 1.0 is uniform arrival")

    prefix_overlap: float = Field(0.0, ge=0.0, le=1.0,
        description="trace | fraction of prompt tokens shared with an earlier request")
    prefix_overlap_per_adapter: float = Field(0.0, ge=0.0, le=1.0,
        description="trace | THE number prefix_caching gates on. Under multi-LoRA the cache "
                    "keys on (prefix, lora_id), so N adapters fragment the hit rate N ways. "
                    "Equals prefix_overlap when there is no LoRA.")

    multi_turn: bool = Field(False, description="trace | conversations reuse context across turns")
    greedy: bool = Field(True, description="trace | temperature==0; equivalence probing assumes it")
    temperature: float = Field(0.0, ge=0.0, description="trace |")
    top_p: float = Field(1.0, gt=0.0, le=1.0, description="trace |")
    structured_generation: float = Field(0.0, ge=0.0, le=1.0,
        description="trace | fraction of requests using guided decoding (JSON schema, regex). "
                    "Carries real per-token overhead AND conflicts with speculative decoding, "
                    "since a constrained sampler can reject otherwise-valid drafts.")

    trace_ref: str | None = Field(None,
        description="trace | path to the request trace these statistics were computed FROM. "
                    "Summary statistics describe a workload; they cannot reproduce one. The "
                    "benchmark samples real (input_len, output_len, arrival, prefix, adapter) "
                    "tuples from here, so a fingerprint without a trace_ref is optimizing "
                    "against a shape someone typed in.")

    @computed_field
    @property
    def decode_fraction(self) -> float:
        """derived | output tokens / all tokens. Gates speculative decoding.

        Deliberately a TOKEN-COUNT ratio, because it is computable from a trace
        alone. Note decode dominates wall-clock far more than this suggests --
        a decode token costs ~100x a prefill token - so the 0.4 threshold in the
        DAG is set low on purpose.
        """
        total = self.mean_input_tokens + self.mean_output_tokens
        return self.mean_output_tokens / total if total else 0.0

    @model_validator(mode="after")
    def _check(self):
        if self.p99_input_tokens > self.p999_input_tokens:
            raise ValueError("p99_input_tokens exceeds p999_input_tokens")
        if self.prefix_overlap_per_adapter > self.prefix_overlap + 1e-9:
            raise ValueError(
                "prefix_overlap_per_adapter cannot exceed the global overlap -- "
                "splitting traffic across adapters can only fragment sharing, never create it"
            )
        return self


# --------------------------------------------------------------------------
# lora
# --------------------------------------------------------------------------

    @classmethod
    def from_trace(cls, path: str, **overrides) -> "WorkloadFingerprint":
        """Derive every statistic from a real trace, so none of them is guessed.

        Trace is JSONL, one record per request:
            {"input_tokens": int, "output_tokens": int, "arrival_ts": float,
             "prefix_id": str|null, "adapter_id": str|null, "temperature": float}
        """
        import json as _json
        from collections import Counter as _Counter

        rows = [_json.loads(l) for l in open(path) if l.strip()]
        if not rows:
            raise ValueError(f"{path}: empty trace")

        def pct(xs, q):
            s = sorted(xs)
            return int(s[min(len(s) - 1, int(q * (len(s) - 1)))])

        ins = [r["input_tokens"] for r in rows]
        outs = [r["output_tokens"] for r in rows]
        ts = sorted(r["arrival_ts"] for r in rows if r.get("arrival_ts") is not None)
        span = (ts[-1] - ts[0]) if len(ts) > 1 else 0.0
        qps = len(rows) / span if span > 0 else 0.0

        # Burstiness from the busiest one-second bucket against the mean.
        burst = 1.0
        if span > 0:
            buckets = _Counter(int(t - ts[0]) for t in ts)
            burst = max(buckets.values()) / max(1e-9, qps)

        prefixes = [r.get("prefix_id") for r in rows if r.get("prefix_id")]
        shared = sum(c for c in _Counter(prefixes).values() if c > 1)
        overlap = shared / len(rows) if rows else 0.0

        adapters = [r.get("adapter_id") for r in rows if r.get("adapter_id")]
        if adapters:
            per = _Counter(zip((r.get("prefix_id") for r in rows), adapters))
            shared_pa = sum(c for c in per.values() if c > 1)
            overlap_pa = shared_pa / len(rows)
        else:
            overlap_pa = overlap

        temps = [r.get("temperature", 0.0) for r in rows]
        fields = dict(
            n_requests=len(rows),
            mean_input_tokens=sum(ins) / len(ins), p99_input_tokens=pct(ins, 0.99),
            p999_input_tokens=pct(ins, 0.999),
            mean_output_tokens=sum(outs) / len(outs), p99_output_tokens=pct(outs, 0.99),
            request_rate_qps=qps, max_concurrency=max(1, int(qps * 2)),
            burstiness=burst, prefix_overlap=overlap,
            prefix_overlap_per_adapter=min(overlap_pa, overlap),
            temperature=sum(temps) / len(temps), greedy=all(t == 0.0 for t in temps),
            trace_ref=path,
        )
        fields.update(overrides)
        return cls(**fields)


class LoraFingerprint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_adapters: int = Field(0, ge=0, description="config | 0 = no LoRA; 1 = merge upstream; >1 = the subtree runs")
    max_rank: int = Field(0, ge=0, description="config | largest rank across adapters; right-sizes buffers")
    target_modules: list[str] = Field(default_factory=list,
        description="config | q_proj/k_proj/... ; MLP targets cost materially more per token")
    adapter_retained: bool = Field(True,
        description="config | adapter artifacts kept after a stage-0 merge. Required by "
                    "lora_unmerge_for_weight_quant, which needs to serve unmerged again.")
    request_distribution: Literal["uniform", "skewed", "unknown"] = Field("unknown",
        description="trace | uniform means every adapter is hot; skewed means a few are")
    adapter_switching_rate: float = Field(0.0, ge=0.0, le=1.0,
        description="trace | fraction of consecutive requests that change adapter")

    max_loras_gpu_resident: int = Field(8, ge=1,
        description="derived | how many adapters fit in GPU memory at this rank. When "
                    "n_adapters exceeds it, cold adapters must be swapped and lora_cpu_cache "
                    "starts to matter.")

    merged_upstream: bool = Field(False,
        description="derived | stage 0 merged a single adapter into the base. When True the "
                    "served model is architecturally the base: the predictor is valid and no "
                    "LoRA node fires.")

    @computed_field
    @property
    def multi_lora_active(self) -> bool:
        """derived | LoRA kernels are actually in the serving path. Gates
        spec_decode_draft off, because a draft cannot track N adapted targets."""
        return self.n_adapters > 1 and not self.merged_upstream

    @model_validator(mode="after")
    def _check(self):
        if self.n_adapters > 0 and self.max_rank == 0:
            raise ValueError("n_adapters > 0 requires max_rank")
        if self.merged_upstream and self.n_adapters != 1:
            raise ValueError(
                f"merged_upstream is only valid for exactly one adapter, got {self.n_adapters}. "
                f"Merging N adapters into one base is a different (lossy) operation."
            )
        return self


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------

class SLO(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ttft_p99_ms: float | None = Field(None, description="user | requests slower than this do not count toward goodput")
    itl_p99_ms: float | None = Field(None, description="user |")
    throughput_target_tok_s: float | None = Field(None, description="user | minimum acceptable throughput")
    quality_budget: float | None = Field(None, ge=0.0, le=1.0,
        description="user | allowed quality loss on the LOSSY branch (--allow-loss). "
                    "None means explore anyway and return the frontier.")
    lossless_quality_budget: float = Field(0.03, ge=0.0, le=1.0,
        description="user | allowed quality movement on the LOSSLESS branch "
                    "(--lossless-tolerance). A lossless step should not move the eval at "
                    "all; anything above this is a defect worth failing on, not a budget "
                    "to spend. Kept separate from quality_budget because conflating them "
                    "is how the lossy gate ended up with zero width -- the measured "
                    "lossless drift WAS the lossy budget.")


class Fingerprint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: ModelFingerprint
    hw: HardwareFingerprint
    workload: WorkloadFingerprint
    lora: LoraFingerprint = Field(default_factory=LoraFingerprint)

    @model_validator(mode="after")
    def _cross(self):
        need = self.workload.p999_input_tokens + self.workload.p99_output_tokens
        if need > self.model.max_model_len:
            raise ValueError(
                f"workload needs {need} tokens of context but max_model_len is "
                f"{self.model.max_model_len} -- the tail of this traffic cannot be served"
            )
        kv_gb = self.model.kv_bytes_per_token * need / 1e9
        if kv_gb + self.model.weight_gb > self.hw.memory_gb:
            raise ValueError(
                f"weights ({self.model.weight_gb:.1f}GB) + KV for one max-length sequence "
                f"({kv_gb:.1f}GB) exceeds device memory ({self.hw.memory_gb}GB) -- "
                f"this model cannot serve this workload on this hardware at all"
            )
        return self


class NodeMeasurement(BaseModel):
    """What `measurements.<node>.<field>` resolves to during traversal."""
    model_config = ConfigDict(extra="allow")
    kept: bool = False
    goodput: float | None = None
    ttft_p99_ms: float | None = None
    itl_p99_ms: float | None = None
    quality: dict[str, float] = Field(default_factory=dict)
    spec_acceptance_rate: float | None = None
    config: dict[str, Any] = Field(default_factory=dict)


class Preconditions(BaseModel):
    """Stage-0 outcomes the DAG reads back."""
    model_config = ConfigDict(extra="forbid")
    merge_verification: dict[str, Any] = Field(default_factory=lambda: {"merged": False})


class Context(BaseModel):
    """Exactly what a predicate may reference. Anything not reachable from here
    is a typo, and the validator should say so rather than quietly returning
    false at hour two of a run."""
    model_config = ConfigDict(extra="forbid")

    fingerprint: Fingerprint
    slo: SLO = Field(default_factory=SLO)
    measurements: dict[str, NodeMeasurement] = Field(default_factory=dict)
    preconditions: Preconditions = Field(default_factory=Preconditions)
    incumbent: dict[str, Any] = Field(default_factory=dict)
    incumbent_metrics: NodeMeasurement | None = Field(None,
        description="What stage 1.3 measured for the incumbent config. Seeds the traversal so "
                    "the FIRST node faces the same accept_band as every other one -- without it "
                    "the first measured node is kept unconditionally, because there is nothing "
                    "to compare against, and its config is adopted on no evidence.")
    accept_band: float = Field(0.02,
        description="from the calibration store. FROZEN for the whole traversal -- a band that "
                    "moved mid-run would leave early and late keep/revert decisions resting on "
                    "different criteria, and the frontier built from both.")
    quality_baseline: dict[str, float] = Field(default_factory=dict,
        description="per-benchmark scores of the incumbent, set at stage 1.3 and refreshed at "
                    "lossless_complete. What every lossy node's delta is measured against.")
    quality_tolerance: dict[str, float] = Field(default_factory=dict,
        description="per-benchmark noise floor. EVOLVES: a default until lossless_complete "
                    "measures it, measured for every lossy node after that.")

    @computed_field
    @property
    def workload(self) -> WorkloadFingerprint:
        """Predicates say `workload.x` as well as `fingerprint.workload.x`;
        both resolve here."""
        return self.fingerprint.workload


def resolve(ctx: Context, path: str) -> Any:
    """Dotted-path lookup with an error that names the failure.

    A missing field must never read as False -- that is how a node silently
    stops running for the rest of the project.
    """
    cur: Any = ctx
    for i, part in enumerate(path.split(".")):
        if isinstance(cur, dict):
            if part not in cur:
                raise KeyError(f"{path}: no key {part!r} at {'.'.join(path.split('.')[:i]) or '<root>'}")
            cur = cur[part]
        else:
            if not hasattr(cur, part):
                raise KeyError(f"{path}: {type(cur).__name__} has no field {part!r}")
            cur = getattr(cur, part)
    return cur
