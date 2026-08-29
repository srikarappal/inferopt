"""Produce quantized variants of the SERVED model, locally.

The pipeline must do its own conversion. A customer brings their own fine-tuned
model; there is no published quantization of it to download, and benchmarking
somebody else's `-AWQ` repo would measure their calibration job rather than
ours. Anything this module cannot produce, the DAG must skip -- not substitute.

Three variants, and they are not the same kind of thing:

  fp8       NO artifact and NO producer. vLLM quantizes bf16 weights to FP8
            during model load (`--quantization fp8`). Zero conversion time,
            zero disk, works on any cc>=8.9 part. It is a launch flag, so the
            DAG treats it like any other flag node.

  int4_awq  Activation-aware weight quantization. Needs calibration data and
            writes a checkpoint. ~20-40 min for a 14B.

  nvfp4     Blackwell-native FP4. Same producer, same cost, needs cc>=10.

CALIBRATION USES THE USER'S OWN TRACE. AWQ picks per-channel scales from
observed activation magnitudes, so the calibration distribution decides which
channels are protected. The stock choice (pileval, wikitext) is generic web
text; this workload's prompts are what the model will actually see. Calibrating
on the real trace is both more faithful and free -- the trace is already loaded.

TOOLCHAIN ISOLATION. llmcompressor pins compressed-tensors==0.18.0 while vLLM
0.26 pins ==0.17.0, so it can never share the serving environment -- the same
constraint that put aiconfigurator in its own directory. It installs to
.quant-pkgs and runs as a subprocess.

That isolation fixes the INSTALL. It does not prove vLLM can READ what the
producer WRITES: compressed-tensors is both writer and reader, and the two
sides are a minor version apart. `--smoke` settles that on a small model in a
few minutes rather than discovering it 40 minutes into a 14B conversion.

HISTORY

  Written after the Hub-lookup approach was rejected outright: pulling
  Qwen/Qwen3-14B-AWQ benchmarks Qwen's calibration job, and a customer's own
  fine-tune has no such repo. The pipeline has to do its own conversion or the
  result does not transfer to a real deployment.

  compressed-tensors is both the writer and the reader. llmcompressor 0.13.0
  pins ==0.18.0; vLLM 0.26 pins ==0.17.0. Isolation into .quant-pkgs solves the
  INSTALL. It does not prove vLLM can read what the producer writes -- hence
  --smoke, which settles it on a 0.6B model in minutes rather than after a
  40-minute conversion of the real one. (First evidence is good: vLLM logged
  `quantization=compressed-tensors`, so it recognised the format.)

  --no-deps skipped packages the env genuinely lacked. accelerate and auto-round
  are not in the serving env at all, so reusing site-packages silently produced
  ModuleNotFoundError inside the conversion job. PRODUCER_PKGS names them.

  The load probe hit the ninja bug. Bare subprocess.run instead of child_env(),
  failing with FileNotFoundError: ninja deep in engine init -- which looked like
  a format incompatibility and was not. The error handler now reports the actual
  exception rather than vLLM's "see root cause above" wrapper, and distinguishes
  "format not recognised" from "recognised, failed later".

  The first recipe quantized every Linear identically, protecting only lm_head.
  AWQ is selective WITHIN a layer (it scales outlier channels found from
  calibration activations) but nothing was selective ACROSS layers.
  sensitivity_ignore now derives protections from the fingerprint.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
QUANT_PKGS = HERE / ".quant-pkgs"
ARTIFACTS = HERE / "artifacts"

# NVIDIA's own quantizer, replacing llmcompressor. Two reasons beyond provenance:
#
#   no pin conflicts   llmcompressor needs compressed-tensors==0.18.0 against
#                      vLLM's ==0.17.0, which forced an isolated --target
#                      directory and a subprocess. modelopt pins only torch>=2.8
#                      and an unpinned numpy, so it installs into the serving env
#                      and runs in-process.
#   one tool           FP8, INT4-AWQ and NVFP4 from one library, plus
#                      AutoQuantize -- a per-layer mixed-precision search that
#                      answers "which layers to protect" by measurement rather
#                      than by the hand-picked list below.
#
# setuptools is pinned because installing modelopt pulls 81.0.0 and vLLM requires
# <81.0.0; without this the serving environment breaks on the next launch.
PRODUCER_PKGS = [
    "nvidia-modelopt",
    # transformers needs accelerate for device_map="auto", which is how a model
    # too large for one GPU gets loaded for calibration. Loose pins: torch>=2.0,
    # numpy>=1.17, nothing that collides with vLLM.
    "accelerate",
    # LAST, and deliberately. Installing modelopt pulls setuptools 81.0.0 while
    # vLLM requires <81.0.0; without pinning it back the serving environment
    # breaks on the next launch -- a quantizer that silently disables the server
    # it is quantizing for.
    "setuptools<81.0.0",
]

# A small model with the same architecture family as the targets, used only to
# prove the produce->load round trip before spending 40 minutes on a real one.
SMOKE_MODEL = "Qwen/Qwen3-0.6B"

N_CALIB = 256          # AWQ converges well before 512; each sample costs a forward pass
CALIB_MAX_TOKENS = 2048


# --------------------------------------------------------------------------
# toolchain
# --------------------------------------------------------------------------

def producer_available() -> bool:
    """True if modelopt is importable. No isolated directory -- see PRODUCER_PKGS."""
    try:
        import modelopt.torch.quantization  # noqa: F401
        return True
    except Exception:
        return False


def setup(log=print) -> bool:
    """Install the producer into .quant-pkgs, isolated from the serving env.

    --no-deps on purpose: llmcompressor 0.13.0's ranges for torch, transformers,
    datasets, numpy and accelerate are ALL satisfied by what the serving env
    already has, so those are reused from site-packages. Only compressed-tensors
    is installed here, where it shadows the env's 0.17.0 for this subprocess
    only. Installing full deps would pull a second ~2.5GB torch for no benefit.
    """
    for pkg in PRODUCER_PKGS:
        log(f"  installing {pkg}")
        r = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", pkg],
            capture_output=True, text=True)
        if r.returncode != 0:
            log(f"  FAILED: {r.stderr.strip()[-500:]}")
            return False
    log(f"  producer ready. Verify the format round-trips: python quantize.py --smoke")
    return True


def _child_env() -> dict:
    """The serving env, plus PATH so the child can find ninja.

    modelopt needs no shadowing -- it has no pin conflicts with vLLM, which is
    most of why it replaced llmcompressor. The subprocess remains only to keep a
    multi-GB model load out of the parent process.
    """
    from evaluator import child_env
    return child_env()


# --------------------------------------------------------------------------
# the conversion job (runs in the child)
# --------------------------------------------------------------------------

def sensitivity_ignore(fp, *, log=print) -> list[str]:
    """Modules to keep at full precision, derived from the fingerprint.

    AWQ is already selective WITHIN a layer: it reads calibration activations,
    finds the ~1% of channels carrying outlier magnitudes, and per-channel
    scales them so they survive 4 bits. That part is automatic.

    ACROSS layers it is not selective at all -- `targets=["Linear"]` quantizes
    every projection identically. Some modules cannot absorb that, and the two
    below are structural rather than empirical, so they are defaults:

      lm_head       projects to vocabulary. Error here perturbs the output
                    distribution directly, with no later layer to absorb it.

      MoE router    (`mlp.gate`, NOT `gate_proj`) selects experts via argmax/
                    top-k. Quantization error near a decision boundary does not
                    degrade the output slightly -- it routes the token to a
                    DIFFERENT expert, a discrete jump. A dense layer's error
                    averages out; a router's does not.

      vision tower  a multimodal encoder is calibrated by text prompts here,
                    which is no calibration at all for image features.

    Deliberately NOT included: the first and last decoder blocks. They are
    widely said to be more sensitive, and that may well hold for this model, but
    it is a claim this pipeline should MEASURE rather than assume, as its own
    lossy node. Baking in an unmeasured protection would cost quality headroom
    on every model to hedge against some.
    """
    ignore = ["lm_head"]
    if not fp.model.is_dense:
        # `re:` prefixes are llmcompressor regexes. `mlp.gate$` anchors to the
        # router and does NOT match `gate_proj`, which is an ordinary FFN
        # projection and should be quantized like the rest.
        ignore.append("re:.*mlp.gate$")
        log(f"            MoE: protecting the expert router (quantization error "
            f"there changes WHICH expert runs, not just by how much)")
    if fp.model.is_multimodal:
        ignore.append("re:.*visual.*")
        log(f"            multimodal: protecting the vision tower (text prompts "
            f"do not calibrate image features)")
    return ignore


# modelopt config per format. These are NVIDIA's own presets, not hand-rolled
# recipes -- the point of switching from llmcompressor was to stop maintaining a
# second opinion about how to quantize.
MODELOPT_CFG = {
    "fp8":      "FP8_DEFAULT_CFG",
    "int4_awq": "INT4_AWQ_CFG",
    "nvfp4":    "NVFP4_DEFAULT_CFG",
}

_JOB = r'''
import json, sys
from pathlib import Path
model_id, kind, out_dir, calib_path, ignore_json = sys.argv[1:6]
ignore = json.loads(ignore_json)
CALIB_MAX_TOKENS = 2048

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_checkpoint

SINGLE = {"fp8": "FP8_DEFAULT_CFG", "int4_awq": "INT4_AWQ_CFG", "nvfp4": "NVFP4_DEFAULT_CFG"}

tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")

prompts = [json.loads(l)["prompt"] for l in open(calib_path) if l.strip()]
print(f"[job] {len(prompts)} calibration prompts from the workload trace", flush=True)

batches = []
for p in prompts:
    enc = tok(p, return_tensors="pt", truncation=True, max_length=CALIB_MAX_TOKENS)
    batches.append({k: v.to(model.device) for k, v in enc.items()})

if kind.startswith("autoquant@"):
    bits = float(kind.split("@", 1)[1])
    # AutoQuantize: score every layer's sensitivity, then solve for a per-layer
    # format assignment that hits `effective_bits`. Sensitive layers keep FP8,
    # tolerant ones drop to NVFP4, and the truly sensitive are skipped entirely.
    #
    # method="kl_div" measures the divergence between unquantized and quantized
    # outputs. The alternative, "gradient", is more principled but needs labels
    # and a backward pass -- and we have prompts from a workload trace, not a
    # labelled set. KL needs only a forward pass returning logits.
    print(f"[job] auto_quantize to effective_bits={bits}, method=kl_div", flush=True)
    model, state = mtq.auto_quantize(
        model,
        constraints={"effective_bits": bits},
        quantization_formats=["NVFP4_DEFAULT_CFG", "FP8_DEFAULT_CFG"],
        data_loader=batches,
        forward_step=lambda m, b: m(**b).logits,
        method="kl_div",
        disabled_layers=ignore or None,
        num_calib_steps=min(512, len(batches)),
        num_score_steps=min(128, len(batches)),
        verbose=True,
    )
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "autoquantize_search.json").write_text(
        json.dumps({"effective_bits": bits,
                    "state": {k: str(v)[:4000] for k, v in (state or {}).items()}},
                   indent=2, default=str))
elif kind in SINGLE:
    # Single-format PTQ. modelopt's presets already exclude lm_head, *router*,
    # *mlp.gate.*, vision towers and MTP heads, so `ignore` only adds to that.
    cfg = getattr(mtq, SINGLE[kind])
    print(f"[job] {SINGLE[kind]}; additionally disabling {ignore}", flush=True)

    def forward_loop(m):
        for i, b in enumerate(batches):
            with torch.no_grad():
                m(**b)
            if (i + 1) % 32 == 0:
                print(f"[job] calibrated {i+1}/{len(batches)}", flush=True)

    if ignore:
        import copy
        cfg = copy.deepcopy(cfg)
        # quant_cfg is a LIST of {quantizer_name, ...} entries in modelopt 0.46,
        # not a pattern-keyed dict.
        for pat in ignore:
            key = pat[3:] if pat.startswith("re:") else pat
            cfg["quant_cfg"].append(
                {"quantizer_name": f"*{key.strip('*.^$')}*", "enable": False})
    model = mtq.quantize(model, cfg, forward_loop)
else:
    raise SystemExit(f"[job] unknown kind {kind!r}")

export_hf_checkpoint(model, export_dir=out_dir)
tok.save_pretrained(out_dir)
print(f"[job] wrote {out_dir}", flush=True)
'''


def _write_calibration(trace_path: str, dest: Path, n: int = N_CALIB) -> int:
    """Sample calibration prompts from the workload trace.

    Sampled across the whole trace rather than taking the head: traces are often
    ordered, and calibrating on the first N requests would fit the scales to
    whatever topic happened to open the capture.
    """
    rows = [l for l in Path(trace_path).read_text().splitlines() if l.strip()]
    step = max(1, len(rows) // n)
    picked = rows[::step][:n]
    dest.write_text("\n".join(picked) + "\n")
    return len(picked)


def ensure_variant(fp, kind: str, trace_path: str, *, log=print) -> str | None:
    """Path to a quantized checkpoint of `model_id`, producing it if needed.

    Returns None for `fp8`, which is a launch flag rather than an artifact --
    the caller sets quantization="fp8" instead of swapping the model path.
    Raises on failure: a silently skipped conversion would leave the DAG
    measuring the unquantized model while reporting it as quantized.
    """
    if kind == "fp8":
        return None      # a load-time flag, no artifact
    if not producer_available():
        raise RuntimeError(
            f"cannot produce {kind}: the producer is not installed.\n"
            f"    python quantize.py --setup")

    model_id = fp.model.id
    # The bit budget is part of the identity: autoquant@6.0 and autoquant@4.5 are
    # different checkpoints and must not share a directory or a cache hit.
    out = ARTIFACTS / f"{model_id.replace('/', '__')}--{kind.replace('@', '_')}"
    if (out / "config.json").exists():
        log(f"  quant     reusing {out.relative_to(HERE)}")
        return str(out)

    out.mkdir(parents=True, exist_ok=True)
    calib = out.parent / f"{out.name}.calib.jsonl"
    n = _write_calibration(trace_path, calib)
    log(f"  quant     producing {kind} from {model_id}")
    log(f"            calibrating on {n} prompts from the workload trace "
        f"(not pileval -- these are the activations the model will actually see)")
    ignore = sensitivity_ignore(fp, log=log)

    job = out.parent / f"{out.name}.job.py"
    job.write_text(_JOB)
    r = subprocess.run([sys.executable, str(job), model_id, kind, str(out),
                        str(calib), json.dumps(ignore)],
                       env=_child_env(), capture_output=True, text=True)
    if r.returncode != 0 or not (out / "config.json").exists():
        shutil.rmtree(out, ignore_errors=True)
        tail = "\n".join((r.stderr or r.stdout).strip().splitlines()[-15:])
        raise RuntimeError(f"{kind} conversion failed:\n{textwrap.indent(tail, '    ')}")
    for line in (r.stdout or "").splitlines():
        if line.startswith("[job]"):
            log(f"            {line[6:]}")
    log(f"  quant     wrote {out.relative_to(HERE)}")
    return str(out)


# --------------------------------------------------------------------------
# smoke test: does vLLM read what the producer writes?
# --------------------------------------------------------------------------

def smoke(log=print) -> bool:
    """Produce a tiny quantized model and load it in vLLM.

    compressed-tensors is the writer (0.18.0, isolated) and the reader (0.17.0,
    in the serving env). One minor version apart. This proves the round trip on
    a 0.6B model in a few minutes, instead of finding the incompatibility after
    a 40-minute conversion of the real one.
    """
    if not producer_available():
        log("  producer not installed; run: python quantize.py --setup")
        return False

    calib = ARTIFACTS / "smoke.trace.jsonl"
    calib.parent.mkdir(parents=True, exist_ok=True)
    # A COMPLETE trace record, not just prompts. The same file is handed to
    # build_fingerprint, and WorkloadFingerprint.from_trace needs input_tokens,
    # output_tokens and arrival_ts to exist -- a prompts-only file raised
    # KeyError: 'input_tokens' before the producer was ever reached.
    calib.write_text("\n".join(
        json.dumps({
            "prompt": f"Explain in one or two sentences why consideration {i} "
                      f"matters when designing a distributed system, and give one "
                      f"concrete example.",
            "input_tokens": 32, "output_tokens": 64,
            "arrival_ts": round(i * 0.1, 3),
            "prefix_id": None, "adapter_id": None, "temperature": 0.0,
        })
        for i in range(64)) + "\n")

    ok = True
    for kind in ("int4_awq", "nvfp4"):
        log(f"\n  --- {kind} on {SMOKE_MODEL} ---")
        try:
            from request import InferOptRequest, build_fingerprint
            fp, _ = build_fingerprint(InferOptRequest(
                model=SMOKE_MODEL, trace=str(calib)))
            path = ensure_variant(fp, kind, str(calib), log=log)
        except Exception as e:
            log(f"  PRODUCE FAILED: {e}")
            ok = False
            continue

        probe = ARTIFACTS / f"load_{kind}.py"
        probe.write_text(
            "import sys\n"
            "from vllm import LLM, SamplingParams\n"
            # 0.15 of a 122GB unified pool is ~18GB -- far more than a 0.6B model
            # needs, and leaves the production OCR server its headroom.
            "llm = LLM(model=sys.argv[1], max_model_len=512, gpu_memory_utilization=0.15,\n"
            "          enforce_eager=True)\n"
            "o = llm.generate(['The capital of France is'], SamplingParams(max_tokens=8))\n"
            "print('[load] OK ->', repr(o[0].outputs[0].text))\n")
        # child_env(), not os.environ: vLLM JIT-builds CUDA extensions and shells
        # out to `ninja`, which lives beside this interpreter. A subprocess
        # inherits only PATH, so a bare env dies deep in engine init with
        # FileNotFoundError: ninja -- which reads like a format problem and is not.
        from evaluator import child_env
        r = subprocess.run([sys.executable, str(probe), path],
                           env=child_env(), capture_output=True, text=True, timeout=1800)
        if "[load] OK" in r.stdout:
            log(f"  vLLM LOADED the {kind} artifact -- format round-trips")
        else:
            blob = (r.stdout or "") + (r.stderr or "")
            # Report the actual exception, not the "see root cause above" wrapper.
            cause = [l for l in blob.splitlines()
                     if any(k in l for k in ("Error:", "error:", "Unsupported", "not supported"))
                     and "core_client" not in l and "launch_core" not in l]
            log(f"  vLLM COULD NOT LOAD the {kind} artifact:")
            log(textwrap.indent("\n".join(cause[-6:]) or blob.strip()[-800:], "    "))
            if "quantization=compressed-tensors" in blob or "quantization=awq" in blob:
                log(f"  -> vLLM RECOGNISED the format, so this is not a version "
                    f"mismatch; read the exception above.")
            else:
                log(f"  -> vLLM did not recognise the format. The producer's "
                    f"compressed-tensors (0.18.0) may be ahead of vLLM's reader "
                    f"(0.17.0); retry with llmcompressor==0.12.0.")
            ok = False
    return ok


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="quantize", description=__doc__.split("\n")[0])
    ap.add_argument("--setup", action="store_true", help="install the isolated producer")
    ap.add_argument("--smoke", action="store_true",
                    help="produce a tiny model and verify vLLM can load it")
    ap.add_argument("--produce", metavar="KIND", choices=["int4_awq", "nvfp4"],
                    help="produce one variant of --model")
    ap.add_argument("--model")
    ap.add_argument("--trace", default="data/trace.jsonl")
    a = ap.parse_args()

    if a.setup:
        return 0 if setup() else 1
    if a.smoke:
        return 0 if smoke() else 1
    if a.produce:
        if not a.model:
            ap.error("--produce needs --model")
        print(ensure_variant(a.model, a.produce, a.trace))
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
