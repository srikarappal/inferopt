"""Generate docs/Audio-and-World-Models.docx -- a literature snapshot.

    python docs/make_audio_world_report.py

DIFFERENT IN KIND FROM make_consolidated_report.py, and the difference matters.
That one reads runs/ and cannot drift from the measurements because it never
transcribes a number. This one transcribes published results we have NOT
reproduced. Every figure in it is a claim by its authors, attributed and dated,
and none of it is evidence about this hardware or these workloads.

It exists because the question "can inferopt do audio, or world models" came up,
the answer was no, and the interesting part was WHY -- the techniques are
plentiful and published, while the thing that is missing is a serving structure
this project already has, pointed at a different objective. That is worth
keeping.

Snapshot date is recorded in the document. A literature summary with no date is
a liability.

The helpers come from make_consolidated_report so the two documents cannot
drift apart in style; this project has paid for duplicated definitions before.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from docx import Document
from docx.shared import Pt

from make_consolidated_report import bullets, para, table

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "Audio-and-World-Models.docx"
SNAPSHOT = "September 2026"


def main() -> int:
    doc = Document()
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(10.5)

    doc.add_heading("Optimized Inference for Audio and World Models", level=0)
    para(doc, "What is published, what is in production serving today, and what "
              "inferopt would have to become to cover it", italic=True, size=12)
    para(doc, f"Literature snapshot, {SNAPSHOT}. NOTHING IN THIS DOCUMENT HAS "
              f"BEEN REPRODUCED HERE. Every number is a claim by its authors, "
              f"attributed below. Read it as a map of the territory, not as "
              f"evidence.", size=9)

    # ------------------------------------------------------------- why
    doc.add_heading("Why this document exists", 1)
    para(doc,
         "inferopt optimizes decode-time serving of autoregressive text "
         "transformers. That is not a stated scope, it is what the code does, "
         "and the boundary is sharper than it looks:")
    bullets(doc, [
        "The load driver posts a text `prompt` to /v1/completions. There is no "
        "path by which an image, an audio clip or a video frame reaches the "
        "server.",
        "The trace schema carries input_tokens, output_tokens, arrival_ts, "
        "prefix_id, adapter_id and temperature. Nothing describes a non-text "
        "input.",
        "is_multimodal IS detected from the HF config and stored on the "
        "fingerprint -- and used for exactly one thing: excluding the vision "
        "tower from quantization in quantize.py. No DAG node gates on it.",
        "dag/vlm.json is named in a fingerprint docstring as owning multimodal "
        "tuning. It does not exist.",
        "Both quality benchmarks are text: MATH-500 and MBPP+.",
    ])
    para(doc,
         "So pointing inferopt at a VLM or a speech model would produce a "
         "plausible-looking wrong answer rather than an error: it would serve "
         "the model, drive it with text prompts, and tune for a workload the "
         "model never sees. Image tokens dominate VLM prefill; audio codec "
         "tokens dominate a speech model's decode. Neither is measured.",
         size=9.5, italic=True)

    # ------------------------------------------------------------- audio
    doc.add_page_break()
    doc.add_heading("Part 1 -- Audio", 1)

    doc.add_heading("The shape of the problem", 2)
    para(doc,
         "A speech language model is a PIPELINE, not a model. The dominant "
         "pattern is two-stage: a Talker predicts codec tokens "
         "autoregressively, and a Code2Wav module reconstructs the waveform "
         "from them. Some models (VoxCPM2) collapse this into a single-stage "
         "hybrid where several components execute inside one instance.")
    para(doc,
         "The two stages have MISMATCHED COMPUTE PROFILES -- the Talker is "
         "latency-bound and the reconstruction is throughput-bound -- which is "
         "the root of most of the systems difficulty below.")

    doc.add_heading("The metric changes, and this is the deepest difference", 2)
    table(doc, ["", "text LLM (what inferopt measures)", "speech LM"], [
        ["first-token latency", "TTFT -- prefill to first token",
         "TTFA (time-to-first-AUDIO) -- prefill, plus generating N codec "
         "tokens, plus a pass through the audio detokenizer"],
        ["steady state", "ITL p99, a percentile to minimize",
         "STREAMING VIABILITY -- a BINARY constraint: each audio chunk must "
         "arrive before playback of the previous chunk ends"],
        ["objective", "goodput: tokens from SLO-satisfying requests",
         "no settled equivalent; viability is a deadline, not a percentile"],
    ], widths=[1.1, 2.3, 3.0])
    para(doc,
         "VoxServe states this directly: text-focused systems “optimize for "
         "the wrong metrics” for speech. A percentile target and a playback "
         "deadline are different mathematics -- a config can hold a 250 ms ITL "
         "p99 and still stutter, because the tail lands inside one chunk "
         "boundary.", size=9.5)

    doc.add_heading("What production OSS serving does today", 2)
    para(doc, "vLLM-Omni, June 2026. Model-specific techniques rather than "
              "universal recipes:", size=9.5)
    bullets(doc, [
        "stage separation and connector chunking, decoupling Talker latency "
        "from Code2Wav throughput",
        "batched decode preprocessing, to cut per-request Python overhead",
        "torch.compile over whole-forward graphs",
        "CFM/LocDiT decode-tail batching -- batching diffusion calls across "
        "requests",
        "GPU-resident decode state, moving multi-codebook updates off the CPU",
        "model-specific attention kernels (e.g. a q_len=1 Triton kernel)",
        "CUDA Graphs with dynamic batch adaptation",
    ])
    para(doc, "Named as NOT covered:", bold=True, size=9.5)
    bullets(doc, [
        "speculative decoding or early exit",
        "cross-request caching",
        "multi-GPU disaggregation",
        "quantization beyond fp32/fp16 alignment",
    ])

    doc.add_heading("Published, not in production serving", 2)
    table(doc, ["technique", "idea", "claimed gain"], [
        ["VADUSA", "multi-token prediction heads plus a Viterbi-based "
                   "speculative decode selecting the best token sequence per step",
         "4-5x per-token, minimal quality cost"],
        ["SSD", "lightweight draft model made by fine-tuning a few parameters "
                "OF THE TARGET, so no separate draft model is trained", "1.4x"],
        ["SpecASR", "tree-structured speculation raising the verification "
                    "success rate, for LLM-based ASR", "reported acceleration"],
        ["TLDR", "compressing audio tokens themselves, shortening the AR "
                 "sequence rather than speeding each step", "-"],
    ], widths=[0.9, 3.7, 1.3])
    para(doc,
         "Speculative decoding is the notable absence. It is the single "
         "largest lossless win in this project's own MoE result (+42% goodput "
         "from ngram spec decode alone), it has 1.4-5x published gains for "
         "speech, and no production speech server implements it. The obstacle "
         "is structural rather than lack of interest: codec tokens are "
         "multi-codebook with delay patterns, so the draft/verify contract is "
         "not the one a text decoder uses.", size=9.5)

    # ------------------------------------------------------------- world
    doc.add_page_break()
    doc.add_heading("Part 2 -- World models", 1)
    para(doc,
         "Read “world model” here as autoregressive video diffusion: a "
         "causal DiT generating frame blocks with few-step denoising and a "
         "rolling KV cache. This is the shape used by driving and interactive "
         "world models, and it is genuinely closer to an LLM than image "
         "diffusion is -- it is autoregressive and it has a KV cache -- while "
         "still not being servable by an LLM server.")

    doc.add_heading("Three families of acceleration", 2)
    table(doc, ["family", "what it exploits", "examples and claims"], [
        ["feature / block caching",
         "temporal redundancy across denoising steps and across chunks: "
         "features change little, so recompute little",
         "X-Cache: 71% block skip, 2.6x, on a PRODUCTION multi-camera driving "
         "world model. WorldCache (heterogeneous token caching), LightCache "
         "(training-free, memory-efficient)"],
        ["KV cache compression",
         "not every past frame's KV matters to the next one",
         "Forcing-KV (hybrid compression), Focused Forcing (content-aware "
         "per-frame KV selection), Future Forcing (training-free policy)"],
        ["sparse attention",
         "attention mass concentrates in local 3D spatio-temporal windows",
         "Sliding Tile Attention: 2.98x over FlashAttention-2, 1.89x over "
         "FlashAttention-3. Radial Attention: O(n log n) with energy decay. "
         "LiteAttention: propagates skippable tiles across timesteps"],
    ], widths=[1.1, 1.9, 3.4])
    para(doc,
         "The enabling empirical fact, and the one worth remembering: in "
         "autoregressive video diffusion, retaining only 30% of attention "
         "computations preserves more than 85% of the attention mass. That is "
         "the same kind of headroom that made KV quantization and speculative "
         "decoding worth the trouble for text.", size=9.5, italic=True)

    # ------------------------------------------------------------- gaps
    doc.add_page_break()
    doc.add_heading("Part 3 -- Where the gaps are", 1)
    para(doc,
         "The techniques are plentiful and published. What is missing in both "
         "domains is SERVING STRUCTURE, and it is structure this project "
         "already has, aimed at a different objective.")

    table(doc, ["gap", "what is missing", "does inferopt have the shape for it?"], [
        ["the objective",
         "goodput is defined on tokens meeting TTFT/ITL. Audio needs TTFA plus "
         "a binary chunk deadline; world models need frames/s against a "
         "real-time budget.",
         "Partly. The SLO is two fields and the objective is one scalar; both "
         "would need to become pluggable. The Pareto machinery is agnostic."],
        ["pipeline scheduling",
         "both domains are multi-stage with mismatched profiles. vLLM and "
         "SGLang schedule ONE model. VoxServe's core claim is that no entity "
         "coordinates resources across the stages.",
         "No. inferopt tunes a single served model and would have to model a "
         "pipeline as the thing under search. This is the largest gap and it "
         "is a serving problem, not a kernel problem."],
        ["caching has no interface",
         "feature caching is to diffusion what KV cache is to LLMs, but every "
         "method is bespoke per model -- there is no flag to flip and measure.",
         "Yes, if a server exposed one. A cache policy is exactly a DAG node "
         "with a sweep."],
        ["sparsity has no quality dial",
         "STA / Radial / LiteAttention are real speedups with real quality "
         "cost. Servers expose an attention BACKEND choice, not a sparsity "
         "POLICY with a tunable level.",
         "Yes. This is precisely the lossless/lossy axis the DAG models, and "
         "nobody has built it for these domains."],
        ["quantization below the LLM",
         "vocoders, VAEs and codec heads sit outside the modelopt toolchain.",
         "Partly. quantize.py already EXCLUDES vision towers by name; the same "
         "exclusion logic applies, but nothing quantizes those components."],
        ["no accuracy probe",
         "no serving harness carries round-trip WER or UTMOS for speech, or "
         "FVD for video, so “did this optimization hurt” is unanswerable.",
         "Yes, structurally. Benchmark is already an interface with a judge "
         "and a metric; these are new implementations, not new machinery."],
    ], widths=[1.1, 2.5, 2.8])

    doc.add_heading("If this were to be built, in order", 2)
    bullets(doc, [
        "A perceptual benchmark first, before any optimization. Every lesson "
        "in this project's history says the probe comes first: RULER was "
        "removed after it moved across a node that cannot change quality, and "
        "a 4-bit MoE result still reads accuracy IMPROVING because the sample "
        "size cannot resolve the loss. An audio ladder with no WER probe would "
        "repeat both mistakes at higher cost.",
        "Then the SLO and objective, made pluggable -- TTFA and a chunk "
        "deadline are not a percentile, and forcing them into one would give "
        "confidently wrong answers rather than errors.",
        "Then the trace schema, which must carry audio or frame inputs before "
        "any measurement means anything.",
        "Only then the DAG nodes: cache policy, sparsity level, codec "
        "quantization. These are the cheap part, and doing them first would "
        "produce numbers nobody could check.",
    ])

    # ------------------------------------------------------------- sources
    doc.add_page_break()
    doc.add_heading("Sources", 1)
    para(doc, f"Retrieved {SNAPSHOT}. None of these results has been "
              f"reproduced on this hardware.", size=9, italic=True)
    table(doc, ["work", "url"], [
        ["VoxServe -- streaming-centric serving for speech LMs",
         "https://arxiv.org/html/2602.00269"],
        ["Engineering TTS Inference in vLLM-Omni",
         "https://vllm.ai/blog/2026-06-23-vllm-omni-tts"],
        ["VADUSA -- multi-token prediction + speculative decoding for TTS",
         "https://arxiv.org/abs/2410.13839"],
        ["Speech Speculative Decoding (SSD)",
         "https://arxiv.org/html/2505.15380"],
        ["SpecASR -- speculative decoding for LLM-based ASR",
         "https://arxiv.org/pdf/2507.18181"],
        ["TLDR -- compressing audio tokens for AR TTS",
         "https://arxiv.org/html/2606.09019v1"],
        ["X-Cache -- cross-chunk block caching for world models",
         "https://arxiv.org/html/2604.20289"],
        ["WorldCache -- heterogeneous token caching",
         "https://arxiv.org/pdf/2603.06331"],
        ["Forcing-KV -- hybrid KV compression for AR video diffusion",
         "https://arxiv.org/pdf/2605.09681"],
        ["Sliding Tile Attention", "https://arxiv.org/html/2502.04507v3"],
        ["LiteAttention -- temporal sparse attention for DiTs",
         "https://arxiv.org/pdf/2511.11062"],
        ["Sparse Forcing -- trainable sparse attention, real-time AR video",
         "https://arxiv.org/html/2604.21221v1"],
        ["A Survey on Cache Methods in Diffusion Models",
         "https://arxiv.org/pdf/2510.19755"],
    ], widths=[3.2, 3.3])

    doc.save(OUT)
    print(f"  wrote {OUT}  ({OUT.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
