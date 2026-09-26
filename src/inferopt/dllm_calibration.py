"""Quantization calibration forwards for a diffusion LM, as the server runs them.

ModelOpt fits activation ranges (the scales FP8 and NVFP4 quantize activations
with) by running the model's forward on sample prompts. For an autoregressive
model that forward, teacher forced over the prompt, IS the prefill forward: the
thing the server runs. A diffusion LM runs something else at every denoising
step, and a causal forward over clean text never produces it:

  canvas style (DiffusionGemma)   the prompt goes through the encoder once; a
                                  canvas of tokens goes through the decoder,
                                  conditioned on the previous step's logits
                                  through a self-conditioning MLP. A plain
                                  forward leaves that MLP at zero, so it gets
                                  no scale at all, and never sees a canvas at
                                  any noise level.
  masked style (LLaDA, Dream)     the prompt plus a block of mask tokens,
                                  bidirectional; the block is unmasked a few
                                  positions per step. Clean text has no mask
                                  token in it.

So a dLLM is calibrated by denoising: the canvas seeded the way vLLM seeds it
(random tokens, or a block of masks), STEPS refinement steps with the
previous logits fed back, positions committed where the model is confident,
temperature falling along the served schedule. Every step is one calibration
forward at a noise level the server actually visits. A served canvas takes 30
to 70 steps; eight spread along the same schedule cover the same range of
noise for a fraction of the cost.
"""

from __future__ import annotations

import inspect

import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM

STEPS = 8
CANVAS = "canvas"
MASKED = "masked"
CAUSAL = "causal"


def load_model(model_id: str, decoding: str = "autoregressive"):
    """The model under its own head. AutoModelForCausalLM has no entry for a
    canvas model (DiffusionGemmaForBlockDiffusion), so a diffusion LM loads
    through the class its config names; a masked dLLM with custom code still
    maps through the causal auto class, which is fine, its forward is the
    same call."""
    kwargs = dict(torch_dtype="auto", device_map="auto", trust_remote_code=True)
    if decoding == "diffusion":
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        for name in getattr(config, "architectures", None) or []:
            cls = getattr(transformers, name, None)
            if cls is not None:
                return cls.from_pretrained(model_id, **kwargs)
    return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)


def style_of(model) -> str:
    """How this model denoises, read off its forward and its config."""
    if "decoder_input_ids" in inspect.signature(model.forward).parameters:
        return CANVAS
    if mask_token_of(model) is not None:
        return MASKED
    return CAUSAL


def mask_token_of(model):
    config = model.config
    for holder in (config, getattr(config, "text_config", None)):
        value = getattr(holder, "mask_token_id", None)
        if value is not None:
            return int(value)
    return None


def vocab_of(model) -> int:
    config = model.config
    return int(getattr(config, "vocab_size", None)
               or getattr(getattr(config, "text_config", None), "vocab_size"))


def _temperature(step: int, steps: int, t_min: float, t_max: float) -> float:
    # vLLM's schedule: hot on the first step, t_min on the last.
    return t_min + (t_max - t_min) * (steps - step) / steps


def denoise_canvas(model, prompt: dict, *, canvas_length: int, steps: int = STEPS,
                   t_min: float = 0.4, t_max: float = 0.8, threshold: float = 0.95,
                   generator=None) -> int:
    """Canvas style: `steps` forwards for one prompt, the previous step's
    logits fed back as self-conditioning, positions frozen once their argmax
    clears `threshold` (BlockRefinementScheduler's rule), the rest rewritten
    with the current guess. Returns the forwards run."""
    ids = prompt["input_ids"]
    canvas = torch.randint(0, vocab_of(model), (ids.shape[0], canvas_length),
                           device=ids.device, generator=generator)
    committed = torch.zeros_like(canvas, dtype=torch.bool)
    previous = None
    for step in range(steps):
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=prompt.get("attention_mask"),
                        decoder_input_ids=canvas, self_conditioning_logits=previous)
        logits = out.logits.float()
        probs = torch.softmax(logits / _temperature(step, steps, t_min, t_max), dim=-1)
        confidence, guess = probs.max(dim=-1)
        canvas = torch.where(committed, canvas, guess)
        committed |= confidence >= threshold
        previous = logits
    return steps


def denoise_masked(model, prompt: dict, *, canvas_length: int, mask_token_id: int,
                   steps: int = STEPS) -> int:
    """Masked style: the prompt plus a block of masks, unmasked a fixed share
    of positions per step in confidence order, the way LLaDA and Dream decode
    a block. Returns the forwards run."""
    ids = prompt["input_ids"]
    block = torch.full((ids.shape[0], canvas_length), mask_token_id,
                       dtype=ids.dtype, device=ids.device)
    seq = torch.cat([ids, block], dim=1)
    per_step = max(1, canvas_length // steps)
    start = ids.shape[1]
    for _ in range(steps):
        with torch.no_grad():
            out = model(input_ids=seq)
        logits = out.logits[:, start:, :].float()
        confidence, guess = torch.softmax(logits, dim=-1).max(dim=-1)
        still_masked = seq[:, start:] == mask_token_id
        if not still_masked.any():
            break
        confidence = torch.where(still_masked, confidence, torch.full_like(confidence, -1.0))
        chosen = confidence.topk(min(per_step, int(still_masked.sum(dim=1).min())), dim=1).indices
        block = seq[:, start:].clone()
        block.scatter_(1, chosen, guess.gather(1, chosen))
        seq = torch.cat([ids, block], dim=1)
    return steps


def calibration_forwards(model, batches: list[dict], decoding: str, canvas_length: int,
                         log=print) -> int:
    """Run the calibration forwards for `batches` and return how many ran.
    An autoregressive model gets the plain forward; a diffusion LM is denoised
    per prompt in the style its architecture decodes."""
    style = style_of(model) if decoding == "diffusion" else CAUSAL
    if decoding == "diffusion" and style == CAUSAL:
        log("[job] diffusion LM with neither a canvas decoder nor a mask token; "
            "calibrating with the plain forward")
    log(f"[job] calibration forward: {style}" + (f", {STEPS} denoising steps per prompt"
                                                if style != CAUSAL else ""))
    ran = 0
    for index, batch in enumerate(batches):
        if style == CANVAS:
            ran += denoise_canvas(model, batch, canvas_length=canvas_length)
        elif style == MASKED:
            ran += denoise_masked(model, batch, canvas_length=canvas_length,
                                  mask_token_id=mask_token_of(model))
        else:
            with torch.no_grad():
                model(**batch)
            ran += 1
        if (index + 1) % 32 == 0:
            log(f"[job] calibrated {index + 1}/{len(batches)} prompts")
    return ran


def scoring_batch(model, batch: dict, decoding: str, canvas_length: int) -> dict:
    """One batch a plain `model(**batch).logits` call can score, for
    AutoQuantize's KL step: a canvas model needs a canvas to decode, a masked
    one a block of masks after the prompt. First-step inputs only; the KL
    ranking of layers does not need the whole schedule."""
    style = style_of(model) if decoding == "diffusion" else CAUSAL
    ids = batch["input_ids"]
    if style == CANVAS:
        return {**batch, "decoder_input_ids": torch.randint(
            0, vocab_of(model), (ids.shape[0], canvas_length), device=ids.device)}
    if style == MASKED:
        block = torch.full((ids.shape[0], canvas_length), mask_token_of(model),
                           dtype=ids.dtype, device=ids.device)
        return {"input_ids": torch.cat([ids, block], dim=1)}
    return batch
