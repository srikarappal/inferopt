"""Prompts go to the server the way a customer's would.

Every measurement here goes through /v1/completions, which applies no chat
template. An instruct model handed a bare question mostly ends its turn at once
or runs to the cap: on DiffusionGemma, 366 of 500 MATH-500 prompts came back
as one token and 69 hit 256. The replay driver sends max_tokens without
ignore_eos, so that shape, not the customer's traffic, was what got measured.

The template is applied HERE, in text, once, and the request still goes through
the same completions path as every other measurement. Nothing about the serving
instrument changes, only the characters in the prompt. Three callers share it
so they cannot disagree: the trace replay (evaluator), the quality gate
(quality) and the control plane's stock response-length pass.

A model with no template is sent its prompts as written, and so is a model
whose tokenizer cannot be loaded: the measurement runs either way.
"""

from __future__ import annotations

import logging

from transformers import AutoTokenizer

log = logging.getLogger(__name__)

_FORMATTERS: dict[str, "ChatFormat | None"] = {}


class ChatFormat:
    """A model's chat template as a str -> str, for one user turn."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        # enable_thinking is Qwen's, whose template otherwise opens a <think>
        # block that changes output length by an order of magnitude. Other
        # templates ignore an unknown kwarg; one that raises on it is asked
        # again without.
        for extra in ({"enable_thinking": False}, {}):
            try:
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, **extra)
            except Exception:
                continue
            return self.without_bos(text)
        return prompt

    def without_bos(self, text: str) -> str:
        """Gemma's template writes <bos> into the text and /v1/completions adds
        one more when it tokenizes; a doubled BOS is a known way to degrade a
        Gemma, so the server's is the one that stays."""
        bos = getattr(self.tokenizer, "bos_token", None)
        if bos and text.startswith(bos):
            return text[len(bos):]
        return text


def chat_formatter(model: str | None) -> ChatFormat | None:
    """The model's template as a ChatFormat, or None when it has none.

    Cached per model: the tokenizer is loaded once per process, not once per
    benchmark row.
    """
    if not model:
        return None
    if model not in _FORMATTERS:
        _FORMATTERS[model] = _load(model)
    return _FORMATTERS[model]


def _load(model: str) -> ChatFormat | None:
    try:
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    except Exception as error:
        log.warning(f"no tokenizer for {model} ({type(error).__name__}); prompts go as written")
        return None
    if not getattr(tokenizer, "chat_template", None):
        return None
    return ChatFormat(tokenizer)


def as_chat_turns(model: str | None, prompts: list[str]) -> list[str]:
    """The prompts as this model's chat turns, or unchanged without a template."""
    formatter = chat_formatter(model)
    if formatter is None:
        return list(prompts)
    return [formatter(prompt) for prompt in prompts]
