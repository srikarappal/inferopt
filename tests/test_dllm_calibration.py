"""A diffusion LM is calibrated through the forward the server runs.
See src/inferopt/dllm_calibration.py."""

from types import SimpleNamespace

import torch

from inferopt import dllm_calibration as cal

VOCAB = 50
CANVAS = 8


class _Out:
    def __init__(self, logits):
        self.logits = logits


class _CanvasModel:
    """DiffusionGemma's shape: prompt in, canvas through a decoder, previous
    logits back in. Confident about even positions, unsure about odd ones."""

    def __init__(self):
        self.config = SimpleNamespace(vocab_size=VOCAB, text_config=None)
        self.calls = []

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None,
                self_conditioning_logits=None, **kwargs):
        self.calls.append({"canvas": decoder_input_ids.clone(),
                           "self_conditioning": self_conditioning_logits is not None})
        logits = torch.zeros(decoder_input_ids.shape[0], decoder_input_ids.shape[1], VOCAB)
        logits[:, ::2, 7] = 50.0          # even slots: certain of token 7
        logits[:, 1::2, :2] = 1.0         # odd slots: a coin between 0 and 1
        return _Out(logits)

    __call__ = forward


class _MaskedModel:
    """LLaDA's shape: one sequence with mask tokens in it, bidirectional."""

    def __init__(self):
        self.config = SimpleNamespace(vocab_size=VOCAB, mask_token_id=49)
        self.calls = []

    def forward(self, input_ids=None, **kwargs):
        self.calls.append(input_ids.clone())
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], VOCAB)
        logits[..., 3] = 5.0
        return _Out(logits)

    __call__ = forward


class _CausalModel:
    def __init__(self):
        self.config = SimpleNamespace(vocab_size=VOCAB)
        self.calls = []

    def forward(self, input_ids=None, **kwargs):
        self.calls.append(input_ids)
        return _Out(torch.zeros(input_ids.shape[0], input_ids.shape[1], VOCAB))

    __call__ = forward


def _prompt():
    return {"input_ids": torch.tensor([[11, 12, 13]]), "attention_mask": torch.ones(1, 3, dtype=torch.long)}


def test_the_style_is_read_off_the_forward_and_the_config():
    assert cal.style_of(_CanvasModel()) == cal.CANVAS
    assert cal.style_of(_MaskedModel()) == cal.MASKED
    assert cal.style_of(_CausalModel()) == cal.CAUSAL


def test_a_canvas_model_is_denoised_with_its_own_logits_fed_back():
    model = _CanvasModel()
    ran = cal.denoise_canvas(model, _prompt(), canvas_length=CANVAS, steps=4,
                             generator=torch.Generator().manual_seed(0))
    assert ran == 4 and len(model.calls) == 4
    assert model.calls[0]["self_conditioning"] is False
    assert all(c["self_conditioning"] for c in model.calls[1:]), "step two onward conditions on step one"
    first, last = model.calls[0]["canvas"], model.calls[-1]["canvas"]
    assert first.shape == (1, CANVAS) and not torch.equal(first, last), "the canvas is refined, not replayed"
    assert torch.all(last[0, ::2] == 7), "confident positions were committed and then frozen"


def test_a_masked_model_is_unmasked_a_few_positions_per_step():
    model = _MaskedModel()
    ran = cal.denoise_masked(model, _prompt(), canvas_length=CANVAS, mask_token_id=49, steps=4)
    assert ran == 4
    masks_per_step = [int((seq[0, 3:] == 49).sum()) for seq in model.calls]
    assert masks_per_step == [8, 6, 4, 2], "two positions leave the mask each step"
    assert torch.all(model.calls[0][0, :3] == torch.tensor([11, 12, 13])), "the prompt is kept in front"


def test_an_autoregressive_model_keeps_the_plain_forward():
    model = _CausalModel()
    notes = []
    assert cal.calibration_forwards(model, [_prompt(), _prompt()], "autoregressive", 0, log=notes.append) == 2
    assert len(model.calls) == 2 and "causal" in notes[0]


def test_a_diffusion_model_runs_steps_forwards_per_prompt():
    model = _CanvasModel()
    notes = []
    ran = cal.calibration_forwards(model, [_prompt(), _prompt()], "diffusion", CANVAS, log=notes.append)
    assert ran == 2 * cal.STEPS and len(model.calls) == 2 * cal.STEPS
    assert "canvas" in notes[0] and "denoising steps" in notes[0]


def test_a_scoring_batch_carries_what_a_plain_call_needs():
    canvas = cal.scoring_batch(_CanvasModel(), _prompt(), "diffusion", CANVAS)
    assert canvas["decoder_input_ids"].shape == (1, CANVAS) and canvas["input_ids"].shape == (1, 3)
    masked = cal.scoring_batch(_MaskedModel(), _prompt(), "diffusion", CANVAS)
    assert masked["input_ids"].shape == (1, 3 + CANVAS) and int((masked["input_ids"] == 49).sum()) == CANVAS
    plain = cal.scoring_batch(_CausalModel(), _prompt(), "autoregressive", 0)
    assert plain is not None and "decoder_input_ids" not in plain


def test_a_diffusion_lm_loads_through_the_class_its_config_names(monkeypatch):
    loaded = []

    class Named:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            loaded.append(("named", model_id))
            return "named-model"
    monkeypatch.setattr(cal.AutoConfig, "from_pretrained",
                        staticmethod(lambda *a, **k: SimpleNamespace(architectures=["FakeForBlockDiffusion"])))
    monkeypatch.setattr(cal.transformers, "FakeForBlockDiffusion", Named, raising=False)
    monkeypatch.setattr(cal.AutoModelForCausalLM, "from_pretrained",
                        staticmethod(lambda model_id, **k: loaded.append(("causal", model_id)) or "causal-model"))
    assert cal.load_model("org/dllm", "diffusion") == "named-model"
    assert cal.load_model("org/ar", "autoregressive") == "causal-model"
    assert loaded == [("named", "org/dllm"), ("causal", "org/ar")]
