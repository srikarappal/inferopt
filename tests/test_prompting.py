"""Prompts reach the server as the model's chat turn, everywhere, or as written
when the model has no template. See src/inferopt/prompting.py for why."""

import json

import pytest

from inferopt import prompting, quality
from inferopt.evaluator import VllmEvaluator
from test_dag_unit import _ctx


class _Tokenizer:
    def __init__(self, chat_template="{{ messages }}", bos_token=None, raises_on_extra=False):
        self.chat_template = chat_template
        self.bos_token = bos_token
        self.raises_on_extra = raises_on_extra

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **extra):
        if extra and self.raises_on_extra:
            raise TypeError("unknown kwarg")
        assert not tokenize and add_generation_prompt
        return f"{self.bos_token or ''}<user>{messages[0]['content']}</user><assistant>"


@pytest.fixture(autouse=True)
def fresh_cache():
    prompting._FORMATTERS.clear()
    yield
    prompting._FORMATTERS.clear()


def test_a_prompt_becomes_one_user_turn():
    assert prompting.ChatFormat(_Tokenizer())("2+2?") == "<user>2+2?</user><assistant>"


def test_the_templates_own_bos_is_dropped_because_the_server_adds_one():
    """Gemma writes <bos> into the text and /v1/completions tokenizes with
    add_special_tokens; a doubled BOS degrades the model."""
    assert prompting.ChatFormat(_Tokenizer(bos_token="<bos>"))("q") == "<user>q</user><assistant>"


def test_a_template_that_refuses_the_thinking_switch_is_asked_again_without_it():
    assert prompting.ChatFormat(_Tokenizer(raises_on_extra=True))("q") == "<user>q</user><assistant>"


def test_a_model_without_a_template_gets_its_prompts_as_written(monkeypatch):
    monkeypatch.setattr(prompting.AutoTokenizer, "from_pretrained",
                        staticmethod(lambda *a, **k: _Tokenizer(chat_template=None)))
    assert prompting.chat_formatter("org/base") is None
    assert prompting.as_chat_turns("org/base", ["a", "b"]) == ["a", "b"]


def test_a_tokenizer_that_cannot_load_does_not_stop_the_measurement(monkeypatch):
    def refuse(*a, **k):
        raise OSError("offline")
    monkeypatch.setattr(prompting.AutoTokenizer, "from_pretrained", staticmethod(refuse))
    assert prompting.as_chat_turns("org/m", ["a"]) == ["a"]
    assert prompting.chat_formatter(None) is None


def test_the_tokenizer_is_loaded_once_per_model(monkeypatch):
    loads = []
    monkeypatch.setattr(prompting.AutoTokenizer, "from_pretrained",
                        staticmethod(lambda model, **k: loads.append(model) or _Tokenizer()))
    prompting.as_chat_turns("org/m", ["a"])
    prompting.as_chat_turns("org/m", ["b"])
    assert loads == ["org/m"]


def test_every_benchmark_goes_through_the_template(monkeypatch):
    """MATH-500 used to be sent raw on the grounds that it was fine; on
    DiffusionGemma 366 of 500 prompts came back as one token."""
    monkeypatch.setattr(prompting.AutoTokenizer, "from_pretrained",
                        staticmethod(lambda *a, **k: _Tokenizer()))
    assert all(b.chat for b in quality.BENCHMARKS.values())
    prompt = quality._chat_wrapper(lambda row: row["problem"], "org/m")
    assert prompt({"problem": "2+2?"}) == "<user>2+2?</user><assistant>"
    assert quality._chat_wrapper(lambda row: row["problem"], None)({"problem": "x"}) == "x"


def _trace(tmp_path):
    path = tmp_path / "trace.jsonl"
    rows = [{"prompt": "2+2?", "input_tokens": 5, "output_tokens": 8},
            {"prompt": "3+3?", "input_tokens": 5, "output_tokens": 8}]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return str(path)


def test_the_replay_sends_the_trace_prompts_as_chat_turns(tmp_path, monkeypatch):
    monkeypatch.setattr(prompting.AutoTokenizer, "from_pretrained",
                        staticmethod(lambda *a, **k: _Tokenizer()))
    ctx = _ctx()
    ev = VllmEvaluator(ctx.fingerprint, ctx.slo, _trace(tmp_path), str(tmp_path / "run"),
                       log=lambda *a: None)
    assert ev.prompts == ["<user>2+2?</user><assistant>", "<user>3+3?</user><assistant>"]


def test_a_trace_of_real_traffic_is_replayed_as_written(tmp_path, monkeypatch):
    """chat_prompts=False: the customer's text is already what they send."""
    monkeypatch.setattr(prompting.AutoTokenizer, "from_pretrained",
                        staticmethod(lambda *a, **k: _Tokenizer()))
    ctx = _ctx()
    ev = VllmEvaluator(ctx.fingerprint, ctx.slo, _trace(tmp_path), str(tmp_path / "run"),
                       log=lambda *a: None, chat_prompts=False)
    assert ev.prompts == ["2+2?", "3+3?"]
