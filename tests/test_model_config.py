import pytest

from spec_generator import model_config


def test_ollama_override_routes(monkeypatch):
    monkeypatch.setenv("SPEC_GENERATOR_MODEL_ITEMS", "ollama:qwen2.5-coder:14b")
    cfg = model_config.for_stage("items")
    assert (cfg.provider, cfg.model) == ("ollama", "qwen2.5-coder:14b")


def test_unknown_stage_raises():
    with pytest.raises(KeyError):
        model_config.for_stage("vibes")
