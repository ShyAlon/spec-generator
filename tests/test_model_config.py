import pytest

from spec_generator import model_config


def test_ollama_override_routes(monkeypatch):
    monkeypatch.setenv("SPEC_GENERATOR_MODEL_ITEMS", "ollama:qwen2.5-coder:14b")
    cfg = model_config.for_stage("items")
    assert (cfg.provider, cfg.model) == ("ollama", "qwen2.5-coder:14b")


def test_unknown_stage_raises():
    with pytest.raises(KeyError):
        model_config.for_stage("vibes")


def test_default_is_local_ollama(monkeypatch):
    # local-first: no stage defaults to a paid/remote provider
    for stage in ("needs", "srs", "architecture", "items", "tests", "review"):
        monkeypatch.delenv(f"SPEC_GENERATOR_MODEL_{stage.upper()}", raising=False)
        cfg = model_config.for_stage(stage)
        assert cfg.provider == "ollama", stage
