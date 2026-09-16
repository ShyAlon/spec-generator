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
        assert cfg.model == "qwen2.5-coder:14b", stage  # his standing default; 7b is the lighter fallback


def test_deepseek_debug_http_wires_payload_capture(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("SPEC_GENERATOR_MODEL_SRS", "deepseek:deepseek-flash")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "x")
    monkeypatch.setenv("VMODEL_DEBUG_HTTP", "1")
    monkeypatch.setenv("VMODEL_LLM_LOG", str(tmp_path / "llm-calls.jsonl"))
    model = model_config.build_chat_model("srs", structured=True)
    assert model.http_client is not None
    assert model.http_client.event_hooks["request"]
    assert model.http_client.event_hooks["response"]
    shown = capsys.readouterr().err
    assert str(tmp_path / "debug-http.jsonl") in shown
    assert "request messages included" in shown


def test_deepseek_warnings_are_once_per_run(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("SPEC_GENERATOR_MODEL_SRS", "deepseek:deepseek-flash")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "x")
    monkeypatch.setenv("VMODEL_DEBUG_HTTP", "1")
    monkeypatch.setenv("VMODEL_LLM_LOG", str(tmp_path / "llm-calls.jsonl"))
    monkeypatch.setattr(model_config, "_OUTPUT_MODE_ANNOUNCED", set())
    monkeypatch.setattr(model_config, "_DEBUG_HTTP_ANNOUNCED", set())
    model_config.build_chat_model("srs", structured=True)
    model_config.build_chat_model("srs", structured=True)
    shown=capsys.readouterr().err
    assert shown.count("output mode: DeepSeek forced function call") == 1
    assert shown.count("WARNING full LLM HTTP debug enabled:") == 1


def test_self_healing_gate_revision_is_warning_not_error():
    from spec_generator.model_config import _console_level
    assert _console_level("WARNING architecture gates requested revisions - re-planning once") == "warning"
    assert _console_level("WARNING contract revision requested - retrying once") == "warning"
    assert _console_level("architecture still failed after repair") == "error"
