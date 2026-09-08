import pytest


# ---- error handling ----

def test_json_parse_error_names_stage():
    import json as j

    from spec_generator.errors import translate_json_error

    with pytest.raises(Exception) as ei:
        from spec_generator.graph import _json
        _json("not json at all", "needs")
    assert "needs" in str(ei.value)


def test_preflight_names_missing_remote_key(monkeypatch):
    from spec_generator import model_config
    from spec_generator.errors import UserFacingError

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(UserFacingError) as ei:
        model_config.build_chat_model("needs")
    assert "DEEPSEEK_API_KEY" in ei.value.render() and "SPEC_GENERATOR_MODEL_NEEDS" in ei.value.render()


def test_ollama_down_and_unpulled_are_friendly():
    import httpx

    from spec_generator.errors import translate_llm_error

    e = translate_llm_error(httpx.ConnectError("refused"), "needs", "ollama", "qwen2.5-coder:7b", "http://localhost:11434/v1")
    assert "ollama serve" in e.render()

    exc = type("NotFoundError", (Exception,), {})("404 model not found")
    e = translate_llm_error(exc, "needs", "ollama", "qwen2.5-coder:7b", "http://localhost:11434/v1")
    assert "ollama pull" in e.render()


def test_missing_intent_file_is_friendly(tmp_path, capsys):
    import sys

    from spec_generator.cli import main
    from spec_generator.errors import run_cli

    sys.argv = ["prog", str(tmp_path / "nope.json")]
    with pytest.raises(SystemExit) as ei:
        run_cli(main)
    assert ei.value.code == 1
    err = capsys.readouterr().err
    assert "not found" in err and "interview" in err


def test_old_contract_intent_fails_loudly(tmp_path, capsys):
    import json as j
    import sys

    from spec_generator.cli import main
    from spec_generator.errors import run_cli

    intent = {"contract_version": "0.1.0", "session_id": "old-1", "problem_statement": "p"}
    f = tmp_path / "old_intent.json"
    f.write_text(j.dumps(intent))
    sys.argv = ["prog", str(f)]
    with pytest.raises(SystemExit):
        run_cli(main)
    err = capsys.readouterr().err
    assert "0.1.0" in err and "re-run the interview" in err
