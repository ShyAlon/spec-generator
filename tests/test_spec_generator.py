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
    monkeypatch.setenv("SPEC_GENERATOR_MODEL_NEEDS", "deepseek:deepseek-chat")  # remote is opt-in
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


def test_greenfield_example_passes_current_compat():
    # 2026-09-12: the shipped greenfield template carried contract_version
    # 0.12.0 - _load_system_baseline's compat check would have REJECTED the
    # very file the error messages tell founders to copy.
    import json
    from pathlib import Path

    from spec_generator.cli import _load_system_baseline
    template = Path(__file__).parent.parent / "examples" / "system-baseline.greenfield.json"
    b = _load_system_baseline(template)
    assert b["mode"] == "greenfield" and b["repositories"] == []


def test_main_prints_the_chosen_baseline(capsys):
    # the run must NAME its architecture anchor before the first model call -
    # a baseline that silently belongs to a different system is the
    # grounded-but-wrong failure mode (2026-09-12 review)
    import inspect
    src = inspect.getsource(__import__("spec_generator.cli", fromlist=["main"]).main)
    assert "System baseline:" in src and "baseline_id" in src and "mode=" in src
