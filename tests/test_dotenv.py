"""Repo-local .env loading (written by vmodel-spec-app/scripts/switch-model.sh):
file fills unset vars, real environment always wins, loaded once."""
import spec_generator.model_config as mc


def _fresh(monkeypatch, tmp_path, content):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(content)
    for v in ("SPEC_GENERATOR_MODEL_SRS", "SPEC_GENERATOR_MODEL_ITEMS"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr(mc, "_DOTENV", None)


def test_dotenv_applies_stage_overrides(monkeypatch, tmp_path):
    _fresh(monkeypatch, tmp_path, "SPEC_GENERATOR_MODEL_SRS=deepseek:deepseek-v4-flash\n")
    assert mc.for_stage("srs").model == "deepseek-v4-flash"
    assert mc.for_stage("items").model == "qwen2.5-coder:14b"  # untouched default


def test_real_environment_wins_over_dotenv(monkeypatch, tmp_path):
    _fresh(monkeypatch, tmp_path, "SPEC_GENERATOR_MODEL_SRS=deepseek:deepseek-v4-flash\n")
    monkeypatch.setenv("SPEC_GENERATOR_MODEL_SRS", "ollama:qwen2.5-coder:7b")
    assert mc.for_stage("srs").model == "qwen2.5-coder:7b"


def test_missing_or_malformed_dotenv_is_inert(monkeypatch, tmp_path):
    _fresh(monkeypatch, tmp_path, "# comment\nNO_EQUALS_SIGN\n\n")
    assert mc.for_stage("srs").provider == "ollama"
    (tmp_path / ".env").unlink()
    monkeypatch.setattr(mc, "_DOTENV", None)
    assert mc.for_stage("srs").provider == "ollama"
