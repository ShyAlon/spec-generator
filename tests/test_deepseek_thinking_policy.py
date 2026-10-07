"""Client-only regression: no model calls."""
import pytest
from spec_generator import model_config


@pytest.mark.parametrize('structured', [True, False])
@pytest.mark.parametrize('stage', ['srs', 'architecture', 'items', 'tests', 'validation', 'review'])
def test_deepseek_disables_thinking_for_every_call_shape(monkeypatch, stage, structured):
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'test-key')
    monkeypatch.setenv(f'SPEC_GENERATOR_MODEL_{stage.upper()}', 'deepseek:deepseek-flash')
    model = model_config.build_chat_model(stage, structured=structured)
    assert model.extra_body == {'thinking': {'type': 'disabled'}}
