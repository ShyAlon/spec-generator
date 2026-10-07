"""Historical architecture state is rejected, never healed in memory or on disk."""
import copy
import json
import pytest
from spec_generator.graph import validated_architecture
from spec_generator.errors import UserFacingError
from test_decision_drivers import ARCH, _arch


@pytest.mark.parametrize('change', ['version', 'driver', 'options', 'maps'])
def test_checkpoint_blocker_preserves_payload_and_artifact_bytes(tmp_path, change):
    payload = _arch(ARCH).model_dump(mode='json')
    if change == 'version':payload['contract_version'] = '0.19.0'
    if change == 'driver':del payload['decisions'][0]['driver']
    if change == 'options':del payload['decisions'][0]['options_considered']
    if change == 'maps':payload['stack'] = {'framework':'React'}
    before = copy.deepcopy(payload)
    artifact = tmp_path/'architecture.json'
    artifact.write_text(json.dumps(payload))
    bytes_before = artifact.read_bytes()
    with pytest.raises(UserFacingError, match='automatic healing is disabled') as error:
        validated_architecture(payload)
    assert 'migrate' in error.value.render() and 'checkpoint' in error.value.render()
    assert payload == before and artifact.read_bytes() == bytes_before


def test_current_architecture_is_validated_without_changing_identity():
    payload = _arch(ARCH).model_dump(mode='json')
    before = copy.deepcopy(payload)
    assert validated_architecture(payload).model_dump(mode='json') == payload
    assert payload == before
