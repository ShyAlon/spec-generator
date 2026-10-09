from spec_generator.skeleton import normalize_generated_ears_statement,merge_srs_fills
from spec_generator.graph import _slot_fill_validator
from vmodel_contracts.text_qa import srs_draft_findings

TEXT='The Virtual DM shall, on every turn of a session, describe the current situation.'

def test_generation_normalizes_before_partial_and_artifact_storage():
    fill={'title':'Turn loop','statement':TEXT,'priority':'must','acceptance':['Respond within 200 ms.']}
    checked=_slot_fill_validator('R-1')(fill)
    assert 'shall,' not in checked['statement'] and fill['statement']==TEXT
    assert checked['acceptance']==fill['acceptance']
    slots=[{'requirement_id':'R-1','kind':'functional','need_ids':['N-1'],'intent_fields':[]}]
    draft=merge_srs_fills(slots,{'fills':{'R-1':fill}})
    assert draft['functional'][0]['statement']==checked['statement']
    assert not [f for f in srs_draft_findings(draft,'Virtual DM on every turn of a session describe situation 200 ms') if f['rule']=='ears_pattern']

def test_normalizer_is_minimal_idempotent_and_does_not_rewrite_numbers():
    text='  When ready,  the system shall, report 200 ms and 0.2 s. '
    normalized=normalize_generated_ears_statement(text)
    assert normalized=='When ready, the system shall report 200 ms and 0.2 s.'
    assert normalize_generated_ears_statement(normalized)==normalized
    assert normalize_generated_ears_statement('The system should respond.')=='The system should respond.'

def test_capitalization_is_preserved():
    assert normalize_generated_ears_statement('The system SHALL, respond.')=='The system SHALL respond.'
