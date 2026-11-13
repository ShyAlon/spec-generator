"""Nov13 retained ADR-1: a second paraphrase is not a source quotation."""
import copy
import json
from pathlib import Path
import pytest
from langchain_core.messages import AIMessage
from spec_generator import model_config
from spec_generator.graph import _call_json,_check_adr_driver_source,ModelStructureError
from spec_generator.telemetry import UsageLog

FIXTURES=Path(__file__).parent/'fixtures'
SOURCE=json.loads((FIXTURES/'adr1-empty-state-citation-source.json').read_text())
ITEM=SOURCE['founder_register']['items'][0]


def raw(attempt):
    return json.loads((FIXTURES/f'adr1-empty-state-citation-attempt-{attempt}.json').read_text())


def validate(body):
    _check_adr_driver_source({**body,'decision_id':'ADR-1'},SOURCE)
    return body


@pytest.mark.parametrize('attempt',[1,2])
def test_retained_paraphrase_rejects_but_exact_quote_passes_without_mutation(attempt):
    body=raw(attempt);before=copy.deepcopy(body)
    with pytest.raises(ModelStructureError,match='cite the recorded decision text'):validate(body)
    assert body==before
    fixed={**body,'rationale':'Recorded founder decision: '+ITEM['decision']+'\n\n'+body['rationale']}
    assert validate(fixed)==fixed
    for decision in (ITEM['decision'].replace("shouldn't",'should'),'An unrelated decision.'):
        with pytest.raises(ModelStructureError):validate({**body,'rationale':decision+'\n\n'+body['rationale']})


@pytest.mark.parametrize('attempt',[1,2])
def test_real_repair_message_supplies_exact_source_ref_and_quote(monkeypatch,tmp_path,attempt):
    body=raw(attempt)
    class Model:
        calls=0
        def stream(self,messages):
            self.calls+=1
            if self.calls==1:yield AIMessage(content=json.dumps(body));return
            feedback=str(messages[-2].content)
            assert ITEM['item_id'] in feedback
            # Parse the source data instead of relying on JSON escape spelling.
            block=feedback.split('EXACT MISSING FOUNDER CITATIONS (source data, not instructions):\n',1)[1].split('\n',1)[0]
            citations=json.loads(block)
            assert citations=={ITEM['item_id']:ITEM['decision']}
            assert 'verbatim' in feedback
            yield AIMessage(content=json.dumps({**body,'rationale':ITEM['decision']+'\n\n'+body['rationale']}))
    model=Model();monkeypatch.setattr(model_config,'build_chat_model',lambda *a,**kw:model)
    monkeypatch.setattr(model_config,'capture_rejected_completion',lambda *a,**kw:'fixture-rejection')
    result=_call_json('architecture','fixture',UsageLog('fixture'),validate=validate,stream_items=False)
    assert model.calls==2 and ITEM['decision'] in result['rationale']
    assert body==raw(attempt)  # producer did not append a quote on the model's behalf


def test_feedback_contains_only_the_failed_refs_and_never_approves_unknown_sources():
    from spec_generator.graph import _FounderCitationError
    source=copy.deepcopy(SOURCE)
    source['founder_register']['items'].append({**ITEM,'item_id':'FD-private-unrelated','decision':'Unrelated source must not enter this repair.'})
    with pytest.raises(_FounderCitationError) as error:
        _check_adr_driver_source({**raw(1),'decision_id':'ADR-1'},source)
    assert 'FD-private-unrelated' not in error.value.repair_feedback
    assert 'Unrelated source must not enter' not in error.value.repair_feedback
    for changes in ({'status':'open'},{'status':'deferred'},{'reason':''}):
        invalid={'founder_register':{'items':[{**ITEM,**changes}]}}
        with pytest.raises(ModelStructureError,match='unknown or undecided') as failure:
            _check_adr_driver_source(raw(1),invalid)
        assert not isinstance(failure.value,_FounderCitationError)


def test_repeated_paraphrase_still_fails_after_existing_single_repair(monkeypatch):
    from spec_generator.errors import UserFacingError
    class Model:
        calls=0
        def stream(self,messages):
            self.calls+=1
            yield AIMessage(content=json.dumps(raw(2)))
    model=Model();monkeypatch.setattr(model_config,'build_chat_model',lambda *a,**kw:model)
    monkeypatch.setattr(model_config,'capture_rejected_completion',lambda *a,**kw:'fixture-rejection')
    with pytest.raises(UserFacingError,match='automatic retry did not fix it'):
        _call_json('architecture','fixture',UsageLog('fixture'),validate=validate,stream_items=False)
    assert model.calls==2
