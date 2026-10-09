from copy import deepcopy
import pytest
from vmodel_contracts import SRSDocument
from vmodel_contracts.semantic_review import make_review
from spec_generator.srs_rework import prepare_rework


def data():
    intent={'session_id':'fake','problem_statement':'People record tasks','constraints':[],'assumptions':[],'glossary':{},'target_users':['People'],'core_value_proposition':'Record tasks','stakeholder_needs':[{'need_id':'SN-1','statement':'People need task evidence','kind':'functional','source_turns':[0]}],'in_scope':['Task records'],'out_of_scope':[],'source_turns':{}}
    req={'requirement_id':'R-1','title':'Task evidence','statement':'The system shall use SQLite to record task evidence.',
         'priority':'must','kind':'functional','acceptance':[{'criterion_id':'AC-1','statement':'Task evidence is recorded.'}],
         'need_ids':['N-1'],'intent_fields':['problem_statement']}
    body=SRSDocument.model_validate({'session_id':'fake','intent_model_ref':'fake','purpose':'Test','scope_summary':'Task evidence',
        'functional':[req,{**deepcopy(req),'requirement_id':'R-2','acceptance':[{'criterion_id':'AC-2','statement':'Task evidence is available.'}]}],'non_functional':[]}).model_dump(mode='json')
    review=make_review('fake','srs',body,[{'finding_id':'f','item_id':'R-1','path':['functional',0],'rule':'solution_language',
        'reason':'Implementation choice in a requirement','original':body['functional'][0]}]).model_dump(mode='json')
    return dict(intent=intent,needs={'needs':[]},conops=None,review=review,revision=review['revision'],item_ids=['R-1'],guidance='Keep behavior only')


def test_scoped_producer_rework_preserves_unselected_and_requires_review():
    kw=data();original=deepcopy(kw['review']);calls=[]
    def producer(**context):
        calls.append(context);return {**context['original'],'statement':'The system shall record task evidence.'}
    result=prepare_rework(**kw,produce=producer)
    assert len(calls)==1 and calls[0]['guidance']=='Keep behavior only'
    assert result['review']['candidate']['functional'][1]==original['candidate']['functional'][1]
    assert kw['review']==original and result['review']['decisions']==[]
    assert result['requires_new_review'] and not result['founder_accepted'] and not result['committed']


@pytest.mark.parametrize('field,value',[('requirement_id','R-9'),('need_ids',['N-9']),('acceptance',[{'criterion_id':'AC-9','statement':'Changed'}])])
def test_rework_rejects_identity_trace_and_acceptance_drift(field,value):
    kw=data()
    with pytest.raises(ValueError):prepare_rework(**kw,produce=lambda **ctx:{**ctx['original'],field:value})


def test_stale_or_unflagged_never_calls_producer():
    kw=data();calls=[]
    for changed in ({'revision':'0'*64},{'item_ids':['R-2']}):
        with pytest.raises(ValueError):prepare_rework(**(kw|changed),produce=lambda **ctx:calls.append(ctx))
    assert calls==[]


def test_model_rework_uses_producer_call_but_does_not_approve(monkeypatch):
    from spec_generator import graph
    from spec_generator.srs_rework import prepare_model_rework
    calls=[]
    def fake(stage,payload,log,progress,**kwargs):
        calls.append((stage,payload,kwargs))
        original=data()['review']['candidate']['functional'][0]
        return {**original,'statement':'The system shall record task evidence.'}
    monkeypatch.setattr(graph,'_call_json',fake)
    result=prepare_model_rework(log=None,**data())
    assert len(calls)==1 and calls[0][0]=='srs' and calls[0][2]['item_label']=='R-1'
    assert 'Keep behavior only' in calls[0][1]
    assert result['requires_new_review'] and result['review']['decisions']==[]


def test_stop_after_accepted_rework_discards_without_mutating_source(tmp_path):
    from spec_generator.stop_fence import StopFence,StopRequested
    kw=data();original=deepcopy(kw['review']);f=StopFence(tmp_path/'stop.json');calls=[]
    def accepted(**context):
        calls.append(1);f.request_stop()
        return {**context['original'],'statement':'The system shall record task evidence.'}
    with f.running():
        with pytest.raises(StopRequested):prepare_rework(**kw,produce=accepted)
        with pytest.raises(StopRequested):prepare_rework(**kw,produce=accepted)
    assert kw['review']==original and calls==[1]
