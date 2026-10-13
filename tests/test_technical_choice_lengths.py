import json,copy
import pytest
from pydantic import ValidationError
from spec_generator import graph,model_config
from spec_generator.telemetry import UsageLog
from spec_generator.errors import UserFacingError
from vmodel_contracts.architecture import technical_choice_data

IDS=['R-1']
GOOD={'title':'Checks','purpose':'Catch broken changes.','default_value':'host-ci','reason':'Browser tests cover access checks and upload recovery without a separate household server.','requirement_ids':IDS,'options':[{'value':'host-ci','label':'Host automation','implication':'Fewer systems to maintain.'},{'value':'own-runner','label':'Own machine','implication':'More control, more upkeep.'}]}
STATE={'srs':{'functional':[{'requirement_id':'R-1'}]},'system_baseline':{},'intent_model':{},'conops':{}}

def test_budgets_come_from_shared_contract():
    assert graph._technical_choice_length_budget()=={'title':100,'purpose':300,'default_value':200,'reason':400,'options[].value':200,'options[].label':160,'options[].implication':400}

@pytest.mark.parametrize('length',[505,852])
def test_exact_length_feedback_without_echo(length):
    d=copy.deepcopy(GOOD);d['reason']='x'*length
    with pytest.raises(ValidationError) as e:technical_choice_data(d,IDS)
    feedback=graph._technical_choice_validation_feedback(e.value)
    assert f'reason is {length} characters; maximum 400' in feedback
    assert d['reason'] not in feedback and 'requirement_ids unchanged' in feedback

@pytest.mark.parametrize('length,valid',[(400,True),(401,False)])
def test_shared_boundary_not_relaxed(length,valid):
    d=copy.deepcopy(GOOD);d['reason']='x'*length
    if valid:assert technical_choice_data(d,IDS)['reason']==d['reason']
    else:
        with pytest.raises(ValidationError):technical_choice_data(d,IDS)

def run(monkeypatch,tmp_path,outputs):
    calls=[]
    def invoke(stage,payload,log,progress=None,feedback=None,**kw):
        calls.append((payload,feedback));return json.dumps(outputs[len(calls)-1])
    monkeypatch.setattr(graph,'_invoke',invoke)
    monkeypatch.setattr(model_config,'capture_rejected_completion',lambda *a:'fixture-only')
    monkeypatch.setattr(model_config,'_console_line',lambda *a:None)
    monkeypatch.setattr(model_config,'call_diagnostic',lambda *a:{})
    return calls

def test_initial_numeric_budget_and_one_targeted_retry(monkeypatch,tmp_path):
    bad=copy.deepcopy(GOOD);bad['reason']='x'*852
    calls=run(monkeypatch,tmp_path,[bad,GOOD])
    result=graph._technical_choice_call(STATE,'ci','brief',UsageLog('s'))
    assert 'reason <= 400' in calls[0][0] and 'options[].label <= 160' in calls[0][0]
    assert calls[0][1] is None and len(calls)==2
    assert 'reason is 852 characters; maximum 400' in calls[1][1]
    assert 'not product content' in calls[1][1]
    assert result['reason']==GOOD['reason'] and result['requirement_ids']==IDS
    assert result['default_value']==GOOD['default_value'] and result['options'][0]['value']==GOOD['options'][0]['value']

def test_second_overlong_stops_without_truncation_or_more_calls(monkeypatch,tmp_path):
    first=copy.deepcopy(GOOD);first['reason']='x'*852
    second=copy.deepcopy(GOOD);second['reason']='y'*505
    calls=run(monkeypatch,tmp_path,[first,second])
    with pytest.raises(UserFacingError):graph._technical_choice_call(STATE,'ci','brief',UsageLog('s'))
    assert len(calls)==2 and len(second['reason'])==505

def test_non_length_failure_not_bypassed(monkeypatch,tmp_path):
    bad=copy.deepcopy(GOOD);bad['default_value']='not-an-option'
    calls=run(monkeypatch,tmp_path,[bad,bad])
    with pytest.raises(UserFacingError):graph._technical_choice_call(STATE,'ci','brief',UsageLog('s'))
    assert len(calls)==2 and 'technical default' in calls[1][1]
