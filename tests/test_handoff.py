import json
from pathlib import Path
import pytest
from spec_generator.handoff import build_handoff
from spec_generator.graph import build_graph
from spec_generator.cli import _make_persist
from spec_generator import model_config
from spec_generator.telemetry import UsageLog
from conftest import SYSTEM_BASELINE,schema_routing_factory
from test_progress import INTENT,FakeModel
from vmodel_contracts.founder_register import FounderRegister
from vmodel_contracts import IntentModel,ConOpsDocument
from vmodel_contracts.conops import *

def outputs(tmp_path,monkeypatch):
 from test_srs import _run,SRS
 result=_run(monkeypatch,SRS)
 interview=tmp_path/'interview';spec=tmp_path/'spec';interview.mkdir();spec.mkdir()
 intent={**INTENT,'founder_register':FounderRegister(session_id=INTENT['session_id']).model_dump()}
 intent=IntentModel.model_validate(intent).model_dump(mode='json');(interview/'intent_model.json').write_text(json.dumps(intent))
 # Reuse typed ConOps fixture from contracts test rather than approximate it.
 conops={'session_id':INTENT['session_id'],'stakeholder_context':[{'stakeholder':'Founder','role':'Owner','goals':['Specs'],'operating_environment':'Browser','source_turns':[0]}],'operational_scenarios':[{'scenario_id':'OS-1','name':'Generate','trigger':'Idea','actors':['Founder'],'operational_flow':['Answer','Review'],'success_outcome':'Specification','exceptions':['Invalid output'],'source_turns':[0]}],
 'concept_options':[{'name':'PWA','summary':'Browser','advantages':['One deploy'],'disadvantages':['Device limits']},{'name':'Native','summary':'Installed','advantages':['APIs'],'disadvantages':['Stores']}],
 'decision':{'selected_concept':'PWA','rationale':'One deployment','founder_confirmed':True,'source_turns':[0]}}
 conops=ConOpsDocument.model_validate(conops);(interview/'conops.json').write_text(conops.model_dump_json())
 b=result['bundle'];persist=_make_persist(intent,spec)
 for stage,key in [('needs','_needs'),('srs','_srs'),('architecture','_architecture'),('items','items'),('tests','_tests')]:persist(stage,b[key],{})
 (spec/'findings-ledger.json').write_text(json.dumps(__import__('spec_generator.findings_ledger',fromlist=['Ledger']).Ledger().to_dict()))  # reviewed, nothing open
 return interview,spec,intent

def test_export_only_outputs_paths_and_no_silent_defaults(tmp_path,monkeypatch):
 interview,spec,intent=outputs(tmp_path,monkeypatch)
 (spec/'llm-calls.jsonl').write_text('PRIVATE DEBUG');(spec/'checkpoints.sqlite').write_text('PRIVATE STATE')
 handoff=build_handoff(interview,spec)
 assert not handoff['blocked_reasons'];assert handoff['session_id']==intent['session_id']
 assert 'spec/AGENTS.md' in handoff['files'];assert 'Never silently substitute defaults' in handoff['files']['spec/AGENTS.md']
 assert 'Decided' in handoff['files']['README.md'];assert not any('PRIVATE' in x for x in handoff['files'].values())
 for path in json.loads(handoff['files']['spec/manifest.json'])['files'].values():assert 'spec/'+path in handoff['files']

def test_incremental_writer_stamps_raw_payloads_so_the_export_gate_accepts_them(tmp_path,monkeypatch):
 # Regression: the writer dumped raw dicts with no contract_version, and Prepare final package
 # then failed with ContractVersionMismatch (payload produced against 'unknown').
 from test_srs import _run,SRS
 from vmodel_contracts import CONTRACT_VERSION
 interview,spec,intent=outputs(tmp_path,monkeypatch)
 b=_run(monkeypatch,SRS)['bundle'];persist=_make_persist(intent,spec)
 def raw(v):
  if isinstance(v,list):return [raw(x) for x in v]
  return {k:x for k,x in v.items() if k!='contract_version'} if isinstance(v,dict) else v
 for stage,key in [('needs','_needs'),('srs','_srs'),('architecture','_architecture'),('items','items'),('tests','_tests')]:persist(stage,raw(b[key]),{})
 for name in ('needs','srs','architecture'):assert json.loads((spec/f'{name}.json').read_text())['contract_version']==CONTRACT_VERSION
 for name in ('items','tests'):assert all(x['contract_version']==CONTRACT_VERSION for x in json.loads((spec/f'{name}.json').read_text()))
 assert not build_handoff(interview,spec)['blocked_reasons']

def test_open_founder_data_blocks_final_readiness_and_other_run_is_denied(tmp_path,monkeypatch):
 interview,spec,intent=outputs(tmp_path,monkeypatch)
 intent['founder_register']['items']=[{'item_id':'FD-style','data_class':'styling','question':'Which style?','owner':'founder','status':'open','scope':'Design','source':'Interview','blocks':['UI']}]
 (interview/'intent_model.json').write_text(json.dumps(intent))
 srs=json.loads((spec/'srs.json').read_text());srs['founder_register']=intent['founder_register'];(spec/'srs.json').write_text(json.dumps(srs))
 handoff=build_handoff(interview,spec);assert handoff['blocked_reasons']==['FD-style: Which style?']
 conops=json.loads((interview/'conops.json').read_text());conops['session_id']='other';(interview/'conops.json').write_text(json.dumps(conops))
 with pytest.raises(ValueError,match='different sessions'):build_handoff(interview,spec)

def test_cli_export_runs_zero_model_calls(tmp_path,monkeypatch,capsys):
 import sys
 from spec_generator.cli import main
 interview,spec,intent=outputs(tmp_path,monkeypatch)
 from vmodel_contracts import SystemBaseline,CONTRACT_VERSION
 (spec/'system-baseline.json').write_text(SystemBaseline.model_validate({**SYSTEM_BASELINE,'contract_version':CONTRACT_VERSION}).model_dump_json())
 def forbidden(*args,**kwargs):raise AssertionError('Model was called for packaging')
 monkeypatch.setattr(model_config,'build_chat_model',forbidden)
 monkeypatch.setattr(sys,'argv',['spec-generator',str(interview/'intent_model.json'),'--conops',str(interview/'conops.json'),'--system-baseline',str(spec/'system-baseline.json'),'--out',str(spec),'--export-handoff'])
 capsys.readouterr();main();out=capsys.readouterr().out;assert json.loads(out)['session_id']==intent['session_id']

def test_semantic_intent_survives_added_register_and_modified_output_blocks(tmp_path,monkeypatch):
 from vmodel_contracts.semantic_review import make_review
 interview,spec,intent=outputs(tmp_path,monkeypatch)
 original={**intent,'founder_register':None}
 review=make_review(intent['session_id'],'intent',original,[{'finding_id':'f','item_id':'scope','path':['in_scope',0],'rule':'grounding','reason':'Check scope','original':original['in_scope'][0]}])
 data=review.model_dump();data['decisions']=[{'item_id':'scope','action':'accept','actor':'founder'}]
 (interview/'semantic-review.json').write_text(json.dumps(data))
 assert not build_handoff(interview,spec)['blocked_reasons']
 intent['in_scope'][0]='Changed outside review';(interview/'intent_model.json').write_text(json.dumps(intent))
 with pytest.raises(ValueError,match='differs from reviewed'):build_handoff(interview,spec)


def test_old_contract_and_unsafe_baseline_are_denied(tmp_path,monkeypatch):
 interview,spec,intent=outputs(tmp_path,monkeypatch)
 intent['contract_version']='0.19.0';(interview/'intent_model.json').write_text(json.dumps(intent))
 with pytest.raises(ValueError,match='0.19'):build_handoff(interview,spec)
 intent['contract_version']='0.22.0';(interview/'intent_model.json').write_text(json.dumps(intent))
 (spec/'system-baseline.json').write_text('x'*2_000_001)
 with pytest.raises(ValueError,match='Baseline exceeds'):build_handoff(interview,spec)

def test_override_evidence_is_in_export_digest(tmp_path,monkeypatch):
 from vmodel_contracts.semantic_review import make_review
 interview,spec,intent=outputs(tmp_path,monkeypatch)
 srs=json.loads((spec/'srs.json').read_text())
 review=make_review('s','srs',srs,[{'finding_id':'f','item_id':'R-1','path':['functional',0],'rule':'solution_language','reason':'Owner choice','original':srs['functional'][0]}])
 data=review.model_dump();data['decisions']=[{'item_id':'R-1','action':'accept','actor':'founder','rationale':'Required product behavior'}]
 (spec/'srs.semantic-review.json').write_text(json.dumps(data))
 h=build_handoff(interview,spec)
 assert 'spec/srs.semantic-review.json' in h['files']
 assert any(d['choice']=='accept' and d['why']=='Required product behavior' for d in json.loads(h['files']['decisions.json']))


def test_handoff_blocks_on_missing_or_open_ledger(tmp_path,monkeypatch):
 interview,spec,_=outputs(tmp_path,monkeypatch)
 from spec_generator.findings_ledger import Ledger
 (spec/'findings-ledger.json').unlink()
 assert any('never content-reviewed' in r for r in build_handoff(interview,spec)['blocked_reasons'])
 ledger=Ledger();ledger.add(rule_id='echo.directive',rule_version='1',source='rule',kind='semantic',targets=['SPEC-1'],artifact_revision='x',summary='gray')
 (spec/'findings-ledger.json').write_text(json.dumps(ledger.to_dict()))
 assert any('open finding' in r for r in build_handoff(interview,spec)['blocked_reasons'])

def test_decided_input_without_a_reason_reads_no_reason_recorded(tmp_path,monkeypatch):
 interview,spec,intent=outputs(tmp_path,monkeypatch)
 intent['founder_register']['items']=[{'item_id':'FD-tone','data_class':'content','question':'Tone?','owner':'founder','status':'decided','decision':'Friendly','scope':'Copy','source':'Interview','blocks':[]}]
 (interview/'intent_model.json').write_text(json.dumps(intent))
 srs=json.loads((spec/'srs.json').read_text());srs['founder_register']=intent['founder_register'];(spec/'srs.json').write_text(json.dumps(srs))
 handoff=build_handoff(interview,spec)
 assert 'FD-tone: Friendly Why: no reason recorded.' in handoff['files']['README.md'] or 'Why: no reason recorded' in json.dumps(handoff['files'])
