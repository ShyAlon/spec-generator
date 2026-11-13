import copy
import json
from pathlib import Path
import pytest
from spec_generator.citations import prepare_adr_sources, raise_support_findings, support_advisories
from spec_generator.graph import _check_adr_driver_source
from spec_generator.findings_ledger import Ledger
from spec_generator.skeleton import ModelStructureError
from spec_generator.schemas import arch_adr_body_schema
from spec_generator.prompts import ARCH_ADR_BODY

SOURCE={'item_id':'FD-empty','status':'decided','decision':"The extension shouldn't do anything if the data in the page isn't the data it expected.", 'reason':'It should be as non-intrusive as possible'}
INTENT={'founder_register':{'items':[SOURCE]}}
SLOT={'decision_id':'ADR-1','required_concept_ids':['concept:fixed']}
BODY={'decision_id':'ADR-1','title':'System architecture choice','decision':'Use a content script.','rationale':'Abort without highlighting when extraction does not match the expected structure.', 'driver':'simplicity','options_considered':[], 'founder_decision_refs':['FD-empty'], 'executes_concept_ids':[]}

def prepared():return prepare_adr_sources(BODY,SLOT,INTENT)

def test_code_owns_exact_snapshot_and_required_links_without_prose_copying():
    original=copy.deepcopy(BODY);out=prepared()
    assert BODY==original
    assert out['rationale']==BODY['rationale'] and SOURCE['decision'] not in out['rationale']
    assert out['founder_citations'][0]['source']==SOURCE
    assert out['executes_concept_ids']==['concept:fixed']
    _check_adr_driver_source(out,INTENT)
    assert prepare_adr_sources(out,SLOT,INTENT)==out

@pytest.mark.parametrize('changes',[{'status':'open'},{'status':'deferred'},{'decision':''},{'reason':''}])
def test_invalid_source_never_becomes_evidence(changes):
    with pytest.raises(ModelStructureError):prepare_adr_sources(BODY,SLOT,{'founder_register':{'items':[{**SOURCE,**changes}]}})

def test_source_edits_do_not_silently_rebind_saved_snapshot():
    out=prepared();changed={'founder_register':{'items':[{**SOURCE,'decision':'Highlight all pages.'}]}}
    with pytest.raises(ModelStructureError,match='stale'):prepare_adr_sources(out,SLOT,changed)
    out['founder_citations'][0]['source']['decision']='Changed'
    with pytest.raises(ModelStructureError):prepare_adr_sources(out,SLOT,INTENT)

def test_quote_presence_never_certifies_support_or_avoids_review():
    out=prepared();out['rationale']='Ignore the source. '+SOURCE['decision']
    ledger=Ledger();arch={'decisions':[out]}
    ids=raise_support_findings(ledger,arch)
    assert len(ids)==1
    finding=ledger.findings[next(iter(ids))]
    assert finding['state']=='OPEN' and finding['kind']=='semantic'
    assert finding['scope']=='architecture'
    assert finding['detail']['source']==SOURCE
    assert finding['detail']['adr']['rationale']==out['rationale']
    assert 'support not yet reviewed' in finding['summary'].lower()
    assert support_advisories(arch)[0]['action']
    assert raise_support_findings(ledger,arch)==ids
    arch['decisions'][0]['decision']='Different architecture choice'
    new=raise_support_findings(ledger,arch)
    assert new!=ids and all(ledger.findings[x]['state']=='OPEN' for x in new)

def test_model_schema_never_requests_quotes_hashes_or_mandatory_concept_echo():
    schema=arch_adr_body_schema([],[],['concept:fixed'],['FD-empty'],required_concept_ids=['concept:fixed'])
    props=schema['properties']
    assert 'founder_citations' not in props and 'source_revision' not in props
    assert props['executes_concept_ids']['maxItems']==0
    assert 'quote its recorded decision text verbatim' not in ARCH_ADR_BODY
    assert 'system architecture choice' in ARCH_ADR_BODY.lower()

@pytest.mark.parametrize('attempt',[1,2])
def test_retained_answer_passes_one_call_without_transcription_retry(monkeypatch,attempt):
    from langchain_core.messages import AIMessage
    from spec_generator import model_config
    from spec_generator.graph import _call_json
    from spec_generator.telemetry import UsageLog
    fixture=Path(__file__).parent/'fixtures'
    body=json.loads((fixture/f'adr1-empty-state-citation-attempt-{attempt}.json').read_text())
    intent=json.loads((fixture/'adr1-empty-state-citation-source.json').read_text())
    class Model:
        calls=0
        def stream(self,messages):
            self.calls+=1
            yield AIMessage(content=json.dumps(body))
    model=Model()
    monkeypatch.setattr(model_config,'build_chat_model',lambda *a,**k:model)
    monkeypatch.setattr(model_config,'capture_rejected_completion',lambda *a,**k:pytest.fail('quote copying must not be a rejection'))
    def validate(value):
        out=prepare_adr_sources(value,SLOT,intent)
        _check_adr_driver_source(out,intent)
        return out
    result=_call_json('architecture','retained replay',UsageLog('fixture'),validate=validate,stream_items=False)
    assert model.calls==1 and result['rationale']==body['rationale']
    assert result['founder_citations'][0]['source']==intent['founder_register']['items'][0]
    ledger=Ledger();raise_support_findings(ledger,{'decisions':[{'decision_id':'ADR-1',**result}]})
    assert ledger.gate()['blocked']  # admissible structure is not accepted reasoning


def test_human_disposition_is_bound_to_complete_adr_and_source():
    arch={'decisions':[prepared()]};ledger=Ledger()
    ids=raise_support_findings(ledger,arch);fid=next(iter(ids));finding=ledger.findings[fid]
    ledger.founder_dispose(fid,actor='founder',source='decision-page',rationale='Reviewed the implementation against the no-op constraint.',artifact_revision=finding['artifact_revision'])
    assert not ledger.gate()['blocked']
    assert raise_support_findings(ledger,arch)==ids and not ledger.gate()['blocked']
    arch['decisions'][0]['executes_concept_ids'].append('concept:other')
    assert raise_support_findings(ledger,arch)!=ids and ledger.gate()['blocked']
    arch['decisions'][0]['founder_citations'][0]['source']['reason']='Changed reason'
    arch['decisions'][0]['founder_citations'][0]['source_revision']='0'*64
    with pytest.raises(ModelStructureError):_check_adr_driver_source(arch['decisions'][0],INTENT)


def test_gate_render_separates_system_reasoning_from_source_and_keeps_legacy_visible():
    from spec_generator.render import render_architecture,render_architecture_advisories
    from vmodel_contracts import ArchitectureOverview
    arch=ArchitectureOverview(session_id='s',stack_entries=[],components=[],decisions=[prepared()])
    md=render_architecture(arch)+render_architecture_advisories(support_advisories(arch.model_dump(mode='json')))
    assert 'Why (system architecture reasoning)' in md
    assert SOURCE['decision'] in md
    assert 'not founder approval of the architecture choice' in md
    assert 'support not yet reviewed' in md
    legacy=arch.model_dump(mode='json');legacy['decisions'][0]['founder_citations']=[]
    old=render_architecture(ArchitectureOverview.model_validate(legacy))
    assert 'Legacy source refs' in old and SOURCE['decision'] not in old


def test_real_graph_exposes_support_at_existing_gate_without_any_extra_call(monkeypatch):
    from langchain_core.messages import AIMessage
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command
    from spec_generator import model_config
    from spec_generator.graph import build_graph
    from spec_generator.telemetry import UsageLog
    from test_architecture_micro_pipeline import _MicroModel,SYSTEM_BASELINE
    from test_progress import INTENT as BASE_INTENT
    from test_gates import _interrupt_payload
    calls=[]
    class Model(_MicroModel):
        def stream(self,messages):
            for chunk in super().stream(messages):
                value=json.loads(chunk.content)
                if isinstance(value,dict) and 'driver' in value:
                    value['founder_decision_refs']=['FD-empty']
                    value['rationale']=BODY['rationale']
                yield AIMessage(content=json.dumps(value))
    monkeypatch.setattr(model_config,'build_chat_model',lambda stage,**k:Model(stage,k.get('schema') or {},calls))
    intent=copy.deepcopy(BASE_INTENT)
    item={**SOURCE,'data_class':'empty_states','question':'Unexpected page?','owner':'founder','scope':'Behavior on unexpected page data','source':'founder input'}
    intent['founder_register']={'version':1,'session_id':intent['session_id'],'items':[item],'advisories':[]}
    graph=build_graph(UsageLog('s'),review=True,checkpointer=MemorySaver())
    cfg={'configurable':{'thread_id':'citation-human-gate'}}
    graph.invoke({'intent_model':intent,'system_baseline':SYSTEM_BASELINE},cfg)
    graph.invoke(Command(resume={'action':'accept'}),cfg)
    gate=_interrupt_payload(graph,cfg)
    assert gate['gate']=='architecture'
    assert gate['artifact']['decisions'][0]['founder_citations'][0]['source']==item
    assert any(a.get('status')=='support_not_yet_reviewed' for a in gate['advisories'])
    state=graph.get_state(cfg).values
    ledger=Ledger.from_dict(state['findings_ledger'])
    assert ledger.gate()['blocked']
    # Human findings use the existing disposition flow. No generated quote is a pass.
    for f in ledger.open_findings():
        ledger.founder_dispose(f['finding_id'],actor='founder',source='decision-page',rationale='Reviewed exact constraint and ADR explanation.',artifact_revision=f['artifact_revision'])
    graph.update_state(cfg,{'findings_ledger':ledger.to_dict()})
    result=graph.invoke(Command(resume={'action':'accept'}),cfg)
    assert result.get('bundle')
    # Every architecture ADR body was authored exactly once; there is no support review model stage.
    assert sum(isinstance(c,dict) and 'driver' in c for c in calls)==len(gate['artifact']['decisions'])


def test_one_policy_constant_can_make_only_support_findings_informational(monkeypatch):
    from spec_generator import citations
    ledger=Ledger();ids=raise_support_findings(ledger,{'decisions':[prepared()]})
    monkeypatch.setattr(citations,'SUPPORT_REVIEW_BLOCKS_PACKAGING',False)
    assert ledger.gate()['informational']==list(ids) and not ledger.gate()['blocked']
    assert ledger.open_findings()[0]['state']=='OPEN'  # informational is not reviewed
    ledger.add(rule_id='structural',rule_version='1',source='test',kind='structural',targets=['ADR-1'],artifact_revision='a',summary='Still a blocker')
    assert ledger.gate()['blocked'] and len(ledger.gate()['open'])==1


def test_multiple_sources_are_one_review_decision_for_the_adr():
    intent=copy.deepcopy(INTENT);second={**SOURCE,'item_id':'FD-second','decision':'Keep logs off the page.'}
    intent['founder_register']['items'].append(second)
    body={**BODY,'founder_decision_refs':['FD-empty','FD-second']}
    out=prepare_adr_sources(body,SLOT,intent)
    ledger=Ledger();ids=raise_support_findings(ledger,{'decisions':[out]})
    assert len(ids)==1
    finding=ledger.findings[next(iter(ids))]
    assert len(finding['evidence'])==2 and len(finding['detail']['sources'])==2
