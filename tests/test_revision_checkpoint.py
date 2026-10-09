from copy import deepcopy
import pytest
from vmodel_contracts import IntentModel,StakeholderNeedsSet,SRSDocument,SystemBaseline
from spec_generator.revision_checkpoint import prepare_revision_seed
from test_progress import INTENT,PAYLOADS,CONTRACT_SRS
from conftest import SYSTEM_BASELINE
from vmodel_contracts.version import CONTRACT_VERSION


def source():
    return {'intent_model':IntentModel.model_validate(INTENT).model_dump(mode='json'),
        'needs':StakeholderNeedsSet.model_validate(PAYLOADS['needs']).model_dump(mode='json'),
        'system_baseline':SystemBaseline.model_validate({**SYSTEM_BASELINE,'contract_version':CONTRACT_VERSION,'repositories':[{**r,'contract_version':CONTRACT_VERSION} for r in SYSTEM_BASELINE['repositories']]}).model_dump(mode='json'),
        'srs':SRSDocument.model_validate(CONTRACT_SRS).model_dump(mode='json'),
        'architecture':{'old':True},'items':[{'old':True}],'tests':[{'old':True}],
        'bundle':{'old':True},'gate_feedback':{'srs':'old','architecture':'old'},
        'semantic_reviews':{'srs':{'old':True},'srs_applied_sha256':'old'},
        'expert_opinions':[{'gate':'srs'},{'gate':'architecture'}]}


def test_srs_revision_seed_is_typed_preserves_source_and_clears_prior_approvals():
    state=source();before=deepcopy(state)
    prepared=prepare_revision_seed(source=state,phase='srs',change_request='Describe offline behavior',session_id='s')
    seed=prepared['seed']
    assert state==before and seed['intent_model']==state['intent_model'] and seed['needs']==state['needs']
    assert seed['gate_feedback']=={'srs':'Describe offline behavior'}
    assert seed['srs'] is None and seed['architecture'] is None and seed['items'] is None and seed['bundle'] is None
    assert seed['semantic_reviews']=={} and seed['expert_opinions']==[]
    assert prepared['as_node']=='needs' and prepared['fresh_checkpoint_required']
    assert not prepared['partial_reuse_allowed'] and not prepared['committed']


@pytest.mark.parametrize('mutation',[lambda s:s['needs'].update(session_id='other'),
    lambda s:s['srs']['functional'][0].update(need_ids=['N-99']),
    lambda s:s['needs']['needs'][0].update(intent_need_ids=['SN-99']),
    lambda s:s['system_baseline'].update(mode='invented')])
def test_revision_seed_rejects_source_identity_contract_and_trace_drift(mutation):
    state=source();mutation(state)
    with pytest.raises(ValueError):prepare_revision_seed(source=state,phase='srs',change_request='Change behavior',session_id='s')


@pytest.mark.parametrize('phase',['needs','intent','conops','items','specs','tests','validation'])
def test_revision_seed_does_not_invent_unsupported_guidance_adapters(phase):
    with pytest.raises(ValueError,match='adapter unavailable'):
        prepare_revision_seed(source=source(),phase=phase,change_request='Change behavior',session_id='s')


def test_revision_seed_rejects_old_version_without_restamping():
    state=source();state['system_baseline']['contract_version']='0.16.0'
    with pytest.raises(ValueError):prepare_revision_seed(source=state,phase='srs',change_request='Change behavior',session_id='s')


def test_architecture_feedback_keeps_reviewed_prior_draft_but_clears_downstream():
    from vmodel_contracts import ArchitectureOverview
    state=source();state['architecture']=ArchitectureOverview.model_validate(PAYLOADS['architecture']).model_dump(mode='json')
    before=deepcopy(state)
    prepared=prepare_revision_seed(source=state,phase='architecture',change_request='Review storage choice',session_id='s')
    seed=prepared['seed']
    assert state==before and seed['srs']==state['srs'] and seed['architecture']==state['architecture']
    assert seed['items'] is None and seed['tests'] is None
    assert seed['gate_feedback']['architecture']=='Review storage choice'
    assert prepared['as_node']=='srs' and not prepared['partial_reuse_allowed']


def test_fresh_revision_checkpoint_consumes_scoped_feedback_without_source_write(tmp_path):
    from spec_generator.revision_checkpoint import seed_fresh_revision_checkpoint
    from spec_generator.stop_fence import StopFence
    from spec_generator.cooperative_run import cli_execution
    from spec_generator.stop_publication import FencedCheckpointer,publication_guard
    from langgraph.graph import StateGraph,END
    from langgraph.checkpoint.memory import MemorySaver
    from typing import TypedDict
    class State(TypedDict,total=False):
        intent_model:dict
        needs:dict
        system_baseline:dict
        srs:dict
        gate_feedback:dict
        result:str
    seen=[]
    def needs(s):return {}
    def srs(s):seen.append(s['gate_feedback']['srs']);return {'result':'fake proposal'}
    fence=StopFence(tmp_path/'stop.json');config={'configurable':{'thread_id':'s'}}
    original=source();before=deepcopy(original)
    with cli_execution(fence,None):
        generation=fence.generation
        builder=StateGraph(State);builder.add_node('needs',needs);builder.add_node('srs',srs)
        builder.set_entry_point('needs');builder.add_edge('needs','srs');builder.add_edge('srs',END)
        graph=builder.compile(checkpointer=FencedCheckpointer(MemorySaver(),publication_guard(fence,generation)))
        prepared=seed_fresh_revision_checkpoint(source=original,source_checkpoint_id='prior',expected_checkpoint_id='prior',
            phase='srs',change_request='Describe offline behavior',session_id='s',graph=graph,config=config)
        assert prepared['checkpoint_seeded'] and not prepared['producer_invoked'] and seen==[]
        assert graph.invoke(None,config)['result']=='fake proposal'
        assert seen==['Describe offline behavior'] and original==before
        with pytest.raises(ValueError,match='must be empty'):
            seed_fresh_revision_checkpoint(source=original,source_checkpoint_id='prior',expected_checkpoint_id='prior',
                phase='srs',change_request='Describe offline behavior',session_id='s',graph=graph,config=config)


def test_checkpoint_seed_refuses_unbound_execution():
    from spec_generator.revision_checkpoint import seed_fresh_revision_checkpoint
    with pytest.raises(ValueError,match='bound cooperative'):
        seed_fresh_revision_checkpoint(source=source(),source_checkpoint_id='prior',expected_checkpoint_id='prior',
            phase='srs',change_request='Change',session_id='s',graph=None,config={})
