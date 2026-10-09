from copy import deepcopy
from types import SimpleNamespace
import pytest
from spec_generator.revision_checkpoint import seed_lineage_revision_checkpoint
from spec_generator.stop_fence import StopFence
from spec_generator.cooperative_run import cli_execution
from test_revision_checkpoint import source


def lineage():
    return {'kind':'closed_revision_execution_lineage','semantic_session_id':'s',
        'parent_execution_session_id':'parent','parent_generation':2,'execution_session_id':'next',
        'generation':3,'phase':'srs','source_checkpoint_id':'checkpoint','state':'planned',
        'installed':False,'dispatch_started':False}


def fence(tmp_path):
    result=StopFence(tmp_path/'stop.json');result.generation=3
    return result


def graph():
    from langgraph.graph import StateGraph,END
    from langgraph.checkpoint.memory import InMemorySaver
    from test_revision_worker import factory
    return factory(source())(None,InMemorySaver())


def test_new_execution_thread_retains_semantic_source_ids(tmp_path):
    original=source();before=deepcopy(original);g=graph();config={'configurable':{'thread_id':'next'}}
    with cli_execution(fence(tmp_path),None):
        result=seed_lineage_revision_checkpoint(source=original,lineage=lineage(),change_request='Describe offline behavior',graph=g,config=config)
        values=g.get_state(config).values
    assert original==before and values['intent_model']['session_id']=='s' and values['needs']['session_id']=='s'
    assert result['execution_session_id']=='next' and not result['installed']
    assert not g.get_state({'configurable':{'thread_id':'s'}}).values


@pytest.mark.parametrize('change',[{'semantic_session_id':'foreign'},{'execution_session_id':'s'},
    {'execution_session_id':'parent'},{'generation':True},{'generation':4},{'installed':True}])
def test_wrong_lineage_fails_before_checkpoint_seed(tmp_path,change):
    g=graph();config={'configurable':{'thread_id':'next'}}
    with cli_execution(fence(tmp_path),None):
        with pytest.raises(ValueError):
            seed_lineage_revision_checkpoint(source=source(),lineage=lineage()|change,change_request='Describe offline behavior',graph=g,config=config)
    assert not g.get_state(config).values


def test_existing_destination_thread_cannot_be_reseeded(tmp_path):
    g=graph();config={'configurable':{'thread_id':'next'}}
    with cli_execution(fence(tmp_path),None):
        seed_lineage_revision_checkpoint(source=source(),lineage=lineage(),change_request='Describe offline behavior',graph=g,config=config)
        with pytest.raises(ValueError,match='must be empty'):
            seed_lineage_revision_checkpoint(source=source(),lineage=lineage(),change_request='changed guidance',graph=g,config=config)


def test_lineage_rejects_wrong_execution_fence_generation(tmp_path):
    g=graph();config={'configurable':{'thread_id':'next'}}
    with cli_execution(StopFence(tmp_path/'stop.json'),None):
        with pytest.raises(ValueError,match='fence generation'):
            seed_lineage_revision_checkpoint(source=source(),lineage=lineage(),change_request='Describe offline behavior',graph=g,config=config)
    assert not g.get_state(config).values


def test_staged_proposal_boundary_retains_semantic_ids_without_invoke(tmp_path):
    from spec_generator.revision_checkpoint import stage_reviewed_lineage_checkpoint,prepare_revision_seed,validate_revision_output
    from test_revision_worker import factory
    from langgraph.checkpoint.memory import InMemorySaver
    original=source();result=prepare_revision_seed(source=original,phase='srs',change_request='Describe offline behavior',session_id='s')['seed']
    result['srs']=deepcopy(original['srs']);result['gate_feedback'].pop('srs')
    validated=validate_revision_output(source=original,result=result,phase='srs',change_request='Describe offline behavior',session_id='s')
    binding=lineage()|{'regenerated_artifact_sha256':validated['artifact_sha256']}
    from spec_generator.graph import build_graph
    invoked=[];g=build_graph(SimpleNamespace(record=lambda *a:None),review=True,checkpointer=InMemorySaver());config={'configurable':{'thread_id':'next'}}
    with cli_execution(fence(tmp_path),None):
        receipt=stage_reviewed_lineage_checkpoint(source=original,proposal_result=result,lineage=binding,
            change_request='Describe offline behavior',graph=g,config=config)
    assert not invoked and not receipt['installed'] and receipt['requires_fresh_gate_review']
    assert g.get_state(config).next==('gate_srs',)
    assert g.get_state(config).values['srs']['session_id']=='s'
    assert g.get_state(config).values['intent_model']==original['intent_model']


def test_staging_digest_mismatch_never_writes_destination(tmp_path):
    from spec_generator.revision_checkpoint import stage_reviewed_lineage_checkpoint,prepare_revision_seed
    original=source();result=prepare_revision_seed(source=original,phase='srs',change_request='Describe offline behavior',session_id='s')['seed']
    result['srs']=deepcopy(original['srs']);result['gate_feedback'].pop('srs')
    g=graph();config={'configurable':{'thread_id':'next'}}
    with cli_execution(fence(tmp_path),None):
        with pytest.raises(ValueError,match='reviewed lineage digest'):
            stage_reviewed_lineage_checkpoint(source=original,proposal_result=result,lineage=lineage()|{'regenerated_artifact_sha256':'0'*64},
                change_request='Describe offline behavior',graph=g,config=config)
    assert not g.get_state(config).values


def test_staging_rejects_graph_that_bypasses_fresh_review(tmp_path):
    from spec_generator.revision_checkpoint import stage_reviewed_lineage_checkpoint,prepare_revision_seed,validate_revision_output
    original=source();result=prepare_revision_seed(source=original,phase='srs',change_request='Describe offline behavior',session_id='s')['seed']
    result['srs']=deepcopy(original['srs']);result['gate_feedback'].pop('srs')
    digest=validate_revision_output(source=original,result=result,phase='srs',change_request='Describe offline behavior',session_id='s')['artifact_sha256']
    g=graph();config={'configurable':{'thread_id':'next'}}
    with cli_execution(fence(tmp_path),None):
        with pytest.raises(ValueError,match='must require edited phase gate review'):
            stage_reviewed_lineage_checkpoint(source=original,proposal_result=result,lineage=lineage()|{'regenerated_artifact_sha256':digest},
                change_request='Describe offline behavior',graph=g,config=config)
    assert not g.get_state(config).values
