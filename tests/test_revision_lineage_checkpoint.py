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
