from types import SimpleNamespace
from copy import deepcopy
import pytest
from langgraph.checkpoint.memory import MemorySaver
from spec_generator.graph import checkpoint_serde
from spec_generator.stop_fence import StopFence
from spec_generator.cooperative_run import cli_execution
from spec_generator.worker_runtime import JobRuntime,job_runtime
from spec_generator.model_config import StageModel
from spec_generator.revision_worker import run_revision_proposal
from test_revision_checkpoint import source

def setup(tmp_path):
    owner='00000000-0000-0000-0000-000000000001'
    root=tmp_path/'tenants'/owner/'sessions'/'s';(root/'spec').mkdir(parents=True);(root/'interview').mkdir()
    cap=SimpleNamespace(scope=SimpleNamespace(owner_id=owner),task_id='task',session_id='s',generation=1)
    state=source();cfg={'configurable':{'thread_id':'s','checkpoint_ns':''}}
    saver=MemorySaver(serde=checkpoint_serde());saver.setup=lambda:None
    from langgraph.checkpoint.base import empty_checkpoint
    checkpoint=empty_checkpoint();checkpoint['channel_values']=state;checkpoint['channel_versions']={k:1 for k in state}
    saved=saver.put(cfg,checkpoint,{'source':'input','step':0,'parents':{}},checkpoint['channel_versions'])
    runtime=JobRuntime({'srs':StageModel('fake','fake')},str(root/'calls.jsonl'),lambda text:None)
    runtime.bind_worker(owner_id=owner,task_id='task',job_id='job',generation=1,session_root=root)
    savers={'spec':saver}
    def backend(namespace):
        if namespace not in savers:
            savers[namespace]=MemorySaver(serde=checkpoint_serde());savers[namespace].setup=lambda:None
        return savers[namespace]
    runtime.checkpoint_factory=backend
    return cap,root,runtime,saved['configurable']['checkpoint_id'],state


def factory(state,callback=None):
    from langgraph.graph import StateGraph,END
    from typing import TypedDict
    class State(TypedDict,total=False):
        intent_model:dict
        needs:dict
        system_baseline:dict
        conops:dict
        srs:dict
        gate_feedback:dict
    def build(log,checkpointer,**kwargs):
        graph=StateGraph(State);graph.add_node('needs',lambda s:{})
        def srs(s):
            if callback:callback()
            assert s['gate_feedback']['srs']=='Describe offline behavior'
            return {'srs':deepcopy(state['srs']),'gate_feedback':{}}
        graph.add_node('srs',srs);graph.set_entry_point('needs');graph.add_edge('needs','srs');graph.add_edge('srs',END)
        return graph.compile(checkpointer=checkpointer)
    return build


def test_owned_revision_checkpoint_runs_fake_and_never_mutates_source(tmp_path):
    cap,root,runtime,cid,state=setup(tmp_path);before=deepcopy(runtime.checkpoint_factory('spec').get_tuple({'configurable':{'thread_id':cap.session_id}}));checks=[]
    with job_runtime(runtime),cli_execution(StopFence(root/'interview'/'stop-request.json'),None):
        proposal=run_revision_proposal(capability=cap,session_root=root,checkpoint_id=cid,phase='srs',
            change_request='Describe offline behavior',validate_authority=lambda:checks.append(1),log=None,graph_factory=factory(state))
    assert checks and not proposal['committed'] and proposal['requires_fresh_review']
    assert runtime.checkpoint_factory('spec').get_tuple({'configurable':{'thread_id':cap.session_id}})==before
    assert (root/'spec'/'revision-proposals'/proposal['proposal_id']/'proposal.json').is_file()


def test_revision_worker_refuses_stale_checkpoint_before_output_creation(tmp_path):
    cap,root,runtime,cid,state=setup(tmp_path)
    with job_runtime(runtime),cli_execution(StopFence(root/'interview'/'stop-request.json'),None):
        with pytest.raises(ValueError,match='checkpoint changed'):
            run_revision_proposal(capability=cap,session_root=root,checkpoint_id='stale',phase='srs',
                change_request='Describe offline behavior',validate_authority=lambda:None,log=None,graph_factory=factory(state))
    assert not (root/'spec'/'revision-proposals').exists()


def test_revoked_authority_during_fake_producer_prevents_proposal_publication(tmp_path):
    cap,root,runtime,cid,state=setup(tmp_path);revoked=[]
    def authority():
        if revoked:raise PermissionError('revoked')
    with job_runtime(runtime),cli_execution(StopFence(root/'interview'/'stop-request.json'),None):
        with pytest.raises(PermissionError,match='revoked'):
            run_revision_proposal(capability=cap,session_root=root,checkpoint_id=cid,phase='srs',
                change_request='Describe offline behavior',validate_authority=authority,log=None,
                graph_factory=factory(state,lambda:revoked.append(True)))
    assert not list((root/'spec'/'revision-proposals').glob('*/proposal.json'))


def test_stop_during_fake_producer_prevents_proposal_publication(tmp_path):
    from spec_generator.stop_fence import StopRequested
    cap,root,runtime,cid,state=setup(tmp_path);fence=StopFence(root/'interview'/'stop-request.json')
    with job_runtime(runtime),cli_execution(fence,None):
        with pytest.raises(StopRequested):
            run_revision_proposal(capability=cap,session_root=root,checkpoint_id=cid,phase='srs',
                change_request='Describe offline behavior',validate_authority=lambda:None,log=None,
                graph_factory=factory(state,fence.request_stop))
    assert not list((root/'spec'/'revision-proposals').glob('*/proposal.json'))


def test_worker_rejects_foreign_owned_root_and_mismatched_runtime(tmp_path):
    cap,root,runtime,cid,state=setup(tmp_path)
    with job_runtime(runtime),cli_execution(StopFence(root/'interview'/'stop-request.json'),None):
        with pytest.raises(ValueError,match='namespace'):
            run_revision_proposal(capability=cap,session_root=tmp_path,checkpoint_id=cid,phase='srs',
                change_request='Describe offline behavior',validate_authority=lambda:None,log=None)
        cap.task_id='foreign'
        with pytest.raises(ValueError,match='another worker'):
            run_revision_proposal(capability=cap,session_root=root,checkpoint_id=cid,phase='srs',
                change_request='Describe offline behavior',validate_authority=lambda:None,log=None)


def test_real_generator_graph_revision_consumes_feedback_and_stops_at_srs(tmp_path,monkeypatch):
    import json
    from langchain_core.messages import AIMessage
    from spec_generator import model_config
    from test_progress import PAYLOADS,FakeModel
    from conftest import schema_routing_factory
    cap,root,runtime,cid,state=setup(tmp_path);messages=[]
    class Model(FakeModel):
        def invoke(self,incoming):
            messages.append('\n'.join(str(m.content) for m in incoming))
            return AIMessage(content=json.dumps(PAYLOADS['srs']))
    monkeypatch.setattr(model_config,'build_chat_model',lambda stage,**kw:schema_routing_factory(stage,kw.get('schema'),lambda:Model(stage)))
    runtime.models['srs']=StageModel('fake','fake');runtime.settings['SPEC_GENERATOR_BULK']='1'
    with job_runtime(runtime),cli_execution(StopFence(root/'interview'/'stop-request.json'),None):
        proposal=run_revision_proposal(capability=cap,session_root=root,checkpoint_id=cid,phase='srs',
            change_request='Describe offline behavior',validate_authority=lambda:None,log=SimpleNamespace(record=lambda *args:None))
    assert messages and any('Describe offline behavior' in text for text in messages)
    assert proposal['phase']=='srs' and not proposal['committed']


def setup_later_generation(tmp_path,*,anchor_task='task'):
    import json
    from vmodel_contracts.execution_anchor import ExecutionAnchor
    cap,old,runtime,cid,state=setup(tmp_path)
    root=old.parent/'execution2';old.rename(root);cap.session_id='execution2';cap.generation=2
    saver=MemorySaver(serde=checkpoint_serde());saver.setup=lambda:None
    from langgraph.checkpoint.base import empty_checkpoint
    cp=empty_checkpoint();cp['channel_values']=state;cp['channel_versions']={k:1 for k in state}
    saved=saver.put({'configurable':{'thread_id':'execution2','checkpoint_ns':''}},cp,{'source':'input','step':0,'parents':{}},cp['channel_versions'])
    runtime=JobRuntime({'srs':StageModel('fake','fake')},str(root/'calls.jsonl'),lambda text:None)
    runtime.bind_worker(owner_id=cap.scope.owner_id,task_id='task',job_id='job',generation=2,session_root=root)
    anchor=ExecutionAnchor(cap.scope.owner_id,anchor_task,'execution2','s',2,'a'*64,'test-installation')
    (root/'execution-generation.json').write_text(json.dumps(anchor.document()))
    fence=StopFence(root/'interview'/'stop-request.json',execution_anchor=anchor)
    savers={'spec':saver}
    def backend(namespace):
        if namespace not in savers:
            savers[namespace]=MemorySaver(serde=checkpoint_serde());savers[namespace].setup=lambda:None
        return savers[namespace]
    runtime.checkpoint_factory=backend
    return cap,root,runtime,saved['configurable']['checkpoint_id'],state,fence


def test_later_generation_proposal_preserves_original_semantic_ids(tmp_path):
    cap,root,runtime,cid,state,fence=setup_later_generation(tmp_path)
    before=deepcopy(runtime.checkpoint_factory('spec').get_tuple({'configurable':{'thread_id':cap.session_id}}))
    with job_runtime(runtime),cli_execution(fence,None):
        proposal=run_revision_proposal(capability=cap,session_root=root,checkpoint_id=cid,phase='srs',
            change_request='Describe offline behavior',validate_authority=lambda:None,log=None,graph_factory=factory(state))
    assert proposal['session_id']=='execution2' and proposal['generation']==2
    assert proposal['artifact']['session_id']=='s' and proposal['artifact']['intent_model_ref']=='s'
    assert before==runtime.checkpoint_factory('spec').get_tuple({'configurable':{'thread_id':cap.session_id}})
    assert not proposal['committed']


def test_later_generation_proposal_rejects_wrong_installed_task(tmp_path):
    cap,root,runtime,cid,state,fence=setup_later_generation(tmp_path,anchor_task='foreign')
    with job_runtime(runtime),cli_execution(fence,None):
        with pytest.raises(ValueError,match='semantic/execution binding'):
            run_revision_proposal(capability=cap,session_root=root,checkpoint_id=cid,phase='srs',
                change_request='Describe offline behavior',validate_authority=lambda:None,log=None,graph_factory=factory(state))
    assert not (root/'spec'/'revision-proposals').exists()


def test_unanchored_ad_hoc_later_generation_refuses_proposal(tmp_path):
    cap,root,runtime,cid,state=setup(tmp_path);cap.generation=2
    runtime=JobRuntime({'srs':StageModel('fake','fake')},str(root/'calls.jsonl'),lambda text:None)
    runtime.bind_worker(owner_id=cap.scope.owner_id,task_id='task',job_id='job',generation=2,session_root=root)
    fence=StopFence(root/'interview'/'stop-request.json');fence.generation=2
    with job_runtime(runtime),cli_execution(fence,None):
        with pytest.raises(ValueError,match='semantic/execution binding'):
            run_revision_proposal(capability=cap,session_root=root,checkpoint_id=cid,phase='srs',
                change_request='Describe offline behavior',validate_authority=lambda:None,log=None,graph_factory=factory(state))
    assert not (root/'spec'/'revision-proposals').exists()
