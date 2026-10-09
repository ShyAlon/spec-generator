from types import SimpleNamespace
from copy import deepcopy
import sqlite3
import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
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
    conn=sqlite3.connect(root/'spec'/'checkpoints.sqlite');saver=SqliteSaver(conn,serde=checkpoint_serde())
    from langgraph.checkpoint.base import empty_checkpoint
    checkpoint=empty_checkpoint();checkpoint['channel_values']=state
    saved=saver.put(cfg,checkpoint,{'source':'input','step':0,'parents':{}},{})
    conn.close()
    runtime=JobRuntime({'srs':StageModel('fake','fake')},str(root/'calls.jsonl'),lambda text:None)
    runtime.bind_worker(owner_id=owner,task_id='task',job_id='job',generation=1,session_root=root)
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
    cap,root,runtime,cid,state=setup(tmp_path);before=(root/'spec'/'checkpoints.sqlite').read_bytes();checks=[]
    with job_runtime(runtime),cli_execution(StopFence(root/'interview'/'stop-request.json'),None):
        proposal=run_revision_proposal(capability=cap,session_root=root,checkpoint_id=cid,phase='srs',
            change_request='Describe offline behavior',validate_authority=lambda:checks.append(1),log=None,graph_factory=factory(state))
    assert checks and not proposal['committed'] and proposal['requires_fresh_review']
    assert (root/'spec'/'checkpoints.sqlite').read_bytes()==before
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
