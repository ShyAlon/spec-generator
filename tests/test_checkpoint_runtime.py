"""Host-supplied checkpoints preserve real revision flow, no SQLite authority."""
from copy import deepcopy
import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.base import empty_checkpoint
from spec_generator.worker_runtime import job_runtime
from spec_generator.cooperative_run import cli_execution
from spec_generator.stop_fence import StopFence
from spec_generator.revision_worker import run_revision_proposal
from test_revision_worker import factory

def setup(tmp_path):
    from types import SimpleNamespace
    from spec_generator.worker_runtime import JobRuntime
    from spec_generator.model_config import StageModel
    from test_revision_checkpoint import source
    owner="00000000-0000-0000-0000-000000000001"
    root=tmp_path/"tenants"/owner/"sessions"/"s"
    (root/"spec").mkdir(parents=True);(root/"interview").mkdir()
    cap=SimpleNamespace(scope=SimpleNamespace(owner_id=owner),task_id="task",session_id="s",generation=1)
    runtime=JobRuntime({"srs":StageModel("fake","fake")},str(root/"calls.jsonl"),lambda text:None)
    runtime.bind_worker(owner_id=owner,task_id="task",job_id="job",generation=1,session_root=root)
    return cap,root,runtime,"unused",source()


def test_bound_revision_uses_checkpoint_factory_without_sqlite(tmp_path,monkeypatch):
    cap,root,runtime,old_id,state=setup(tmp_path)
    savers={}
    def backend(namespace):
        if namespace not in savers:
            saver=MemorySaver();saver.setup=lambda:None;savers[namespace]=saver
        return savers[namespace]
    runtime.checkpoint_factory=backend
    cp=empty_checkpoint();cp['channel_values']=deepcopy(state);cp['channel_versions']={k:1 for k in state}
    cfg=backend('spec').put({'configurable':{'thread_id':'s','checkpoint_ns':''}},cp,{'source':'input','step':0,'parents':{}},cp['channel_versions'])
    def forbidden(*args,**kwargs):raise AssertionError('SQLite forbidden for hosted job')
    import sqlite3
    monkeypatch.setattr(sqlite3,'connect',forbidden)
    with job_runtime(runtime),cli_execution(StopFence(root/'interview/stop-request.json'),None):
        proposal=run_revision_proposal(capability=cap,session_root=root,
            checkpoint_id=cfg['configurable']['checkpoint_id'],phase='srs',change_request='Describe offline behavior',
            validate_authority=lambda:None,log=None,graph_factory=factory(state))
    assert not proposal['committed'] and proposal['requires_fresh_review']
    assert len(savers)==2
    assert backend('spec').get_tuple(cfg).checkpoint['channel_values']==state
    assert not list(root.rglob('*.sqlite'))


def test_factory_needs_bound_root_and_cannot_escape(tmp_path):
    from spec_generator.checkpoint_runtime import checkpoint
    cap,root,runtime,cid,state=setup(tmp_path)
    calls=[];runtime.checkpoint_factory=lambda ns:calls.append(ns)
    with job_runtime(runtime):
        with pytest.raises(ValueError,match='outside'):checkpoint(tmp_path/'foreign')
        with pytest.raises(ValueError,match='outside'):checkpoint(root/'spec'/ '..')
    assert calls==[]


def test_read_state_rejects_wrong_review_phase_without_opening_backend(tmp_path):
    from spec_generator.checkpoint_runtime import read_state
    cap,root,runtime,cid,state=setup(tmp_path)
    calls=[];runtime.checkpoint_factory=lambda ns:calls.append(ns)
    with job_runtime(runtime):
        with pytest.raises(ValueError,match='review phase'):
            read_state(root/'spec',session_id='s',expected_checkpoint_id=cid,expected_review_phase='needs')
    assert calls==[]


def test_bound_proposal_retains_through_host_sink_without_creating_files(tmp_path,monkeypatch):
    from pathlib import Path
    cap,root,runtime,cid,state=setup(tmp_path)
    savers={};retained=[]
    def backend(namespace):
        if namespace not in savers:
            saver=MemorySaver();saver.setup=lambda:None;savers[namespace]=saver
        return savers[namespace]
    runtime.checkpoint_factory=backend;runtime.proposal_sink=retained.append
    cp=empty_checkpoint();cp['channel_values']=deepcopy(state);cp['channel_versions']={k:1 for k in state}
    cfg=backend('spec').put({'configurable':{'thread_id':'s','checkpoint_ns':''}},cp,{'source':'input','step':0,'parents':{}},cp['channel_versions'])
    original=Path.mkdir
    def forbid_proposal(self,*a,**kw):
        if 'revision-proposals' in self.parts:raise AssertionError('proposal files forbidden')
        return original(self,*a,**kw)
    monkeypatch.setattr(Path,'mkdir',forbid_proposal)
    with job_runtime(runtime),cli_execution(StopFence(root/'interview/stop-request.json'),None):
        proposal=run_revision_proposal(capability=cap,session_root=root,checkpoint_id=cfg['configurable']['checkpoint_id'],phase='srs',
            change_request='Describe offline behavior',validate_authority=lambda:None,log=None,graph_factory=factory(state))
    assert retained==[proposal] and not (root/'spec/revision-proposals').exists()
    assert backend('spec').get_tuple(cfg).checkpoint['channel_values']==state


def test_unbound_checkpoint_never_creates_a_local_database(tmp_path):
    from spec_generator.checkpoint_runtime import checkpoint,read_state
    with pytest.raises(ValueError,match='host checkpoint'):
        checkpoint(tmp_path/'spec')
    with pytest.raises(ValueError,match='host checkpoint'):
        read_state(tmp_path/'spec',session_id='s',expected_checkpoint_id='old')
    assert not list(tmp_path.rglob('*'))


def test_standalone_cli_refuses_before_any_command_or_provider_work(monkeypatch):
    from spec_generator import cli
    calls=[]
    monkeypatch.setattr(cli,'_main',lambda *args:calls.append(args))
    with pytest.raises(ValueError,match='delivery-api'):
        cli.main(['unused-intent.json'])
    assert calls==[]
