import pytest
from spec_generator.child_execution import graph_execution,broker_call_id,invocation


def test_checkpoint_cursor_replay_and_new_checkpoint():
    class Graph:
        checkpoint='checkpoint-1'
        def get_state(self,cfg):
            return type('State',(),{'config':{'configurable':{'checkpoint_id':self.checkpoint}}})()
    graph=Graph()
    def call():
        with invocation('srs'):return broker_call_id('srs')
    with pytest.raises(PermissionError):call()
    with graph_execution(graph,{'configurable':{'thread_id':'session'}}):
        a=call();assert call()!=a
    with graph_execution(graph,{'configurable':{'thread_id':'session'}}):assert call()==a
    graph.checkpoint='checkpoint-2'
    with graph_execution(graph,{'configurable':{'thread_id':'session'}}):assert call()!=a


def test_gate_suspends_without_process_local_answer_and_reuses_durable_answer():
    from spec_generator.child_execution import DurableGate,GateSuspended,gate_key
    decision={'decision_id':'srs-review','prompt':'Approve?'};published=[]
    with pytest.raises(GateSuspended) as suspended:DurableGate({},published.append)(decision)
    assert published==[{'kind':'gate','key':gate_key(decision),'decision':decision}]
    assert suspended.value.key==gate_key(decision)
    assert DurableGate({gate_key(decision):'a'},published.append)(decision)=='a'


def test_real_cli_gate_suspension_then_fresh_graph_restore(tmp_path):
    import io
    from langgraph.graph import StateGraph,START,END
    from langgraph.types import interrupt,Command
    from reference_checkpoint import reference_checkpoint
    from spec_generator.child_execution import DurableGate,GateSuspended
    from spec_generator.cli import _run_with_gates
    from spec_generator.cli_io import bind,RunIO
    # Actual CLI gate loop with a fresh graph over the same reference saver. Renderer is real and
    # uses contract-shaped SRS, not a process-local fake answer queue.
    from test_progress import INTENT,CONTRACT_SRS
    def gate(state):
        interrupt({'gate':'srs','artifact':CONTRACT_SRS,'advisories':[]})
        return {'done':True}
    def graph(key):
        builder=StateGraph(dict);builder.add_node('gate',gate);builder.add_edge(START,'gate');builder.add_edge('gate',END)
        return builder.compile(checkpointer=reference_checkpoint(key))
    cfg={'configurable':{'thread_id':'gate-session'}};events=[]
    key=tmp_path/'gate-reference'
    with bind(RunIO(io.StringIO(),io.StringIO(),DurableGate({},events.append))):
        with pytest.raises(GateSuspended):_run_with_gates(graph(key),cfg,dict(INTENT),tmp_path)
    assert events and events[0]['kind']=='gate'
    key=tmp_path/'gate-reference'
    with bind(RunIO(io.StringIO(),io.StringIO(),DurableGate({events[0]['key']:'a'},events.append))):
        result=_run_with_gates(graph(key),cfg,dict(INTENT),tmp_path,resume=True)
    assert result['done'] is True and len(events)==1


def test_job_cursor_survives_changed_latest_checkpoint_after_crash(monkeypatch):
    from spec_generator import child_execution as module
    from spec_generator.worker_runtime import JobRuntime,job_runtime
    from spec_generator.model_config import StageModel
    from langchain_core.runnables.config import var_child_runnable_config
    class Graph:
        checkpoint='first'
        def get_state(self,cfg):return type('State',(),{'config':{'configurable':{'checkpoint_id':self.checkpoint}}})()
    graph=Graph();runtime=JobRuntime({'srs':StageModel('ollama','fake')},'/tmp/owned/log',lambda s:None)
    runtime.bind_worker(owner_id='owner',task_id='task',job_id='job-1',generation=1,session_root='/tmp/owned')
    token=var_child_runnable_config.set({'configurable':{'checkpoint_ns':'srs:task','__pregel_scratchpad':type('Scratch',(),{'step':3})()}})
    try:
        with job_runtime(runtime),graph_execution(graph,{'configurable':{'thread_id':'session'}}):
            with invocation('srs'):first=broker_call_id('srs')
        graph.checkpoint='newest-orphan-after-crash'
        with job_runtime(runtime),graph_execution(graph,{'configurable':{'thread_id':'session'}}):
            with invocation('srs'):assert broker_call_id('srs')==first
    finally:var_child_runnable_config.reset(token)


def test_gate_identity_changes_when_review_artifact_changes():
    from spec_generator.child_execution import gate_key
    a={'decision_id':'srs-review','context':{'artifact_sha256':'a'*64}}
    b={'decision_id':'srs-review','context':{'artifact_sha256':'b'*64}}
    assert gate_key(a)!=gate_key(b)


def test_item_metadata_is_optional_and_does_not_rekey_a_provider_call():
 from spec_generator.child_execution import invocation,broker_call_id,broker_call_metadata,graph_execution
 from types import SimpleNamespace
 class Graph:
  def get_state(self,cfg):return SimpleNamespace(config={'configurable':{'checkpoint_id':'point'}})
 cfg={'configurable':{'thread_id':'session'}}
 ids=[]
 for label in ('ADR-1',None):
  with graph_execution(Graph(),cfg):
   with invocation('architecture',label):
    ids.append(broker_call_id('architecture'))
    assert broker_call_metadata('architecture')==({'item':label,'fields':[]} if label else None)
 assert ids[0]==ids[1]
 assert broker_call_metadata('architecture') is None
