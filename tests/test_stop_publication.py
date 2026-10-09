import threading
import pytest
from spec_generator.stop_fence import StopFence,StopRequested
from spec_generator.stop_publication import publication_guard,FencedSink,FencedCheckpointer
from langgraph.checkpoint.memory import MemorySaver


def test_sink_rejects_late_write_preserves_read(tmp_path):
    f=StopFence(tmp_path/'stop.json');saved=[]
    class Sink:
        def save(self,value):saved.append(value)
        def load(self):return list(saved)
    with f.running() as gen:
        sink=FencedSink(Sink(),publication_guard(f,gen))
        sink.save('earlier');f.request_stop()
        with pytest.raises(StopRequested):sink.save('late')
        assert sink.load()==['earlier']
    assert f.state=='STOPPED'


def test_checkpoint_rejects_write_without_context_in_new_thread(tmp_path):
    f=StopFence(tmp_path/'stop.json');errors=[]
    with f.running() as gen:
        cp=FencedCheckpointer(MemorySaver(),publication_guard(f,gen))
        f.request_stop()
        def worker():
            try:cp.put_writes({},[('result','late')],'task')
            except Exception as exc:errors.append(exc)
        t=threading.Thread(target=worker);t.start();t.join()
    assert len(errors)==1 and isinstance(errors[0],StopRequested)


def test_deadline_surviving_thread_holds_quiescence(tmp_path):
    from spec_generator.model_config import _run_with_deadline
    f=StopFence(tmp_path/'stop.json');entered=threading.Event();finish=threading.Event();exited=threading.Event()
    def accepted():
        entered.set();finish.wait(3);exited.set()
    with f.running():
        with pytest.raises(TimeoutError):_run_with_deadline(accepted,.03,'fake')
        assert entered.is_set();f.request_stop()
    assert f.state=='STOP_REQUESTED'
    finish.set();assert exited.wait(2)
    # wait for the wrapper's finally, not merely the provider's last statement
    for _ in range(100):
        if f.state=='STOPPED':break
        threading.Event().wait(.005)
    assert f.state=='STOPPED'


def test_cli_partial_clear_and_lifecycle_mutations_stop_atomically(tmp_path):
    from spec_generator.cooperative_run import cli_execution
    from spec_generator.cli import _make_partials
    from spec_generator.artifact_lifecycle import ArchitectureLifecycle
    f=StopFence(tmp_path/'stop.json');partials=_make_partials(tmp_path)
    with cli_execution(f,None):
        partials.save('items',[{'saved':True}])
        lifecycle=ArchitectureLifecycle(tmp_path,'fake',{})
        f.request_stop()
        with pytest.raises(StopRequested):partials.clear('items')
        with pytest.raises(StopRequested):lifecycle.bootstrap_complete({'prior':True},{})
    assert partials.load('items')==[{'saved':True}]
    assert not (tmp_path/'artifact-lifecycle.jsonl').exists()


def test_accepted_local_stream_drains_bills_once_discards_and_blocks_next(tmp_path):
    from spec_generator.attempt_admission import admitted_stream
    from spec_generator.stop_fence import stopped_usage_sink
    from types import SimpleNamespace
    f=StopFence(tmp_path/'stop.json');billed=[];calls=[];delivered=[]
    def accepted():
        calls.append(1)
        yield SimpleNamespace(content='early',usage_metadata=None)
        f.request_stop()
        yield SimpleNamespace(content='late',usage_metadata=None)
        yield SimpleNamespace(content='',usage_metadata={'input_tokens':4,'output_tokens':2})
    options=dict(stage='fake',provider='fake',model='fake',messages=[],max_output_tokens=10)
    with f.running(),stopped_usage_sink(lambda response,**kw:billed.append(response.usage_metadata)):
        with pytest.raises(StopRequested):
            for chunk in admitted_stream(accepted,**options):delivered.append(chunk.content)
        with pytest.raises(StopRequested):list(admitted_stream(accepted,**options))
    assert delivered==['early'] and calls==[1]
    assert billed==[{'input_tokens':4,'output_tokens':2}] and f.state=='STOPPED'


def test_stop_blocks_progress_and_raw_capture(tmp_path,monkeypatch):
    from spec_generator.progress import ProgressBus,ProgressEvent
    from spec_generator.model_config import log_raw_completion
    f=StopFence(tmp_path/'stop.json');events=[];bus=ProgressBus();bus.subscribe(events.append)
    monkeypatch.setenv('SPEC_GENERATOR_LOG_RAW','1')
    monkeypatch.setenv('VMODEL_LLM_LOG',str(tmp_path/'raw.jsonl'))
    with f.running():
        f.request_stop();bus.emit(ProgressEvent('fake','Fake','completed',1))
        with pytest.raises(StopRequested):log_raw_completion('fake',None,1,'discarded')
    assert events==[] and not (tmp_path/'raw.jsonl').exists()


def test_execution_adapter_accounts_stopped_invoke_and_never_restarts(tmp_path):
    from spec_generator.cooperative_run import producer_execution
    from spec_generator.attempt_admission import admitted_invoke
    from types import SimpleNamespace
    f=StopFence(tmp_path/'stop.json');records=[]
    class Log:
        def record(self,*args):records.append(args)
    def accepted():
        f.request_stop()
        return SimpleNamespace(usage_metadata={'input_tokens':3,'output_tokens':2})
    options=dict(stage='fake',provider='fake',model='fake',messages=[],max_output_tokens=5)
    with producer_execution(f,Log()):
        with pytest.raises(StopRequested):admitted_invoke(accepted,**options)
    assert records==[('fake','fake','fake',{'input_tokens':3,'output_tokens':2,'total_tokens':5})]
    with pytest.raises(StopRequested):
        with producer_execution(f,Log()):pass


def test_real_graph_checkpoint_never_publishes_old_result(tmp_path):
    from langgraph.graph import StateGraph,START,END
    from typing import TypedDict
    f=StopFence(tmp_path/'stop.json');raw=MemorySaver()
    class State(TypedDict):result:str
    def node(state):f.request_stop();return {'result':'discarded'}
    graph=StateGraph(State);graph.add_node('accepted',node);graph.add_edge(START,'accepted');graph.add_edge('accepted',END)
    with f.running() as gen:
        cp=FencedCheckpointer(raw,publication_guard(f,gen))
        compiled=graph.compile(checkpointer=cp)
        with pytest.raises(StopRequested):compiled.invoke({'result':'prior'},{'configurable':{'thread_id':'fake'}})
    checkpoints=list(raw.list({'configurable':{'thread_id':'fake'}}))
    assert checkpoints and all(c.checkpoint['channel_values'].get('result')!='discarded' for c in checkpoints)


def test_cli_atomic_docs_and_audit_reject_stopped_generation(tmp_path):
    from spec_generator.cooperative_run import cli_execution,cli_stop_binding
    from spec_generator.cli import _atomic_write,_append_audit
    from spec_generator.docs import refresh_docs
    f=StopFence(tmp_path/'stop.json')
    with cli_execution(f,None) as gen:
        assert cli_stop_binding()==(f,gen)
        _atomic_write(tmp_path/'prior.txt','saved')
        f.request_stop()
        for write in (lambda:_atomic_write(tmp_path/'late.txt','late'),
                      lambda:_append_audit(tmp_path/'audit.jsonl',{'late':True}),
                      lambda:refresh_docs(tmp_path/'spec',start=True)):
            with pytest.raises(StopRequested):write()
    assert cli_stop_binding() is None
    assert (tmp_path/'prior.txt').read_text()=='saved'
    assert not (tmp_path/'late.txt').exists() and not (tmp_path/'audit.jsonl').exists()
    assert not (tmp_path/'spec').exists()
