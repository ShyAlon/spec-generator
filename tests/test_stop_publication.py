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
