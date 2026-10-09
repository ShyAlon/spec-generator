from concurrent.futures import ThreadPoolExecutor
import threading
import pytest
from spec_generator.stop_fence import StopFence,StopRequested,check_execution


def test_stop_does_not_wait_on_call_and_late_result_is_fenced(tmp_path):
    fence=StopFence(tmp_path/'stop.json');entered=threading.Event();release=threading.Event();calls=[];published=[]
    def work():
        with fence.running() as generation:
            calls.append('accepted');entered.set();release.wait(2)
            calls.append('usage recorded')
            fence.check(generation);published.append('late result')
    with ThreadPoolExecutor() as pool:
        future=pool.submit(work);assert entered.wait(1)
        assert fence.request_stop()=='STOP_REQUESTED'
        assert fence.state=='STOP_REQUESTED' and fence.active==1
        assert StopFence(fence.path).state=='STOP_REQUESTED'  # shared same-host writer remains active
        release.set()
        with pytest.raises(StopRequested):future.result()
    assert calls==['accepted','usage recorded'] and published==[] and fence.state=='STOPPED'
    with pytest.raises(StopRequested):
        with fence.running():pass


def test_stop_replay_does_not_increment_generation_and_no_auto_resume(tmp_path):
    f=StopFence(tmp_path/'stop.json');assert f.request_stop()=='STOPPED'
    generation=f.generation;f.request_stop();assert f.generation==generation
    restored=StopFence(f.path)
    assert restored.requested and restored.generation==generation
    with pytest.raises(StopRequested):
        with restored.running():pass


def test_next_provider_check_blocks_but_settlement_can_finish(tmp_path):
    f=StopFence(tmp_path/'stop.json')
    with f.running():
        check_execution();f.request_stop()
        # Existing accepted response usage settlement does not call this check.
        with pytest.raises(StopRequested):check_execution()
