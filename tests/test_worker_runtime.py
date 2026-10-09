from concurrent.futures import ThreadPoolExecutor
import json,threading
import pytest
from spec_generator.worker_runtime import JobRuntime,job_runtime,current_runtime
from spec_generator import model_config as m


def test_parallel_jobs_do_not_share_models_budget_diagnostics_logs_or_console(tmp_path):
    barrier=threading.Barrier(2);outputs={}
    def run(owner):
        lines=[]
        runtime=JobRuntime({'needs':m.StageModel('fake',owner)},str(tmp_path/(owner+'.jsonl')),lines.append,
            settings={'VMODEL_LLM_MAX_CALLS':'1','VMODEL_LLM_HEARTBEAT_S':'0'})
        with job_runtime(runtime):
            barrier.wait(2)
            m._guard_enter('needs')
            m._call_diagnostics()[('needs',None,1)]={'owner':owner}
            m._log_llm({'owner':owner});m._console_line(owner)
            assert m.for_stage('needs').model==owner
            with pytest.raises(KeyError):m.for_stage('srs')
            with pytest.raises(Exception):m._guard_enter('needs')
            assert m._run_with_deadline(lambda:m.for_stage('needs').model,1,'needs')==owner
            assert m._watchdog() is runtime.watchdog
        outputs[owner]=(lines,runtime)
        assert current_runtime() is None
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(run,['owner-a','owner-b']))
    for owner,(lines,runtime) in outputs.items():
        assert lines==[owner] and runtime.calls==2
        assert runtime.diagnostics=={('needs',None,1):{'owner':owner}}
        assert json.loads((tmp_path/(owner+'.jsonl')).read_text())['owner']==owner


def test_heartbeat_inherits_only_its_job_context(tmp_path):
    lines=[];signal=threading.Event()
    class Sink:
        _enabled=True
        def emit(self,*args):signal.set()
    runtime=JobRuntime({'needs':m.StageModel('fake','owner')},str(tmp_path/'owner.jsonl'),lines.append,
        progress=Sink(),settings={'VMODEL_LLM_HEARTBEAT_S':'.01'})
    with job_runtime(runtime):stop=m._start_heartbeat('private job')
    assert signal.wait(2);stop()
    assert any('private job' in line for line in lines)


def test_private_runtime_does_not_inherit_experiment_caps_replay_or_raw_capture(tmp_path,monkeypatch):
    for key,value in {'VMODEL_CAP_NEEDS':'1','VMODEL_REPLAY_DIR':'/not-this-job','SPEC_GENERATOR_BULK':'1','SPEC_GENERATOR_LOG_RAW':'1','SPEC_GENERATOR_FROM':'srs'}.items():monkeypatch.setenv(key,value)
    with job_runtime(JobRuntime({'needs':m.StageModel('fake','fake')},str(tmp_path/'calls.jsonl'),lambda text:None)):
        assert m._env('VMODEL_CAP_NEEDS') is None
        assert m._env('VMODEL_REPLAY_DIR') is None
        assert not m.bulk_fills_enabled() and not m.raw_capture_enabled()
        assert 'WITHOUT' not in m._rerun_guidance('needs')


def test_worker_runtime_binding_cannot_be_lent_to_another_job_or_owner(tmp_path):
    root=tmp_path/'owned';root.mkdir()
    runtime=JobRuntime({'needs':m.StageModel('fake','fake')},str(root/'calls.jsonl'),lambda text:None)
    binding=dict(owner_id='a',task_id='t',job_id='j',generation=1,session_root=root)
    runtime.bind_worker(**binding);runtime.bind_worker(**binding)
    for change in ({'owner_id':'b'},{'task_id':'other'},{'job_id':'new'},{'generation':2}):
        with pytest.raises(ValueError,match='another worker job'):runtime.bind_worker(**(binding|change))


@pytest.mark.parametrize('unsafe',['outside','relative','symlink'])
def test_worker_runtime_telemetry_cannot_escape_owned_session(tmp_path,unsafe):
    root=tmp_path/'owned';root.mkdir();outside=tmp_path/'outside.jsonl'
    sink=outside if unsafe=='outside' else 'calls.jsonl' if unsafe=='relative' else root/'calls.jsonl'
    if unsafe=='symlink':sink.symlink_to(outside)
    runtime=JobRuntime({'needs':m.StageModel('fake','fake')},str(sink),lambda text:None)
    with pytest.raises(ValueError,match='owned runtime log'):
        runtime.bind_worker(owner_id='a',task_id='t',job_id='j',generation=1,session_root=root)
    assert runtime.worker_binding is None and not outside.exists()
