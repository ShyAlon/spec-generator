from spec_generator.structured_progress import StructuredProgress,PREFIX
from vmodel_contracts.progress_signal import ProgressSignal

def test_structured_frame_uses_canonical_phase_and_order(capsys,monkeypatch):
    monkeypatch.setenv("VMODEL_PROGRESS_TASK_ID","task")
    monkeypatch.setenv("VMODEL_PROGRESS_SESSION_ID","session")
    sink=StructuredProgress('task','session','specs')
    sink.emit('items','Writing items','started')
    sink.emit('items','Writing items','heartbeat','Model call still running')
    rows=capsys.readouterr().out.splitlines()
    a,b=[ProgressSignal.model_validate_json(row.removeprefix(PREFIX)) for row in rows]
    assert all(row.startswith(PREFIX) for row in rows)
    assert (a.phase_key,b.phase_key)==('specs','specs')
    assert (a.cursor,b.cursor)==(1,2)
    assert b.status=='heartbeat'


def test_every_generator_stage_reports_under_its_own_phase():
    from spec_generator.structured_progress import STAGE_PHASE
    assert STAGE_PHASE['validation']=='validation' and STAGE_PHASE['tests']=='tests'
    sink=StructuredProgress('task','session',STAGE_PHASE['validation'])
    import io,contextlib
    buf=io.StringIO()
    with contextlib.redirect_stdout(buf):sink.emit('validation','Writing validation items','started')
    signal=ProgressSignal.model_validate_json(buf.getvalue().strip().removeprefix(PREFIX))
    assert signal.phase_key=='validation'
