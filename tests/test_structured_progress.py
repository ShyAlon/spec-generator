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
