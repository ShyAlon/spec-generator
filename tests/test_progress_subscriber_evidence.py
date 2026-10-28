import logging
from spec_generator.progress import ProgressBus, ProgressEvent


def test_subscriber_failure_is_counted_safe_bounded_and_nonfatal(caplog, monkeypatch):
    bus = ProgressBus()
    received = []
    def broken(event): raise ValueError('fake-secret-payload')
    bus.subscribe(broken)
    bus.subscribe(received.append)
    event = ProgressEvent('safe_stage', 'private-label', 'started', 0, detail='private-detail')
    with caplog.at_level(logging.WARNING):
        for _ in range(100): bus.emit(event)
    assert len(received) == 100
    assert bus.subscriber_failures == {0: 100}
    records = [r for r in caplog.records if 'progress.subscriber_failed' in r.getMessage()]
    assert len(records) == 1
    message = records[0].getMessage()
    assert 'stage=safe_stage' in message and 'category=ValueError' in message
    assert 'fake-secret' not in message and 'private-' not in message
    assert records[0].exc_info is None


def test_diagnostic_failure_cannot_kill_progress(monkeypatch):
    from spec_generator import progress
    bus = ProgressBus()
    def broken(event): raise RuntimeError('private')
    def log_broken(*args, **kwargs): raise OSError('logger broken')
    monkeypatch.setattr(progress._logger, 'warning', log_broken)
    bus.subscribe(broken)
    received = []
    bus.subscribe(received.append)
    bus.emit(ProgressEvent('stage', 'label', 'started', 0))
    assert len(received) == 1 and bus.subscriber_failures == {0: 1}


def test_warning_resumes_after_one_minute_with_cumulative_count(caplog, monkeypatch):
    from spec_generator import progress
    clock = [0.0]
    monkeypatch.setattr(progress.time, 'monotonic', lambda: clock[0])
    bus = ProgressBus()
    def broken(event): raise ValueError('secret')
    bus.subscribe(broken)
    event = ProgressEvent('stage\nforged line', 'label', 'started', 0)
    with caplog.at_level(logging.WARNING):
        bus.emit(event)
        clock[0] = 59.9
        bus.emit(event)
        clock[0] = 60
        bus.emit(event)
    assert len(caplog.records) == 2
    assert 'count=3' in caplog.records[-1].getMessage()
    assert 'stage=stage_forged_line' in caplog.records[-1].getMessage()
