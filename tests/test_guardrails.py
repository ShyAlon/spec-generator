"""Guardrails around every LLM call: run-wide call budget, per-call wall-clock
deadline, stream stall limit - informative kills, never silent hangs. Plus the
per-item start/end instrumentation in the progress stream."""
import re
import time

import pytest

import spec_generator.model_config as mc
from spec_generator.errors import UserFacingError
from spec_generator.progress import ProgressBus, ProgressEvent, stderr_printer
from spec_generator.stream_items import ItemReporter


class FakeModel:
    def __init__(self, text="ok", invoke_sleep_s=0.0, chunks=(), stall_s=0.0):
        self._text, self._invoke_sleep = text, invoke_sleep_s
        self._chunks, self._stall = chunks, stall_s

    def invoke(self, messages):
        time.sleep(self._invoke_sleep)
        return self._text

    def stream(self, messages):
        for c in self._chunks:
            yield c
        if self._stall:
            time.sleep(self._stall)
            yield type("C", (), {"content": "tail"})()


@pytest.fixture(autouse=True)
def _clean_guard_env(monkeypatch):
    monkeypatch.setattr(mc, "_CALLS_USED", 0)
    for v in ("VMODEL_LLM_CALL_TIMEOUT_S", "VMODEL_LLM_STALL_TIMEOUT_S", "VMODEL_LLM_MAX_CALLS"):
        monkeypatch.delenv(v, raising=False)


def test_invoke_under_deadline_returns_normally():
    assert mc.invoke_checked(FakeModel("done"), [], "srs") == "done"


def test_slow_invoke_is_killed_with_an_informative_error(monkeypatch):
    monkeypatch.setenv("VMODEL_LLM_CALL_TIMEOUT_S", "1")
    with pytest.raises(UserFacingError) as ei:
        mc.invoke_checked(FakeModel(invoke_sleep_s=5), [], "srs")
    err = ei.value.render()
    assert "killed" in err and "'srs'" in err
    assert "ollama" in err and "qwen2.5-coder:14b" in err  # stage + provider + model named
    assert "rerun the same command" in err


def test_run_call_budget_trips_with_an_informative_error(monkeypatch):
    monkeypatch.setenv("VMODEL_LLM_MAX_CALLS", "2")
    mc.invoke_checked(FakeModel(), [], "srs")
    mc.invoke_checked(FakeModel(), [], "srs")
    with pytest.raises(UserFacingError) as ei:
        mc.invoke_checked(FakeModel(), [], "srs")
    err = ei.value.render()
    assert "budget" in err and "2 calls" in err and "VMODEL_LLM_MAX_CALLS" in err


def test_stream_passthrough_unchanged():
    chunks = [type("C", (), {"content": t})() for t in ('{"a": ', "1}")]
    out = "".join(c.content for c in mc.stream_checked(FakeModel(chunks=chunks), [], "srs"))
    assert out == '{"a": 1}'


def test_stalled_stream_is_killed_with_an_informative_error(monkeypatch):
    monkeypatch.setenv("VMODEL_LLM_STALL_TIMEOUT_S", "1")
    monkeypatch.setenv("VMODEL_LLM_CALL_TIMEOUT_S", "60")
    chunks = [type("C", (), {"content": '{"a":'})()]
    gen = mc.stream_checked(FakeModel(chunks=chunks, stall_s=5), [], "items")
    assert next(gen).content == '{"a":'  # first chunk flows through
    with pytest.raises(UserFacingError) as ei:
        next(gen)
    err = ei.value.render()
    assert "killed" in err and "stall" in err and "'items'" in err


def test_item_reporter_carries_start_and_end():
    bus = ProgressBus()
    events = []
    bus.subscribe(events.append)
    r = ItemReporter(bus, "needs")
    time.sleep(0.05)
    r.feed('[{"need_id": "R-1", "title": "first"}, ')
    time.sleep(0.05)
    r.feed('[{"need_id": "R-1", "title": "first"}, {"need_id": "R-2", "title": "second"}]')
    items = [e for e in events if e.status == "item"]
    assert len(items) == 2
    assert all(e.elapsed_s is not None and e.elapsed_s >= 0 for e in items)
    assert items[0].ts <= items[1].ts
    assert items[1].ts - items[1].elapsed_s >= items[0].ts - 1e-6  # item 2 starts when item 1 ended


def test_printer_renders_item_timestamps(capsys):
    bus = ProgressBus()
    bus.subscribe(stderr_printer())
    end = time.time()
    bus.emit(ProgressEvent("items", "", "item", end, 3.2, "SPEC-gateway Set up Kong API gateway"))
    out = capsys.readouterr().err
    assert re.search(r"\[\d{2}:\d{2}:\d{2}-\d{2}:\d{2}:\d{2}\] SPEC-gateway Set up Kong API gateway", out)
    # events without timing (retry notices) keep the old plain line
    bus.emit(ProgressEvent("srs", "", "item", end, None, "model output was not valid JSON - retrying once"))
    assert "- model output was not valid JSON - retrying once" in capsys.readouterr().err
