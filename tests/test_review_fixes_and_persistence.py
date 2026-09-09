"""2026-09-09 live failure + ask: (1) the review-fix application path re-fired
the srs stage against the retired full-document validator while the grammar
constrained the model to fills ('functional: Field required; fills: Extra
inputs are not permitted') - the fix path now re-fires the FILL stage with the
fixes; (2) the bundle persists incrementally, stage by stage, atomically."""
import json

import pytest
from langgraph.checkpoint.memory import MemorySaver

from spec_generator import model_config
from spec_generator.cli import _atomic_write, _make_persist
from spec_generator.graph import build_graph
from spec_generator.telemetry import UsageLog
from test_progress import INTENT, PAYLOADS, FakeModel

FIXES = [{"fix": "R-1 statement too vague - name the emitted file format"}]


class QueueModel(FakeModel):
    """Per-stage payload queue; falls back to PAYLOADS[stage] when drained."""
    QUEUES = {}

    def invoke(self, messages):
        self.calls += 1
        self.last_human = messages[-1].content
        q = self.QUEUES.get(self.stage) or []
        payload = q.pop(0) if q else PAYLOADS[self.stage]
        return type("R", (), {"content": json.dumps(payload)})()


@pytest.fixture
def queue_models(monkeypatch):
    box = {}
    QueueModel.QUEUES = {}
    monkeypatch.setattr(model_config, "build_chat_model", lambda stage, **kw: box.setdefault(stage, QueueModel(stage)))
    return box


def test_review_fixes_refire_the_fill_stage(queue_models):
    QueueModel.QUEUES["review"] = [FIXES, []]  # round 1 flags, round 2 clean
    g = build_graph(UsageLog("s"), checkpointer=MemorySaver())
    r = g.invoke({"intent_model": dict(INTENT)}, {"configurable": {"thread_id": "rf1"}})
    assert "bundle" in r
    srs = queue_models["srs"]
    assert srs.calls == 2  # initial author + one fix application
    apply_payload = srs.last_human
    assert "REVIEW FIXES" in apply_payload and FIXES[0]["fix"] in apply_payload
    assert "REQUIREMENT SLOTS" in apply_payload  # fill shape, not the retired full-document ask
    # the fix path went through the skeleton merge validator: downstream stages
    # re-ran against the corrected SRS and the pipeline completed
    assert queue_models["items"].calls == 2 and queue_models["tests"].calls == 2  # re-derived from the fixed SRS
    assert queue_models["review"].calls == 2


def test_clean_review_never_pays_the_fix_stage(queue_models):
    g = build_graph(UsageLog("s"), checkpointer=MemorySaver())
    g.invoke({"intent_model": dict(INTENT)}, {"configurable": {"thread_id": "rf2"}})
    assert queue_models["srs"].calls == 1
    assert queue_models["review"].calls == 1  # clean: stops before pass 2


def test_persist_hook_fires_per_stage_in_order(queue_models):
    fired = []
    g = build_graph(UsageLog("s"), checkpointer=MemorySaver(),
                    persist=lambda stage, data, state: fired.append(stage))
    g.invoke({"intent_model": dict(INTENT)}, {"configurable": {"thread_id": "p1"}})
    assert fired == ["needs", "srs", "architecture", "items", "tests"]


def test_incremental_bundle_on_disk_and_atomic(tmp_path, queue_models):
    persist = _make_persist(dict(INTENT), tmp_path)
    g = build_graph(UsageLog("s"), checkpointer=MemorySaver(), persist=persist)
    g.invoke({"intent_model": dict(INTENT)}, {"configurable": {"thread_id": "p2"}})
    for f in ("needs.json", "needs.md", "srs.json", "srs.md",
              "architecture.json", "architecture.md", "items.json", "tests.json"):
        assert (tmp_path / f).exists(), f
    assert (tmp_path / "specs" / "SPEC-api.md").exists()
    assert "Driver:" in (tmp_path / "architecture.md").read_text()  # rendered, not raw JSON
    assert not list(tmp_path.glob("*.tmp")) and not list((tmp_path / "specs").glob("*.tmp"))


def test_late_failure_leaves_completed_stages_on_disk(tmp_path, monkeypatch):
    class DieAtTests(QueueModel):
        def invoke(self, messages):
            if self.stage == "tests":
                raise RuntimeError("process died")
            return super().invoke(messages)

    box = {}
    monkeypatch.setattr(model_config, "build_chat_model", lambda stage, **kw: box.setdefault(stage, DieAtTests(stage)))
    persist = _make_persist(dict(INTENT), tmp_path)
    g = build_graph(UsageLog("s"), checkpointer=MemorySaver(), persist=persist)
    with pytest.raises(RuntimeError):
        g.invoke({"intent_model": dict(INTENT)}, {"configurable": {"thread_id": "p3"}})
    for f in ("needs.json", "srs.json", "architecture.json", "items.json", "specs/SPEC-api.md"):
        assert (tmp_path / f).exists(), f
    assert not (tmp_path / "tests.json").exists()  # the failed stage wrote nothing


def test_atomic_write_leaves_no_tmp(tmp_path):
    _atomic_write(tmp_path / "x.json", '{"a": 1}')
    assert (tmp_path / "x.json").read_text() == '{"a": 1}'
    assert not (tmp_path / "x.json.tmp").exists()
